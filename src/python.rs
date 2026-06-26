use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::asv::AsvExplainer;
use crate::error::CausasvError;
use crate::graph::{Dag as RustDag, NodeId};
use crate::sampler::SamplingConfig;

#[pyclass(name = "CausalDAG")]
pub struct PyCausalDAG {
    pub(crate) inner: RustDag,
}

#[pymethods]
impl PyCausalDAG {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustDag::new(),
        }
    }

    /// Construct a DAG from any graph object that has an `edges()` method returning
    /// (from, to) pairs — compatible with networkx DiGraph and similar libraries.
    #[staticmethod]
    fn from_networkx(g: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut inner = RustDag::new();
        for edge in g.call_method0("edges")?.try_iter()? {
            let edge = edge?;
            let from: String = edge.get_item(0)?.extract()?;
            let to: String = edge.get_item(1)?.extract()?;
            let from_id = inner.add_node(&from);
            let to_id = inner.add_node(&to);
            inner.add_edge(from_id, to_id).map_err(py_err)?;
        }
        Ok(Self { inner })
    }

    /// Construct a DAG from a list of (from, to) edge tuples.
    /// Nodes are created automatically.
    #[staticmethod]
    fn from_edges(edges: Vec<(String, String)>) -> PyResult<Self> {
        let mut inner = RustDag::new();
        for (from, to) in &edges {
            let from_id = inner.add_node(from);
            let to_id = inner.add_node(to);
            inner.add_edge(from_id, to_id).map_err(py_err)?;
        }
        Ok(Self { inner })
    }

    /// Add a directed edge. Nodes are created automatically if they do not exist.
    fn add_edge(&mut self, from_name: &str, to_name: &str) -> PyResult<()> {
        let from = self.inner.add_node(from_name);
        let to = self.inner.add_node(to_name);
        self.inner.add_edge(from, to).map_err(py_err)
    }

    /// Validate the graph (check for cycles, empty graph, etc.).
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(py_err)
    }
}

#[pyclass(name = "ASVExplainer")]
pub struct PyASVExplainer {
    inner: AsvExplainer,
    /// NodeId index → node name, used to bridge coalitions to Python strings.
    names: Vec<String>,
}

#[pymethods]
impl PyASVExplainer {
    /// Create an explainer from a CausalDAG. Validates the graph immediately.
    #[new]
    fn new(dag: &PyCausalDAG) -> PyResult<Self> {
        dag.inner.validate().map_err(py_err)?;
        let names: Vec<String> = dag
            .inner
            .all_nodes()
            .map(|id| dag.inner.node_name(id).unwrap().to_string())
            .collect();
        Ok(Self {
            inner: AsvExplainer::new(dag.inner.clone()),
            names,
        })
    }

    /// Compute ASV values.
    ///
    /// Args:
    ///   value_fn: callable (list[str]) -> float.
    ///             Receives a sorted list of node names in the coalition.
    ///   method:   "auto" (default), "approx", "exact", or "exact_tree".
    ///             "auto" selects exact for n≤8, exact_tree for rooted trees, approx otherwise.
    ///   n_samples: number of samples for approximate method (default 10_000).
    ///   seed:     RNG seed for reproducibility (default None = random).
    ///
    /// Returns:
    ///   dict[str, float] mapping node name to its ASV value.
    #[pyo3(signature = (value_fn, method = "auto", n_samples = 10_000, seed = None))]
    fn explain(
        &self,
        _py: Python<'_>,
        value_fn: PyObject,
        method: &str,
        n_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<HashMap<String, f64>> {
        let names = &self.names;
        // `move` + `Python::with_gil` makes the closure Send + Sync (required for parallel approx).
        // GIL is re-acquired per call; threads serialize on it, so Python users see no speedup
        // but the code stays correct.
        let rust_fn = move |coalition: &[NodeId]| -> Result<f64, CausasvError> {
            Python::with_gil(|py| {
                let name_list: Vec<&str> = coalition
                    .iter()
                    .map(|id| names[id.0 as usize].as_str())
                    .collect();
                value_fn
                    .call1(py, (name_list,))
                    .and_then(|r| r.extract::<f64>(py))
                    .map_err(|e| CausasvError::ValueFunctionError(e.to_string()))
            })
        };

        let make_cfg = || {
            let mut cfg = SamplingConfig::new(n_samples);
            if let Some(s) = seed {
                cfg = cfg.with_seed(s);
            }
            cfg
        };

        let result = match method {
            "auto" => self.inner.auto(rust_fn, make_cfg()),
            "approx" => self.inner.approximate(rust_fn, make_cfg()),
            "exact" => self.inner.exact(rust_fn),
            "exact_tree" => self.inner.exact_tree(rust_fn),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown method '{method}': use 'auto', 'approx', 'exact', or 'exact_tree'"
                )))
            }
        }
        .map_err(py_err)?;

        Ok(result
            .values
            .iter()
            .map(|(id, &v)| (names[id.0 as usize].clone(), v))
            .collect())
    }
}

#[pymodule]
fn causasv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCausalDAG>()?;
    m.add_class::<PyASVExplainer>()?;
    Ok(())
}

fn py_err(e: CausasvError) -> PyErr {
    PyValueError::new_err(e.to_string())
}
