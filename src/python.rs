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
    ///   method:   "approx" (default), "exact", or "exact_tree".
    ///   n_samples: number of samples for approximate method (default 10_000).
    ///   seed:     RNG seed for reproducibility (default None = random).
    ///
    /// Returns:
    ///   dict[str, float] mapping node name to its ASV value.
    #[pyo3(signature = (value_fn, method = "approx", n_samples = 10_000, seed = None))]
    fn explain(
        &self,
        py: Python<'_>,
        value_fn: PyObject,
        method: &str,
        n_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<HashMap<String, f64>> {
        let names = &self.names;
        let rust_fn = |coalition: &[NodeId]| -> Result<f64, CausasvError> {
            let name_list: Vec<&str> = coalition
                .iter()
                .map(|id| names[id.0 as usize].as_str())
                .collect();
            value_fn
                .call1(py, (name_list,))
                .and_then(|r| r.extract::<f64>(py))
                .map_err(|e| CausasvError::ValueFunctionError(e.to_string()))
        };

        let result = match method {
            "approx" => {
                let mut cfg = SamplingConfig::new(n_samples);
                if let Some(s) = seed {
                    cfg = cfg.with_seed(s);
                }
                self.inner.approximate(rust_fn, cfg)
            }
            "exact" => self.inner.exact(rust_fn),
            "exact_tree" => self.inner.exact_tree(rust_fn),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown method '{method}': use 'approx', 'exact', or 'exact_tree'"
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
