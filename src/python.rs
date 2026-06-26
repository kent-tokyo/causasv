use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::asv::{AsvExplainer, AsvResult};
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

    /// Return all node names in insertion order.
    fn nodes(&self) -> Vec<String> {
        self.inner
            .all_nodes()
            .map(|id| self.inner.node_name(id).unwrap().to_string())
            .collect()
    }

    /// Return all edges as (from, to) name pairs.
    fn edges(&self) -> Vec<(String, String)> {
        self.inner
            .all_nodes()
            .flat_map(|from_id| {
                let from = self.inner.node_name(from_id).unwrap().to_string();
                self.inner
                    .children_raw(from_id)
                    .iter()
                    .map(move |&to_id| {
                        (
                            from.clone(),
                            self.inner.node_name(to_id).unwrap().to_string(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Return a Graphviz DOT representation of the DAG.
    fn to_dot(&self) -> String {
        let mut s = String::from("digraph {\n");
        for from_id in self.inner.all_nodes() {
            let from = self.inner.node_name(from_id).unwrap();
            for &to_id in self.inner.children_raw(from_id) {
                let to = self.inner.node_name(to_id).unwrap();
                s.push_str(&format!("  {from} -> {to};\n"));
            }
        }
        s.push('}');
        s
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

impl PyASVExplainer {
    /// Shared computation logic for both explain() and explain_with_diagnostics().
    fn run(
        &self,
        value_fn: Py<PyAny>,
        method: &str,
        n_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<AsvResult> {
        let names = &self.names;
        let rust_fn = move |coalition: &[NodeId]| -> Result<f64, CausasvError> {
            Python::attach(|py| {
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
        match method {
            "auto" => self.inner.auto(rust_fn, make_cfg()),
            "approx" => self.inner.approximate(rust_fn, make_cfg()),
            "exact" => self.inner.exact(rust_fn),
            "exact_tree" => self.inner.exact_tree(rust_fn),
            "exact_dag" => self.inner.exact_dag(rust_fn),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown method '{method}': use 'auto', 'approx', 'exact', 'exact_tree', or 'exact_dag'"
                )));
            }
        }
        .map_err(py_err)
    }

    fn values_map(&self, result: &AsvResult) -> HashMap<String, f64> {
        result
            .values
            .iter()
            .map(|(id, &v)| (self.names[id.0 as usize].clone(), v))
            .collect()
    }
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
    /// Returns a `dict[str, float]` mapping node name to its ASV value.
    /// For approximate methods, use `explain_with_diagnostics()` to also get ESS.
    #[pyo3(signature = (value_fn, method = "auto", n_samples = 10_000, seed = None))]
    fn explain(
        &self,
        _py: Python<'_>,
        value_fn: Py<PyAny>,
        method: &str,
        n_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<HashMap<String, f64>> {
        let result = self.run(value_fn, method, n_samples, seed)?;
        Ok(self.values_map(&result))
    }

    /// Compute ASV values with diagnostics.
    ///
    /// Returns a dict with keys:
    ///   - `"values"`: dict[str, float] — ASV per node
    ///   - `"ess"`: float | None — effective sample size (approx only)
    ///   - `"n_samples"`: int — orderings used
    ///   - `"is_exact"`: bool
    #[pyo3(signature = (value_fn, method = "auto", n_samples = 10_000, seed = None))]
    fn explain_with_diagnostics<'py>(
        &self,
        py: Python<'py>,
        value_fn: Py<PyAny>,
        method: &str,
        n_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self.run(value_fn, method, n_samples, seed)?;
        let ess_ratio = result
            .effective_sample_size
            .map(|e| e / result.n_samples as f64);
        let d = PyDict::new(py);
        d.set_item("values", self.values_map(&result))?;
        d.set_item("ess", result.effective_sample_size)?;
        d.set_item("ess_ratio", ess_ratio)?;
        d.set_item("n_samples", result.n_samples)?;
        d.set_item("seed", result.seed)?;
        d.set_item("is_exact", result.is_exact)?;
        d.set_item("method", method)?;
        Ok(d)
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
