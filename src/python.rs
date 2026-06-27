use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::asv::{AsvExplainer, AsvResult};
use crate::error::CausasvError;
use crate::graph::{Dag as RustDag, NodeId};
use crate::sampler::{AdaptiveSamplingConfig, SamplingConfig};

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

    /// Return a structural summary of the DAG.
    ///
    /// Keys:
    ///   - `"n_nodes"`: int
    ///   - `"n_edges"`: int
    ///   - `"is_dag"`: bool — always True for a validated graph
    ///   - `"is_rooted_tree"`: bool — single root, all others have in-degree 1
    ///   - `"n_roots"`: int — nodes with in-degree 0
    ///   - `"n_leaves"`: int — nodes with out-degree 0
    ///   - `"max_depth"`: int — length of longest root-to-leaf path
    ///   - `"recommended_method"`: str — which AsvExplainer method auto() would pick
    ///   - `"estimated_dense_states"`: int | None — 2^n_nodes; None if n > 63
    fn inspect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let n = self.inner.node_count();
        let in_deg = self.inner.in_degrees();
        let n_roots = in_deg.iter().filter(|&&d| d == 0).count();
        let n_edges: usize = self
            .inner
            .all_nodes()
            .map(|id| self.inner.children_raw(id).len())
            .sum();
        let n_leaves = self
            .inner
            .all_nodes()
            .filter(|&id| self.inner.children_raw(id).is_empty())
            .count();
        // max_depth = number of topological layers - 1
        let max_depth = {
            let mut remaining_in = in_deg.clone();
            let mut depth = 0usize;
            let mut remaining: Vec<bool> = vec![true; n];
            loop {
                let layer: Vec<usize> = (0..n)
                    .filter(|&i| remaining[i] && remaining_in[i] == 0)
                    .collect();
                if layer.is_empty() {
                    break;
                }
                depth += 1;
                for &i in &layer {
                    remaining[i] = false;
                    for &child in self.inner.children_raw(crate::graph::NodeId(i as u32)) {
                        remaining_in[child.0 as usize] -= 1;
                    }
                }
            }
            depth.saturating_sub(1)
        };
        let is_rooted_tree = crate::tree::find_rooted_tree_root(&self.inner).is_ok();
        let recommended = if n <= 8 {
            "exact"
        } else if is_rooted_tree {
            "exact_tree"
        } else if n <= 20 {
            "exact_dag"
        } else {
            "approx"
        };
        let dense_states: Option<u64> = if n <= 63 { Some(1u64 << n) } else { None };
        let d = PyDict::new(py);
        d.set_item("n_nodes", n)?;
        d.set_item("n_edges", n_edges)?;
        d.set_item("is_dag", true)?;
        d.set_item("is_rooted_tree", is_rooted_tree)?;
        d.set_item("n_roots", n_roots)?;
        d.set_item("n_leaves", n_leaves)?;
        d.set_item("max_depth", max_depth)?;
        d.set_item("recommended_method", recommended)?;
        d.set_item("estimated_dense_states", dense_states)?;
        Ok(d)
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

    /// Adaptive approximate ASV: runs sampling in batches until convergence or max_samples.
    ///
    /// Returns a dict with keys: values, ess, ess_ratio, n_samples, seed, is_exact,
    /// method, converged, stderr.
    ///
    /// When `ci` is set (e.g. `ci=0.95`), also returns `ci_low`, `ci_high` using a
    /// normal approximation: value ± Φ⁻¹((1+ci)/2) × stderr.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (value_fn, min_samples=1_000, max_samples=100_000,
                        batch_size=1_000, rel_tol=0.01, ess_ratio_min=0.10, seed=None, ci=None))]
    fn explain_adaptive<'py>(
        &self,
        py: Python<'py>,
        value_fn: Py<PyAny>,
        min_samples: usize,
        max_samples: usize,
        batch_size: usize,
        rel_tol: f64,
        ess_ratio_min: f64,
        seed: Option<u64>,
        ci: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if let Some(ci_level) = ci
            && !(0.0 < ci_level && ci_level < 1.0)
        {
            return Err(PyValueError::new_err(
                "ci must be in (0, 1), e.g. ci=0.95 for a 95% interval",
            ));
        }
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
        let config = AdaptiveSamplingConfig {
            min_samples,
            max_samples,
            batch_size,
            rel_tol,
            ess_ratio_min,
            seed,
        };
        let result = self
            .inner
            .approximate_adaptive(rust_fn, config)
            .map_err(py_err)?;
        let ess_ratio = result
            .effective_sample_size
            .map(|e| e / result.n_samples as f64);
        let values_map = self.values_map(&result);
        let stderr_map: HashMap<String, f64> = result
            .stderr
            .as_ref()
            .map(|m| {
                m.iter()
                    .map(|(id, &v)| (self.names[id.0 as usize].clone(), v))
                    .collect()
            })
            .unwrap_or_default();
        let d = PyDict::new(py);
        d.set_item("values", &values_map)?;
        d.set_item("ess", result.effective_sample_size)?;
        d.set_item("ess_ratio", ess_ratio)?;
        d.set_item("n_samples", result.n_samples)?;
        d.set_item("seed", result.seed)?;
        d.set_item("is_exact", result.is_exact)?;
        d.set_item("method", "approx_adaptive")?;
        d.set_item("converged", result.converged)?;
        d.set_item("stderr", &stderr_map)?;
        if let Some(ci_level) = ci {
            let z = normal_quantile((1.0 + ci_level) / 2.0);
            let ci_low: HashMap<String, f64> = values_map
                .iter()
                .map(|(k, &v)| (k.clone(), v - z * stderr_map.get(k).copied().unwrap_or(0.0)))
                .collect();
            let ci_high: HashMap<String, f64> = values_map
                .iter()
                .map(|(k, &v)| (k.clone(), v + z * stderr_map.get(k).copied().unwrap_or(0.0)))
                .collect();
            d.set_item("ci", ci_level)?;
            d.set_item("ci_low", ci_low)?;
            d.set_item("ci_high", ci_high)?;
        }
        Ok(d)
    }
}

/// Normal quantile function Φ⁻¹(p) via rational approximation.
///
/// Abramowitz & Stegun 26.2.17 — |error| < 4.5 × 10⁻⁴.
/// Sufficient precision for CI display (ci=0.95 → z=1.9600, exact=1.95996).
fn normal_quantile(p: f64) -> f64 {
    let t = (-2.0 * (1.0 - p).ln()).sqrt();
    let c = [2.515_517, 0.802_853, 0.010_328];
    let d = [1.432_788, 0.189_269, 0.001_308];
    t - (c[0] + c[1] * t + c[2] * t * t) / (1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t)
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
