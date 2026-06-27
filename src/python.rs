use std::collections::{HashMap, HashSet, VecDeque};

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

    /// Return a JSON string representing the DAG.
    ///
    /// Format: `{"nodes":["a","b"],"edges":[{"from":"a","to":"b"}]}`
    fn to_json(&self) -> String {
        let nodes: Vec<String> = self
            .inner
            .all_nodes()
            .map(|id| format!("\"{}\"", self.inner.node_name(id).unwrap()))
            .collect();
        let edges: Vec<String> = self
            .inner
            .all_nodes()
            .flat_map(|from_id| {
                let from = self.inner.node_name(from_id).unwrap().to_string();
                self.inner
                    .children_raw(from_id)
                    .iter()
                    .map(move |&to_id| {
                        format!(
                            "{{\"from\":\"{}\",\"to\":\"{}\"}}",
                            from,
                            self.inner.node_name(to_id).unwrap()
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        format!(
            "{{\"nodes\":[{}],\"edges\":[{}]}}",
            nodes.join(","),
            edges.join(",")
        )
    }

    /// Construct a DAG from the JSON format produced by `to_json()`.
    ///
    /// Accepts `{"nodes":[...],"edges":[{"from":"a","to":"b"},...]}`
    /// Unknown keys are ignored. Nodes listed only in `"nodes"` (no edges) are added as isolates.
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        // ponytail: hand-rolled parser — avoids serde dependency for simple known format
        let mut inner = RustDag::new();
        // Extract nodes array content
        if let Some(nodes_start) = s.find("\"nodes\":[") {
            let rest = &s[nodes_start + 9..];
            if let Some(end) = rest.find(']') {
                let names_raw = &rest[..end];
                for part in names_raw.split(',') {
                    let name = part.trim().trim_matches('"');
                    if !name.is_empty() {
                        inner.add_node(name);
                    }
                }
            }
        }
        // Extract edges
        let mut search = s;
        while let Some(from_pos) = search.find("\"from\":\"") {
            let after_from = &search[from_pos + 8..];
            let from_end = after_from.find('"').ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "malformed JSON: missing closing quote after from",
                )
            })?;
            let from_name = &after_from[..from_end];
            let to_start = after_from[from_end..].find("\"to\":\"").ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("malformed JSON: missing 'to' key")
            })?;
            let after_to = &after_from[from_end + to_start + 6..];
            let to_end = after_to.find('"').ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "malformed JSON: missing closing quote after to",
                )
            })?;
            let to_name = &after_to[..to_end];
            let from_id = inner.add_node(from_name);
            let to_id = inner.add_node(to_name);
            inner.add_edge(from_id, to_id).map_err(py_err)?;
            search = &after_to[to_end..];
        }
        Ok(Self { inner })
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

    /// Return sorted ancestor names — nodes from which `name` is reachable.
    fn ancestors(&self, name: &str) -> PyResult<Vec<String>> {
        let start = self
            .inner
            .node_id(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown node: {name}")))?;
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        for &p in self.inner.parents_raw(start) {
            if visited.insert(p) {
                queue.push_back(p);
            }
        }
        while let Some(node) = queue.pop_front() {
            for &p in self.inner.parents_raw(node) {
                if visited.insert(p) {
                    queue.push_back(p);
                }
            }
        }
        let mut result: Vec<String> = visited
            .iter()
            .map(|&id| self.inner.node_name(id).unwrap().to_string())
            .collect();
        result.sort();
        Ok(result)
    }

    /// Return sorted descendant names — nodes reachable from `name`.
    fn descendants(&self, name: &str) -> PyResult<Vec<String>> {
        let start = self
            .inner
            .node_id(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown node: {name}")))?;
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        for &c in self.inner.children_raw(start) {
            if visited.insert(c) {
                queue.push_back(c);
            }
        }
        while let Some(node) = queue.pop_front() {
            for &c in self.inner.children_raw(node) {
                if visited.insert(c) {
                    queue.push_back(c);
                }
            }
        }
        let mut result: Vec<String> = visited
            .iter()
            .map(|&id| self.inner.node_name(id).unwrap().to_string())
            .collect();
        result.sort();
        Ok(result)
    }

    /// Return nodes grouped by topological layer (roots in layer 0).
    ///
    /// All nodes in layer k have their parents in layers 0..k.
    fn topological_layers(&self) -> Vec<Vec<String>> {
        let n = self.inner.node_count();
        let mut remaining_in = self.inner.in_degrees();
        let mut layers = Vec::new();
        let mut remaining: Vec<bool> = vec![true; n];
        loop {
            let layer: Vec<usize> = (0..n)
                .filter(|&i| remaining[i] && remaining_in[i] == 0)
                .collect();
            if layer.is_empty() {
                break;
            }
            for &i in &layer {
                remaining[i] = false;
                for &child in self.inner.children_raw(crate::graph::NodeId(i as u32)) {
                    remaining_in[child.0 as usize] -= 1;
                }
            }
            let mut names: Vec<String> = layer
                .iter()
                .map(|&i| {
                    self.inner
                        .node_name(crate::graph::NodeId(i as u32))
                        .unwrap()
                        .to_string()
                })
                .collect();
            names.sort();
            layers.push(names);
        }
        layers
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

    /// Shared batched computation: calls value_fn_batch(list[list[str]]) -> list[float].
    fn run_batch(
        &self,
        value_fn_batch: Py<PyAny>,
        method: &str,
        n_samples: usize,
        seed: Option<u64>,
        batch_size: usize,
    ) -> PyResult<AsvResult> {
        if !matches!(method, "approx" | "auto") {
            return Err(PyValueError::new_err(
                "value_fn_batch is only supported for method='approx' or 'auto'",
            ));
        }
        let names = &self.names;
        let rust_batch_fn = move |coalitions: &[Vec<NodeId>]| -> Result<Vec<f64>, CausasvError> {
            Python::attach(|py| {
                let py_coalitions: Vec<Vec<&str>> = coalitions
                    .iter()
                    .map(|coal| {
                        coal.iter()
                            .map(|id| names[id.0 as usize].as_str())
                            .collect()
                    })
                    .collect();
                value_fn_batch
                    .call1(py, (py_coalitions,))
                    .and_then(|r| r.extract::<Vec<f64>>(py))
                    .map_err(|e| CausasvError::ValueFunctionError(e.to_string()))
            })
        };
        let mut cfg = SamplingConfig::new(n_samples).with_batch_size(batch_size);
        if let Some(s) = seed {
            cfg = cfg.with_seed(s);
        }
        self.inner
            .approximate_batched(rust_batch_fn, cfg)
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
    /// Pass `value_fn_batch` instead of (or alongside) `value_fn` to amortize Python
    /// GIL overhead over `batch_size` samples per call (useful for large models).
    /// `value_fn_batch` takes `list[list[str]]` and returns `list[float]`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (value_fn=None, method="auto", n_samples=10_000, seed=None,
                        value_fn_batch=None, batch_size=256))]
    fn explain(
        &self,
        _py: Python<'_>,
        value_fn: Option<Py<PyAny>>,
        method: &str,
        n_samples: usize,
        seed: Option<u64>,
        value_fn_batch: Option<Py<PyAny>>,
        batch_size: usize,
    ) -> PyResult<HashMap<String, f64>> {
        if let Some(batch_fn) = value_fn_batch {
            Ok(self.values_map(&self.run_batch(batch_fn, method, n_samples, seed, batch_size)?))
        } else if let Some(vfn) = value_fn {
            Ok(self.values_map(&self.run(vfn, method, n_samples, seed)?))
        } else {
            Err(PyValueError::new_err(
                "must provide either value_fn or value_fn_batch",
            ))
        }
    }

    /// Compute ASV values with diagnostics.
    ///
    /// Returns a dict with keys:
    ///   - `"values"`: dict[str, float] — ASV per node
    ///   - `"ess"`: float | None — effective sample size (approx only)
    ///   - `"n_samples"`: int — orderings used
    ///   - `"is_exact"`: bool
    ///
    /// Pass `value_fn_batch` to amortize Python GIL overhead over `batch_size` samples per call.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (value_fn=None, method="auto", n_samples=10_000, seed=None,
                        value_fn_batch=None, batch_size=256))]
    fn explain_with_diagnostics<'py>(
        &self,
        py: Python<'py>,
        value_fn: Option<Py<PyAny>>,
        method: &str,
        n_samples: usize,
        seed: Option<u64>,
        value_fn_batch: Option<Py<PyAny>>,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = if let Some(batch_fn) = value_fn_batch {
            self.run_batch(batch_fn, method, n_samples, seed, batch_size)?
        } else if let Some(vfn) = value_fn {
            self.run(vfn, method, n_samples, seed)?
        } else {
            return Err(PyValueError::new_err(
                "must provide either value_fn or value_fn_batch",
            ));
        };
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
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (value_fn, min_samples=1_000, max_samples=100_000,
                        batch_size=1_000, rel_tol=0.01, ess_ratio_min=0.10, seed=None))]
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
    ) -> PyResult<Bound<'py, PyDict>> {
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
        d.set_item("values", self.values_map(&result))?;
        d.set_item("ess", result.effective_sample_size)?;
        d.set_item("ess_ratio", ess_ratio)?;
        d.set_item("n_samples", result.n_samples)?;
        d.set_item("seed", result.seed)?;
        d.set_item("is_exact", result.is_exact)?;
        d.set_item("method", "approx_adaptive")?;
        d.set_item("converged", result.converged)?;
        d.set_item("stderr", stderr_map)?;
        Ok(d)
    }

    /// Adaptive batched approximate ASV: same convergence logic as `explain_adaptive`
    /// but calls `value_fn_batch(list[list[str]]) -> list[float]` once per sampling batch.
    ///
    /// Each sampling batch of `batch_size` samples becomes one Python function call,
    /// reducing GIL acquisition overhead for large models.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (value_fn_batch, min_samples=1_000, max_samples=100_000,
                        batch_size=1_000, rel_tol=0.01, ess_ratio_min=0.10, seed=None))]
    fn explain_adaptive_batch<'py>(
        &self,
        py: Python<'py>,
        value_fn_batch: Py<PyAny>,
        min_samples: usize,
        max_samples: usize,
        batch_size: usize,
        rel_tol: f64,
        ess_ratio_min: f64,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let names = &self.names;
        let rust_batch_fn = move |coalitions: &[Vec<NodeId>]| -> Result<Vec<f64>, CausasvError> {
            Python::attach(|py| {
                let py_coalitions: Vec<Vec<&str>> = coalitions
                    .iter()
                    .map(|coal| {
                        coal.iter()
                            .map(|id| names[id.0 as usize].as_str())
                            .collect()
                    })
                    .collect();
                value_fn_batch
                    .call1(py, (py_coalitions,))
                    .and_then(|r| r.extract::<Vec<f64>>(py))
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
            .approximate_adaptive_batched(rust_batch_fn, config)
            .map_err(py_err)?;
        let ess_ratio = result
            .effective_sample_size
            .map(|e| e / result.n_samples as f64);
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
        d.set_item("values", self.values_map(&result))?;
        d.set_item("ess", result.effective_sample_size)?;
        d.set_item("ess_ratio", ess_ratio)?;
        d.set_item("n_samples", result.n_samples)?;
        d.set_item("seed", result.seed)?;
        d.set_item("is_exact", result.is_exact)?;
        d.set_item("method", "approx_adaptive_batch")?;
        d.set_item("converged", result.converged)?;
        d.set_item("stderr", stderr_map)?;
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
