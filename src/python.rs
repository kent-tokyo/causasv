use std::collections::{HashMap, HashSet, VecDeque};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::asv::{AsvExplainer, AsvResult};
use crate::cpdag::Cpdag as RustCpdag;
use crate::error::CausasvError;
use crate::graph::{Dag as RustDag, NodeId};
use crate::numerics::normal_quantile;
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
        let nodes: Vec<serde_json::Value> = self
            .inner
            .all_nodes()
            .map(|id| serde_json::Value::from(self.inner.node_name(id).unwrap()))
            .collect();
        let edges: Vec<serde_json::Value> = self
            .inner
            .all_nodes()
            .flat_map(|from_id| {
                let from = self.inner.node_name(from_id).unwrap().to_string();
                self.inner
                    .children_raw(from_id)
                    .iter()
                    .map(move |&to_id| {
                        serde_json::json!({
                            "from": from,
                            "to": self.inner.node_name(to_id).unwrap(),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        serde_json::json!({"nodes": nodes, "edges": edges}).to_string()
    }

    /// Construct a DAG from the JSON format produced by `to_json()`.
    ///
    /// Accepts `{"nodes":[...],"edges":[{"from":"a","to":"b"},...]}`, parsed as
    /// structural JSON (whitespace/formatting-insensitive, matching e.g. Python's
    /// `json.dumps()` default output, not just `to_json()`'s own compact style).
    /// Unknown keys are ignored. Nodes listed only in `"nodes"` (no edges) are added
    /// as isolates. Malformed JSON, or `"nodes"`/`"edges"` present but not shaped as
    /// documented, raises `ValueError` rather than silently producing an empty or
    /// partial graph.
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        let value: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| PyValueError::new_err(format!("malformed JSON: {e}")))?;
        let mut inner = RustDag::new();

        if let Some(nodes) = value.get("nodes") {
            let nodes = nodes.as_array().ok_or_else(|| {
                PyValueError::new_err("malformed JSON: \"nodes\" must be an array")
            })?;
            for n in nodes {
                let name = n.as_str().ok_or_else(|| {
                    PyValueError::new_err("malformed JSON: \"nodes\" entries must be strings")
                })?;
                inner.add_node(name);
            }
        }

        if let Some(edges) = value.get("edges") {
            let edges = edges.as_array().ok_or_else(|| {
                PyValueError::new_err("malformed JSON: \"edges\" must be an array")
            })?;
            for e in edges {
                let from = e.get("from").and_then(|v| v.as_str()).ok_or_else(|| {
                    PyValueError::new_err("malformed JSON: edge missing string \"from\"")
                })?;
                let to = e.get("to").and_then(|v| v.as_str()).ok_or_else(|| {
                    PyValueError::new_err("malformed JSON: edge missing string \"to\"")
                })?;
                let from_id = inner.add_node(from);
                let to_id = inner.add_node(to);
                inner.add_edge(from_id, to_id).map_err(py_err)?;
            }
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

#[pyclass(name = "CausalCPDAG")]
pub struct PyCpdag {
    pub(crate) inner: RustCpdag,
}

#[pymethods]
impl PyCpdag {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustCpdag::new(),
        }
    }

    /// Construct a CPDAG from directed and undirected (from, to)/(a, b) edge
    /// tuples. Nodes are created automatically.
    #[staticmethod]
    #[pyo3(signature = (directed=Vec::new(), undirected=Vec::new()))]
    fn from_edges(
        directed: Vec<(String, String)>,
        undirected: Vec<(String, String)>,
    ) -> PyResult<Self> {
        let mut inner = RustCpdag::new();
        for (from, to) in &directed {
            let from_id = inner.add_node(from);
            let to_id = inner.add_node(to);
            inner.add_directed_edge(from_id, to_id).map_err(py_err)?;
        }
        for (a, b) in &undirected {
            let a_id = inner.add_node(a);
            let b_id = inner.add_node(b);
            inner.add_undirected_edge(a_id, b_id).map_err(py_err)?;
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

    /// Return all directed edges as (from, to) name pairs.
    fn directed_edges(&self) -> Vec<(String, String)> {
        self.inner
            .directed_edges()
            .map(|(from, to)| {
                (
                    self.inner.node_name(from).unwrap().to_string(),
                    self.inner.node_name(to).unwrap().to_string(),
                )
            })
            .collect()
    }

    /// Return all undirected edges as (a, b) name pairs, `a < b` by `NodeId`.
    fn undirected_edges(&self) -> Vec<(String, String)> {
        self.inner
            .undirected_edges()
            .map(|(a, b)| {
                (
                    self.inner.node_name(a).unwrap().to_string(),
                    self.inner.node_name(b).unwrap().to_string(),
                )
            })
            .collect()
    }

    /// Add a compelled (directed) edge. Nodes are created automatically.
    fn add_directed_edge(&mut self, from_name: &str, to_name: &str) -> PyResult<()> {
        let from = self.inner.add_node(from_name);
        let to = self.inner.add_node(to_name);
        self.inner.add_directed_edge(from, to).map_err(py_err)
    }

    /// Add an unresolved (undirected) edge. Nodes are created automatically.
    fn add_undirected_edge(&mut self, a_name: &str, b_name: &str) -> PyResult<()> {
        let a = self.inner.add_node(a_name);
        let b = self.inner.add_node(b_name);
        self.inner.add_undirected_edge(a, b).map_err(py_err)
    }

    /// Cheap structural validation (non-empty, no directed cycle). Does not
    /// verify full CPDAG validity -- see `validate_cpdag`.
    fn validate_pdag(&self) -> PyResult<()> {
        self.inner.validate_pdag().map_err(py_err)
    }

    /// `validate_pdag` plus: a consistent DAG extension exists. Does not
    /// verify the directed edges are exactly the compelled edges of a
    /// genuine Markov equivalence class -- see the Rust doc comment on
    /// `Cpdag::validate_cpdag` for the precise boundary.
    fn validate_cpdag(&self) -> PyResult<()> {
        self.inner.validate_cpdag().map_err(py_err)
    }

    /// Construct a consistent DAG extension (Dor-Tarsi / Chickering
    /// PDAG-to-DAG algorithm). Raises `ValueError` if no valid extension
    /// exists (e.g. a chordless undirected cycle).
    fn consistent_extension(&self) -> PyResult<PyCausalDAG> {
        let inner = self.inner.consistent_extension().map_err(py_err)?;
        Ok(PyCausalDAG { inner })
    }

    /// Return the sub-CPDAG induced by `names`: nodes not listed are
    /// dropped, along with every edge touching a dropped node. Edge kind
    /// (directed/undirected) is preserved for retained pairs.
    fn induced_subgraph(&self, names: Vec<String>) -> PyResult<PyCpdag> {
        let ids: Vec<NodeId> = names
            .iter()
            .map(|name| {
                self.inner
                    .node_id(name)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown node: {name}")))
            })
            .collect::<PyResult<_>>()?;
        let inner = self.inner.induced_subgraph(&ids).map_err(py_err)?;
        Ok(PyCpdag { inner })
    }

    /// Return a JSON string representing the CPDAG.
    ///
    /// Format: `{"nodes":["a","b"],"directed_edges":[{"from":"a","to":"b"}],"undirected_edges":[{"a":"b","b":"c"}]}`
    fn to_json(&self) -> String {
        let nodes: Vec<serde_json::Value> = self
            .inner
            .all_nodes()
            .map(|id| serde_json::Value::from(self.inner.node_name(id).unwrap()))
            .collect();
        let directed: Vec<serde_json::Value> = self
            .inner
            .directed_edges()
            .map(|(from, to)| {
                serde_json::json!({
                    "from": self.inner.node_name(from).unwrap(),
                    "to": self.inner.node_name(to).unwrap(),
                })
            })
            .collect();
        let undirected: Vec<serde_json::Value> = self
            .inner
            .undirected_edges()
            .map(|(a, b)| {
                serde_json::json!({
                    "a": self.inner.node_name(a).unwrap(),
                    "b": self.inner.node_name(b).unwrap(),
                })
            })
            .collect();
        serde_json::json!({
            "nodes": nodes,
            "directed_edges": directed,
            "undirected_edges": undirected,
        })
        .to_string()
    }

    /// Construct a CPDAG from the JSON format produced by `to_json()`.
    ///
    /// Accepts `{"nodes":[...],"directed_edges":[{"from":"a","to":"b"},...],"undirected_edges":[{"a":"b","b":"c"},...]}`,
    /// parsed as structural JSON (whitespace-insensitive). Unknown keys are
    /// ignored. Malformed JSON, or shapes not matching the above, raises
    /// `ValueError`.
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        let value: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| PyValueError::new_err(format!("malformed JSON: {e}")))?;
        let mut inner = RustCpdag::new();

        if let Some(nodes) = value.get("nodes") {
            let nodes = nodes.as_array().ok_or_else(|| {
                PyValueError::new_err("malformed JSON: \"nodes\" must be an array")
            })?;
            for n in nodes {
                let name = n.as_str().ok_or_else(|| {
                    PyValueError::new_err("malformed JSON: \"nodes\" entries must be strings")
                })?;
                inner.add_node(name);
            }
        }

        if let Some(edges) = value.get("directed_edges") {
            let edges = edges.as_array().ok_or_else(|| {
                PyValueError::new_err("malformed JSON: \"directed_edges\" must be an array")
            })?;
            for e in edges {
                let from = e.get("from").and_then(|v| v.as_str()).ok_or_else(|| {
                    PyValueError::new_err("malformed JSON: directed edge missing string \"from\"")
                })?;
                let to = e.get("to").and_then(|v| v.as_str()).ok_or_else(|| {
                    PyValueError::new_err("malformed JSON: directed edge missing string \"to\"")
                })?;
                let from_id = inner.add_node(from);
                let to_id = inner.add_node(to);
                inner.add_directed_edge(from_id, to_id).map_err(py_err)?;
            }
        }

        if let Some(edges) = value.get("undirected_edges") {
            let edges = edges.as_array().ok_or_else(|| {
                PyValueError::new_err("malformed JSON: \"undirected_edges\" must be an array")
            })?;
            for e in edges {
                let a = e.get("a").and_then(|v| v.as_str()).ok_or_else(|| {
                    PyValueError::new_err("malformed JSON: undirected edge missing string \"a\"")
                })?;
                let b = e.get("b").and_then(|v| v.as_str()).ok_or_else(|| {
                    PyValueError::new_err("malformed JSON: undirected edge missing string \"b\"")
                })?;
                let a_id = inner.add_node(a);
                let b_id = inner.add_node(b);
                inner.add_undirected_edge(a_id, b_id).map_err(py_err)?;
            }
        }

        Ok(Self { inner })
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
        parallel: bool,
        num_threads: Option<usize>,
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
            let mut cfg = SamplingConfig::new(n_samples).with_parallel(parallel);
            if let Some(s) = seed {
                cfg = cfg.with_seed(s);
            }
            if let Some(t) = num_threads {
                cfg = cfg.with_num_threads(t);
            }
            cfg
        };
        match method {
            "auto" => self.inner.auto(rust_fn, make_cfg()),
            "approx" => self.inner.approximate(rust_fn, make_cfg()),
            "exact" => self.inner.exact(rust_fn),
            "exact_tree" => self.inner.exact_tree(rust_fn),
            "exact_dag" => self.inner.exact_dag(rust_fn),
            "exact_dag_sparse" => self.inner.exact_dag_sparse(rust_fn),
            "uniform_sparse" => self.inner.approximate_uniform_sparse(rust_fn, make_cfg()),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown method '{method}': use 'auto', 'approx', 'exact', 'exact_tree', \
                     'exact_dag', 'exact_dag_sparse', or 'uniform_sparse'"
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
    /// Pass `value_fn_batch` to amortize Python GIL overhead over `batch_size` samples per call.
    /// Set `parallel=True` with a `seed` for deterministic parallel sampling.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (value_fn=None, method="auto", n_samples=10_000, seed=None,
                        value_fn_batch=None, batch_size=256, parallel=false, num_threads=None))]
    fn explain(
        &self,
        py: Python<'_>,
        value_fn: Option<Py<PyAny>>,
        method: &str,
        n_samples: usize,
        seed: Option<u64>,
        value_fn_batch: Option<Py<PyAny>>,
        batch_size: usize,
        parallel: bool,
        num_threads: Option<usize>,
    ) -> PyResult<HashMap<String, f64>> {
        if let Some(batch_fn) = value_fn_batch {
            Ok(self.values_map(&self.run_batch(batch_fn, method, n_samples, seed, batch_size)?))
        } else if let Some(vfn) = value_fn {
            let result =
                py.detach(|| self.run(vfn, method, n_samples, seed, parallel, num_threads))?;
            Ok(self.values_map(&result))
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
    ///   - `"parallel"`: bool — whether parallel sampling was used
    ///   - `"num_threads"`: int | None — worker count for seeded parallel
    ///   - `"deterministic"`: bool — True when seed + parallel → per-worker seeds
    ///
    /// Pass `value_fn_batch` to amortize Python GIL overhead over `batch_size` samples per call.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (value_fn=None, method="auto", n_samples=10_000, seed=None,
                        value_fn_batch=None, batch_size=256, parallel=false, num_threads=None))]
    fn explain_with_diagnostics<'py>(
        &self,
        py: Python<'py>,
        value_fn: Option<Py<PyAny>>,
        method: &str,
        n_samples: usize,
        seed: Option<u64>,
        value_fn_batch: Option<Py<PyAny>>,
        batch_size: usize,
        parallel: bool,
        num_threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = if let Some(batch_fn) = value_fn_batch {
            self.run_batch(batch_fn, method, n_samples, seed, batch_size)?
        } else if let Some(vfn) = value_fn {
            py.detach(|| self.run(vfn, method, n_samples, seed, parallel, num_threads))?
        } else {
            return Err(PyValueError::new_err(
                "must provide either value_fn or value_fn_batch",
            ));
        };
        let ess_ratio = result
            .effective_sample_size
            .map(|e| e / result.n_samples as f64);
        let deterministic = seed.is_some() && parallel;
        let d = PyDict::new(py);
        d.set_item("values", self.values_map(&result))?;
        d.set_item("ess", result.effective_sample_size)?;
        d.set_item("ess_ratio", ess_ratio)?;
        d.set_item("n_samples", result.n_samples)?;
        d.set_item("seed", result.seed)?;
        d.set_item("is_exact", result.is_exact)?;
        d.set_item("method", method)?;
        d.set_item("parallel", parallel)?;
        d.set_item("num_threads", num_threads)?;
        d.set_item("deterministic", deterministic)?;
        d.set_item("n_order_ideals", result.n_order_ideals)?;
        d.set_item("state_ratio", result.state_ratio)?;
        d.set_item("memory_mb", result.memory_mb)?;
        d.set_item("fallback_from", result.fallback_from.as_deref())?;
        d.set_item("fallback_reason", result.fallback_reason.as_deref())?;
        d.set_item("selected_method", result.method_used.unwrap_or(method))?;
        Ok(d)
    }

    /// Quality-first one-stop entry point: exact when feasible, uniform sparse adaptive
    /// otherwise. Always returns stderr and ci_low/ci_high when ci is set.
    ///
    /// Uses `auto_quality` dispatch under the hood — the same as `explain_adaptive` but
    /// exact methods are tried first and the approximate fallback has ESS = n_samples.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (value_fn, min_samples=1_000, max_samples=100_000,
                        batch_size=1_000, rel_tol=0.01, seed=None, ci=None))]
    fn explain_quality<'py>(
        &self,
        py: Python<'py>,
        value_fn: Py<PyAny>,
        min_samples: usize,
        max_samples: usize,
        batch_size: usize,
        rel_tol: f64,
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
            ess_ratio_min: 0.0, // uniform sparse: ESS always = n_samples, gate is irrelevant
            seed,
        };
        let result = self.inner.auto_quality(rust_fn, config).map_err(py_err)?;
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
        d.set_item(
            "selected_method",
            result.method_used.unwrap_or("auto_quality"),
        )?;
        d.set_item("converged", result.converged)?;
        d.set_item("stderr", &stderr_map)?;
        d.set_item("fallback_from", result.fallback_from.as_deref())?;
        d.set_item("fallback_reason", result.fallback_reason.as_deref())?;
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

    /// Adaptive approximate ASV: runs sampling in batches until convergence or max_samples.
    ///
    /// Returns a dict with keys: values, ess, ess_ratio, n_samples, seed, is_exact,
    /// method, converged, stderr.
    ///
    /// When `ci` is set (e.g. `ci=0.95`), also returns `ci_low`, `ci_high` using a
    /// normal approximation: value ± Φ⁻¹((1+ci)/2) × stderr.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (value_fn, min_samples=1_000, max_samples=100_000,
                        batch_size=1_000, rel_tol=0.01, ess_ratio_min=0.10, seed=None, ci=None,
                        method="approx"))]
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
        method: &str,
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
        let result = match method {
            "approx" => self.inner.approximate_adaptive(rust_fn, config),
            "uniform_sparse" => self.inner.approximate_uniform_sparse_adaptive(rust_fn, config),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown method '{method}' for explain_adaptive: use 'approx' or 'uniform_sparse'"
                )));
            }
        }
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
        let method_label = if method == "uniform_sparse" {
            "uniform_sparse_adaptive"
        } else {
            "approx_adaptive"
        };
        let d = PyDict::new(py);
        d.set_item("values", &values_map)?;
        d.set_item("ess", result.effective_sample_size)?;
        d.set_item("ess_ratio", ess_ratio)?;
        d.set_item("n_samples", result.n_samples)?;
        d.set_item("seed", result.seed)?;
        d.set_item("is_exact", result.is_exact)?;
        d.set_item("method", method_label)?;
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

    /// Adaptive batched approximate ASV: same convergence logic as `explain_adaptive`
    /// but calls `value_fn_batch(list[list[str]]) -> list[float]` once per sampling batch.
    ///
    /// Each sampling batch of `batch_size` samples becomes one Python function call,
    /// reducing GIL acquisition overhead for large models.
    /// Quality-first batched ASV: like `explain_quality` but accepts a batch value function.
    ///
    /// Routes `n ≤ 63` to uniform sparse adaptive batch sampling (ESS = n_samples, no IS
    /// variance). Falls back to IS-adaptive batch for `n > 63`. Returns the same dict keys
    /// as `explain_quality`: values, stderr, ess, ess_ratio, n_samples, seed, is_exact,
    /// selected_method, converged, fallback_from, fallback_reason, and optionally ci/ci_low/ci_high.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (value_fn_batch, min_samples=1_000, max_samples=100_000,
                        batch_size=1_000, rel_tol=0.01, seed=None, ci=None))]
    fn explain_quality_batch<'py>(
        &self,
        py: Python<'py>,
        value_fn_batch: Py<PyAny>,
        min_samples: usize,
        max_samples: usize,
        batch_size: usize,
        rel_tol: f64,
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
            ess_ratio_min: 0.0, // uniform: ESS = n_samples always
            seed,
        };
        let n = self.names.len();
        let (result, selected_method) = if n <= 63 {
            let r = self
                .inner
                .approximate_uniform_sparse_adaptive_batched(rust_batch_fn, config)
                .map_err(py_err)?;
            (r, "uniform_sparse_adaptive_batch")
        } else {
            let r = self
                .inner
                .approximate_adaptive_batched(rust_batch_fn, config)
                .map_err(py_err)?;
            (r, "approx_adaptive_batch")
        };
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
        d.set_item("selected_method", selected_method)?;
        d.set_item("converged", result.converged)?;
        d.set_item("stderr", &stderr_map)?;
        d.set_item("fallback_from", result.fallback_from.as_deref())?;
        d.set_item("fallback_reason", result.fallback_reason.as_deref())?;
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

/// Normal quantile function Φ⁻¹(p) via rational approximation.
///
/// Abramowitz & Stegun 26.2.17 — |error| < 4.5 × 10⁻⁴.
/// Sufficient precision for CI display (ci=0.95 → z=1.9600, exact=1.95996).
#[pymodule]
fn causasv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCausalDAG>()?;
    m.add_class::<PyCpdag>()?;
    m.add_class::<PyASVExplainer>()?;
    Ok(())
}

fn py_err(e: CausasvError) -> PyErr {
    PyValueError::new_err(e.to_string())
}
