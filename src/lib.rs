//! Fast Causal Asymmetric Shapley Values for DAGs.
//!
//! # Quick Start
//!
//! ```rust
//! use causasv::{AsvExplainer, CausasvError, Dag};
//!
//! fn main() -> Result<(), CausasvError> {
//!     let mut dag = Dag::new();
//!     let x = dag.add_node("X");
//!     let y = dag.add_node("Y");
//!     dag.add_edge(x, y)?;
//!
//!     let explainer = AsvExplainer::new(dag);
//!     let result = explainer.exact(|coalition| Ok(coalition.len() as f64))?;
//!
//!     for (id, val) in &result.values {
//!         println!("{id:?} = {val:.4}");
//!     }
//!     Ok(())
//! }
//! ```
//!
//! For large graphs use [`AsvExplainer::approximate`] or [`AsvExplainer::auto`].
//! See the [README](https://github.com/kent-tokyo/causasv) for Python usage and benchmarks.

mod approx;
mod cache;
mod dag_dp;
mod error;
mod graph;
mod sampler;
mod topo;
mod tree;
mod value_function;

pub mod asv;

#[cfg(feature = "python")]
pub mod python;

pub use asv::{AsvExplainer, AsvResult};
pub use error::CausasvError;
pub use graph::{Dag, NodeId};
pub use sampler::SamplingConfig;
pub use topo::{enumerate_topos, topo_sort};
pub use value_function::ValueFn;
