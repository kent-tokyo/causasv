from typing import Any, Callable

ValueFn = Callable[[list[str]], float]
ValueFnBatch = Callable[[list[list[str]]], list[float]]

class CausalDAG:
    def __init__(self) -> None: ...
    @staticmethod
    def from_networkx(g: Any) -> CausalDAG: ...
    @staticmethod
    def from_edges(edges: list[tuple[str, str]]) -> CausalDAG: ...
    def nodes(self) -> list[str]: ...
    def edges(self) -> list[tuple[str, str]]: ...
    def add_edge(self, from_name: str, to_name: str) -> None: ...
    def validate(self) -> None: ...
    def ancestors(self, name: str) -> list[str]: ...
    def descendants(self, name: str) -> list[str]: ...
    def topological_layers(self) -> list[list[str]]: ...
    def to_json(self) -> str: ...
    def to_dot(self) -> str: ...

class CausalCPDAG:
    def __init__(self) -> None: ...
    @staticmethod
    def from_edges(
        directed: list[tuple[str, str]] = ...,
        undirected: list[tuple[str, str]] = ...,
    ) -> CausalCPDAG: ...
    def nodes(self) -> list[str]: ...
    def directed_edges(self) -> list[tuple[str, str]]: ...
    def undirected_edges(self) -> list[tuple[str, str]]: ...
    def add_directed_edge(self, from_name: str, to_name: str) -> None: ...
    def add_undirected_edge(self, a_name: str, b_name: str) -> None: ...
    def validate_pdag(self) -> None: ...
    def validate_cpdag(self) -> None: ...
    def consistent_extension(self) -> CausalDAG: ...
    def induced_subgraph(self, names: list[str]) -> CausalCPDAG: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(s: str) -> CausalCPDAG: ...

class ASVExplainer:
    def __init__(self, dag: CausalDAG) -> None: ...
    def explain(
        self,
        value_fn: ValueFn | None = ...,
        method: str = "auto",
        n_samples: int = 10_000,
        seed: int | None = ...,
        value_fn_batch: ValueFnBatch | None = ...,
        batch_size: int = 256,
        parallel: bool = False,
        num_threads: int | None = ...,
    ) -> dict[str, float]: ...
    def explain_with_diagnostics(
        self,
        value_fn: ValueFn | None = ...,
        method: str = "auto",
        n_samples: int = 10_000,
        seed: int | None = ...,
        value_fn_batch: ValueFnBatch | None = ...,
        batch_size: int = 256,
        parallel: bool = False,
        num_threads: int | None = ...,
    ) -> dict[str, Any]: ...
    def explain_quality(
        self,
        value_fn: ValueFn,
        min_samples: int = 1_000,
        max_samples: int = 100_000,
        batch_size: int = 1_000,
        rel_tol: float = 0.01,
        seed: int | None = ...,
        ci: float | None = ...,
    ) -> dict[str, Any]: ...
    def explain_adaptive(
        self,
        value_fn: ValueFn,
        min_samples: int = 1_000,
        max_samples: int = 100_000,
        batch_size: int = 1_000,
        rel_tol: float = 0.01,
        ess_ratio_min: float = 0.10,
        seed: int | None = ...,
        ci: float | None = ...,
        method: str = "approx",
    ) -> dict[str, Any]: ...
    def explain_quality_batch(
        self,
        value_fn_batch: ValueFnBatch,
        min_samples: int = 1_000,
        max_samples: int = 100_000,
        batch_size: int = 1_000,
        rel_tol: float = 0.01,
        seed: int | None = ...,
        ci: float | None = ...,
    ) -> dict[str, Any]: ...
    def explain_adaptive_batch(
        self,
        value_fn_batch: ValueFnBatch,
        min_samples: int = 1_000,
        max_samples: int = 100_000,
        batch_size: int = 1_000,
        rel_tol: float = 0.01,
        ess_ratio_min: float = 0.10,
        seed: int | None = ...,
    ) -> dict[str, Any]: ...
