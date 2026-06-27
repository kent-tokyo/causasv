# causasv — 基于非对称 Shapley 值的因果特征归因

[![CI](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml/badge.svg)](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/causasv.svg)](https://crates.io/crates/causasv)
[![Docs.rs](https://docs.rs/causasv/badge.svg)](https://docs.rs/causasv)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
<br>
[![MSRV](https://img.shields.io/badge/MSRV-1.85%2B-orange.svg)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://doc.rust-lang.org/nomicon/meet-safe-and-unsafe.html)

[English](README.md) | [日本語](README_ja.md) | **中文**

`causasv` 在用户提供的有向无环图（DAG）上计算**非对称 Shapley 值（ASV）**，用于因果特征归因。这是一个以 Rust 为核心、附带 Python 绑定的引擎，专为需要特征重要性遵循已知因果结构的 XAI 工作流而设计。

## 什么是 ASV？

非对称 Shapley 值（ASV）通过仅对**拓扑有效的特征排列**取平均来推广 Shapley 值，而非对所有排列取平均。给定因果 DAG G 和价值函数 v：

```
φ_i = (1 / |Π(G)|) Σ_{π ∈ Π(G)} [v(pre(i,π) ∪ {i}) − v(pre(i,π))]
```

其中 Π(G) 是 G 的所有线性扩展（拓扑排序）的集合，pre(i,π) 是在排列 π 中出现在特征 i 之前的特征集合。

## ASV 与 SHAP 的区别

标准 SHAP 忽略因果结构，对所有 n! 个特征排列取平均。ASV 将平均限制在与因果 DAG 一致的排列上——原因始终出现在其效果之前。这产生了遵循因果方向的归因结果。

## 为什么因果 DAG 很重要？

当特征之间存在因果关系时，SHAP 可能会将实际上由其后代节点中介的效果归因于某个变量。ASV 通过限制哪些排列被视为有效来防止这种情况。

## Rust 示例

```rust
use causasv::{AsvExplainer, Dag, SamplingConfig};

fn main() -> Result<(), causasv::CausasvError> {
    let mut dag = Dag::new();
    let education = dag.add_node("education");
    let income = dag.add_node("income");
    let risk = dag.add_node("risk_score");
    dag.add_edge(education, income)?;
    dag.add_edge(income, risk)?;
    dag.validate()?;

    let explainer = AsvExplainer::new(dag);

    // 通过重要性加权拓扑排序采样计算近似 ASV
    let values = explainer.approximate(
        |coalition| {
            Ok(coalition.len() as f64)
        },
        SamplingConfig::new(10_000).with_seed(42),
    )?;

    for (node, value) in &values.values {
        println!("节点 {:?}: ASV = {:.4}", node, value);
    }
    Ok(())
}
```

## Python 示例

```python
from causasv import CausalDAG, ASVExplainer

# 从边列表构建 DAG
dag = CausalDAG.from_edges([("education", "income"), ("income", "risk_score")])

# 或从 networkx DiGraph 构建
# import networkx as nx; G = nx.DiGraph(); G.add_edge(...)
# dag = CausalDAG.from_networkx(G)

explainer = ASVExplainer(dag)

values = explainer.explain(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="auto",   # n≤8 用 exact，有根树用 exact_tree，否则用 approx
    n_samples=10_000,
    seed=42,
)
# values: dict[str, float]，特征名 → ASV 值
```

Python 的 `value_fn` 接收联合中存在的特征名排序列表，必须返回一个浮点数。

使用 `explain_with_diagnostics()` 获取 ESS、种子和方法名：

```python
info = explainer.explain_with_diagnostics(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="approx",
    n_samples=10_000,
    seed=42,
)
print(info["values"])          # dict[str, float]
print(info["ess"])             # float — ESS ≈ n_samples 表示可靠
print(info["ess_ratio"])       # float — ESS / n_samples，接近 1 为好
print(info["method"])          # str — 输入的方法名
print(info["selected_method"]) # str — auto() 实际选择的方法
print(info["fallback_from"])   # str | None — 回退来源
```

使用 `explain_adaptive()` 进行自动收敛检测和逐特征置信区间：

```python
info = explainer.explain_adaptive(
    value_fn=lambda feature_names: my_model_score(feature_names),
    min_samples=1_000,
    max_samples=100_000,
    batch_size=1_000,
    seed=42,
    ci=0.95,          # 可选：添加 ci_low / ci_high
)
print(info["values"])     # dict[str, float]
print(info["stderr"])     # dict[str, float] — 每个特征的 IS 标准误差
print(info["ci_low"])     # dict[str, float] — 95% 置信区间下界
print(info["ci_high"])    # dict[str, float] — 95% 置信区间上界
print(info["converged"])  # bool — 是否在 max_samples 前达到 rel_tol
print(info["ess_ratio"])  # float — ESS / n_samples
```

批量联合评估（减少大型模型的 Python GIL 开销），使用 `value_fn_batch`：

```python
# value_fn_batch 接收 list[list[str]]，返回 list[float]
info = explainer.explain_with_diagnostics(
    value_fn_batch=lambda coalitions: [my_model_score(c) for c in coalitions],
    method="approx",
    n_samples=50_000,
    batch_size=512,
    seed=42,
)
```

确定性并行近似，同时传入 `seed` 和 `parallel=True`：

```python
info = explainer.explain_with_diagnostics(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="approx",
    n_samples=100_000,
    seed=42,
    parallel=True,
    num_threads=4,
)
print(info["deterministic"])  # True when seed + parallel
```

使用 `explain_stability()` 验证近似排名在不同种子下的一致性：

```python
from causasv import explain_stability

result = explain_stability(
    explainer,
    value_fn=lambda feature_names: my_model_score(feature_names),
    seeds=[1, 2, 3, 4, 5],
    method="approx",
    n_samples=10_000,
)
print(result["rank_stability"])  # 平均 Kendall tau；1.0 = 完全稳定
print(result["std_values"])      # dict[str, float] — 越小越稳定
print(result["mean_values"])     # dict[str, float] — 种子间的平均 ASV
```

使用 `ASVEnsembleExplainer` 测量多个候选 DAG 间的敏感性：

```python
from causasv import CausalDAG, ASVEnsembleExplainer

dag1 = CausalDAG.from_edges([("A", "B"), ("B", "C")])
dag2 = CausalDAG.from_edges([("A", "B"), ("A", "C")])
ensemble = ASVEnsembleExplainer([dag1, dag2])
result = ensemble.explain_with_sensitivity(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="auto",
)
print(result["mean_values"])     # dict[str, float] — DAG 间的平均 ASV
print(result["std_values"])      # dict[str, float] — DAG 间的标准差
print(result["rank_stability"])  # float — 平均 Kendall tau
print(result["per_dag_values"])  # list[dict[str, float]]
```

检查和导出 DAG：

```python
dag.nodes()                     # ["education", "income", "risk_score"]
dag.edges()                     # [("education", "income"), ("income", "risk_score")]
dag.to_dot()                    # 'digraph {\n  education -> income;\n  ...\n}'
dag.to_json()                   # '{"nodes":[...],"edges":[...]}'
dag.ancestors("risk_score")     # ["education", "income"]
dag.descendants("education")    # ["income", "risk_score"]
dag.topological_layers()        # [["education"], ["income"], ["risk_score"]]

# 从 JSON 恢复 DAG
dag2 = CausalDAG.from_json(dag.to_json())

# 转换为 networkx（需单独安装 networkx）
import networkx as nx
G = nx.DiGraph(dag.edges())
```

sklearn 兼容模型的高级 API `TabularExplainer`（需要 numpy）：

```python
from causasv import CausalDAG, TabularExplainer

dag = CausalDAG.from_edges([("education", "income"), ("income", "risk_score")])

explainer = TabularExplainer.from_model(
    model=my_classifier,
    dag=dag,
    background=X_train,
    feature_names=["education", "income", "risk_score"],
)
values = explainer.explain_instance(X_test[0], method="auto")
```

## 精确计算 vs 近似计算

| 方法 | 适用场景 | API |
|------|---------|-----|
| `exact` | 小型 DAG（n ≤ ~8）；枚举所有线性扩展 | `explainer.exact(value_fn)` |
| `exact_tree` | 有根有向树；顺序理想 DP | `explainer.exact_tree(value_fn)` |
| `exact_dag` | 一般 DAG，n ≤ 20；密集顺序理想 DP | `explainer.exact_dag(value_fn)` |
| `exact_dag_sparse` | 稀疏 DAG，n ≤ 28；仅对有效顺序理想 BFS | `explainer.exact_dag_sparse(value_fn)` |
| `approx` | 任意 DAG（n > 28 或超出内存限制）；IS 采样 | `explainer.approximate(value_fn, SamplingConfig::new(n))` |

`auto` 调度：n ≤ 8 → `exact`；有根树 → `exact_tree`；n ≤ 20 → `exact_dag`；20 < n ≤ 28 → `exact_dag_sparse`；否则 → `approx`。

`exact_dag_sparse` 只访问有效顺序理想（所有节点的父节点也存在的集合）。对于稀疏 DAG，这可能比 2^n 少几个数量级，返回 `n_order_ideals`、`state_ratio` 和 `memory_mb` 诊断信息。

近似估计器使用自归一化重要性采样来校正前沿采样器引入的偏差，因此即使对于近似结果，效率公理（Σφ_i = v(V) − v(∅)）也精确成立。

## 状态

实验性 — v0.8.0。在 v1.0 之前公共 API 可能会发生变化。

## 算法状态

| 方法 | 实现 | 备注 |
|------|------|------|
| `exact` | 枚举所有线性扩展 | 参考 oracle；实用范围 n ≤ ~8 |
| `exact_tree` | 有根树验证 + 顺序理想 DP | 高效；使用钩子长度公式 |
| `exact_dag` | 2^n 状态上的顺序理想 DP | 一般 DAG，n ≤ 20；O(2^n × n) |
| `exact_dag_sparse` | 有效顺序理想 BFS + 懒惰 dp_ind | 稀疏 DAG，n ≤ 28；内存有界 |
| `approx` | 拓扑排序上的自归一化 IS | 任意 DAG；校正前沿采样器偏差 |

## 特性矩阵

| 特性 | Rust | Python | 状态 |
|------|:----:|:------:|------|
| 精确 ASV（暴力枚举） | ✓ | ✓ | 稳定 |
| 有根树精确 DP | ✓ | ✓ | 实验性 |
| 一般 DAG 精确 DP（n ≤ 20） | ✓ | ✓ | 实验性 |
| 稀疏 DAG 精确 DP（n ≤ 28） | ✓ | ✓ | 实验性 |
| 带 ESS 的近似 ASV | ✓ | ✓ | 实验性 |
| 自适应近似 + CI | ✓ | ✓ | 实验性 |
| 种子确定性并行近似 | ✓ | ✓ | 实验性 |
| 批量联合评估 | ✓ | ✓ | 实验性 |
| sklearn / NumPy 辅助函数（TabularExplainer） | — | ✓ | 实验性 |
| DAG 集成 / 敏感性 ASV | — | ✓ | 实验性 |
| DAG 结构检查 | — | ✓ | 实验性 |
| 图导出（DOT / JSON / networkx） | — | ✓ | 实验性 |

## 论文对应

*Beyond Shapley: Efficient Computation of Asymmetric Shapley Values*

| 算法组件 | causasv |
|---------|---------|
| ASV 定义 | ✓ `exact`（暴力 oracle） |
| 有根树精确算法 | ✓ `exact_tree`（顺序理想 DP + 钩子长度公式） |
| 一般 DAG 精确 DP | ✓ `exact_dag`（顺序理想 DP，n ≤ 20） |
| 一般 DAG 的重要性采样近似 | ✓ `approx` |
| 稀疏 DAG 精确 DP | ✓ `exact_dag_sparse`（顺序理想 BFS，n ≤ 28） |
| 因果发现 | — 超出范围 |

## 性能

Apple M 系列（arm64，release 构建）部分结果。`v(S) = |S|`。完整表格见 [docs/benchmarks.md](docs/benchmarks.md)。

| DAG | n | 方法 | 时间 |
|-----|---|------|------|
| 链式 | 7 | `exact`（暴力） | 2.7 µs |
| 平衡树 | 15 | `exact_tree`（DP） | 2.8 ms |
| 毛毛虫树 | 10 | `exact_tree`（DP） | 170 µs |
| 链式 | 16 | `exact_dag`（密集 DP） | 5.3 ms |
| 链式 | 24 | `exact_dag_sparse` | 15 µs |
| 两条并行链 | 20 | `exact_dag`（密集，100万状态） | **87.9 ms** |
| 两条并行链 | 20 | `exact_dag_sparse`（121状态） | **91 µs**（约1000倍） |
| 链式 | 10 | `approx`（1k 采样） | 916 µs |
| 链式 | 20 | `approx` 串行种子（10k） | 18.2 ms |
| 链式 | 20 | `approx` 并行 4 线程（10k） | 7.4 ms |

使用 `cargo bench` 重现结果。

## 当前限制

- 暴力精确 ASV 对线性扩展数量呈指数级增长；仅适用于 n ≤ ~8 的节点。
- `exact_tree` 需要有根有向树（单根，所有其他节点入度为 1）。n ≤ 20 的一般 DAG 使用 `exact_dag`，n ≤ 28 的稀疏 DAG 使用 `exact_dag_sparse`，更大的 DAG 使用 `approx`。
- 没有内置的因果发现、模型训练或自动图构建。

## 与其他工具的比较

`causasv` 不是 SHAP 的替代品，也不是通用可解释性框架。它解决一个具体问题：

> 在用户提供的因果 DAG 上计算非对称 Shapley 值。

| 工具 | 焦点 | ASV / 因果 DAG |
|------|------|---------------|
| [SHAP](https://github.com/shap/shap) | 通用 Shapley / SHAP | 否 — 仅标准 Shapley |
| [Captum](https://captum.ai/) | PyTorch 模型可解释性 | 否 |
| [shapr](https://github.com/NorskRegnesentral/shapr) | 条件 / 因果 Shapley（R + Python） | 是 — 更广泛的范围，R 优先 |
| [shapflex](https://pypi.org/project/shapflex/) | 带因果知识的 ASV（Python alpha） | 是 — 类似概念 |
| **causasv** | 用户提供因果 DAG 上的 ASV | **核心焦点** |

## 构建 Python 绑定

```bash
cd py
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --features python
python -m pytest tests/
```

## 引用

> Fryer, D., Strümke, I., & Nguyen, H. (2021). *Shapley values for feature selection: The good, the bad, and the axioms.* IEEE Access.

关于非对称公式和高效树计算，请参阅启发本库的论文：

> Beyond Shapley: Efficient Computation of Asymmetric Shapley Values

## 许可证

在以下任一许可证下授权：

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

由您选择。
