# causasv — 基于非对称 Shapley 值的因果特征归因

[![CI](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml/badge.svg)](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/badge/CodeQL-enabled-blue.svg)](https://github.com/kent-tokyo/causasv/security/code-scanning)
[![Security](https://github.com/kent-tokyo/causasv/actions/workflows/security.yml/badge.svg)](https://github.com/kent-tokyo/causasv/actions/workflows/security.yml)
<br>
[![Crates.io](https://img.shields.io/crates/v/causasv.svg)](https://crates.io/crates/causasv)
[![Docs.rs](https://docs.rs/causasv/badge.svg)](https://docs.rs/causasv)
[![Downloads](https://img.shields.io/crates/d/causasv.svg)](https://crates.io/crates/causasv)
[![GitHub release](https://img.shields.io/github/v/release/kent-tokyo/causasv)](https://github.com/kent-tokyo/causasv/releases)
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
            // 用户提供的价值函数：给定特征联合返回分数
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
print(info["values"])     # dict[str, float]
print(info["ess"])        # float — ESS ≈ n_samples 表示可靠；ESS ≪ n_samples 表示方差大
print(info["ess_ratio"])  # float — ESS / n_samples，接近 1 为好
print(info["n_samples"])  # int
print(info["seed"])       # int | None
print(info["is_exact"])   # bool
print(info["method"])     # str
```

检查和导出 DAG：

```python
dag.nodes()   # ["education", "income", "risk_score"]
dag.edges()   # [("education", "income"), ("income", "risk_score")]
dag.to_dot()  # 'digraph {\n  education -> income;\n  income -> risk_score;\n}'

# 转换为 networkx（需单独安装 networkx）
import networkx as nx
G = nx.DiGraph(dag.edges())
```

使用 `make_tabular_value_fn` 将 sklearn 模型封装为价值函数（需要 numpy）：

```python
from causasv import make_tabular_value_fn

value_fn = make_tabular_value_fn(
    model=my_classifier,      # 任何 sklearn 兼容的模型
    x=X_test[0],             # 待解释的实例，形状 (n_features,)
    background=X_train,      # 参考数据集；列均值作为缺失特征的基准
    feature_names=["education", "income", "risk_score"],
)
values = explainer.explain(value_fn, method="auto")
```

## 精确计算 vs 近似计算

| 方法 | 适用场景 | API |
|------|---------|-----|
| `exact` | 小型 DAG（n ≤ ~8）；枚举所有线性扩展 | `explainer.exact(value_fn)` |
| `exact_tree` | 有根有向树；顺序理想 DP | `explainer.exact_tree(value_fn)` |
| `exact_dag` | 一般 DAG，n ≤ 20；顺序理想 DP | `explainer.exact_dag(value_fn)` |
| `approx` | 任意 DAG（n > 20）；重要性加权采样 | `explainer.approximate(value_fn, SamplingConfig::new(n))` |

`auto` 调度：n ≤ 8 → `exact`；有根树 → `exact_tree`；n ≤ 20 → `exact_dag`；否则 → `approx`。

近似估计器使用自归一化重要性采样来校正前沿采样器引入的偏差，因此即使对于近似结果，效率公理（Σφ_i = v(V) − v(∅)）也精确成立。

## 状态

实验性 — v0.6.0。在 v1.0 之前公共 API 可能会发生变化。

## 算法状态

| 方法 | 实现 | 备注 |
|------|------|------|
| `exact` | 枚举所有线性扩展 | 参考 oracle；实用范围 n ≤ ~8 |
| `exact_tree` | 有根树验证 + 顺序理想 DP | 高效；使用钩子长度公式 |
| `exact_dag` | 2^n 状态上的顺序理想 DP | 一般 DAG，n ≤ 20；O(2^n × n) |
| `approx` | 拓扑排序上的自归一化 IS | 任意 DAG；校正前沿采样器偏差 |

## 特性矩阵

| 特性 | Rust | Python | 状态 |
|------|:----:|:------:|------|
| 精确 ASV（暴力枚举） | ✓ | ✓ | 稳定 |
| 有根树精确 DP | ✓ | ✓ | 实验性 |
| 一般 DAG 精确 DP（n ≤ 20） | ✓ | ✓ | 实验性 |
| 带 ESS 的近似 ASV | ✓ | ✓ | 实验性 |
| 自适应近似 | ✓ | ✓ | 实验性 |
| sklearn / NumPy 辅助函数 | — | ✓ | 实验性 |
| 图导出（DOT / networkx） | planned | ✓ | 实验性 |

## 论文对应

*Beyond Shapley: Efficient Computation of Asymmetric Shapley Values*

| 算法组件 | causasv |
|---------|---------|
| ASV 定义 | ✓ `exact`（暴力 oracle） |
| 有根树精确算法 | ✓ `exact_tree`（顺序理想 DP + 钩子长度公式） |
| 一般 DAG 精确 DP | ✓ `exact_dag`（顺序理想 DP，n ≤ 20） |
| 一般 DAG 的重要性采样近似 | ✓ `approx` |
| 稀疏/内存受限精确 DAG DP | planned（`exact_dag_sparse`，n > 20） |
| 因果发现 | — 超出范围 |

## 性能

Apple M 系列（arm64，release 构建）基准测试。`v(S) = |S|`（可加价值函数）。

| 基准测试 | n | L(T) | 方法 | 时间 |
|---------|---|-------|------|------|
| 平衡二叉树 | 7 | 80 | `exact`（枚举） | ~70 µs |
| 平衡二叉树 | 7 | 80 | `exact_tree`（DP） | ~145 µs |
| 平衡二叉树 | 15 | ~22 M | `exact` | — （不可行） |
| 平衡二叉树 | 15 | ~22 M | `exact_tree`（DP） | ~7.8 ms |
| 毛毛虫树 | 10 | 945 | `exact_tree`（DP） | ~347 µs |
| 近似（链式） | 10 | — | `approx`（1k 采样） | ~2.9 ms |

使用 `cargo bench` 重现结果。

## 当前限制

- 暴力精确 ASV 对线性扩展数量呈指数级增长；仅适用于 n ≤ ~8 的节点。
- `exact_tree` 需要有根有向树（单根，所有其他节点入度为 1）。对于 n ≤ 20 的一般 DAG，使用 `exact_dag`；更大的 DAG 使用 `approx`。
- Python 绑定提供 `nodes()`、`edges()`、`to_dot()` 和 `make_tabular_value_fn`；图级 DOT 导出有效，但 Rust 侧导出尚未实现。
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
