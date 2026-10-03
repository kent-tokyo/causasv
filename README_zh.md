# causasv

[![CI](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml/badge.svg)](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/badge/CodeQL-enabled-blue.svg)](https://github.com/kent-tokyo/causasv/security/code-scanning)
[![Crates.io](https://img.shields.io/crates/v/causasv.svg)](https://crates.io/crates/causasv)
[![PyPI](https://img.shields.io/pypi/v/causasv.svg)](https://pypi.org/project/causasv/)
[![Docs.rs](https://docs.rs/causasv/badge.svg)](https://docs.rs/causasv)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

[English](README.md) | [日本語](README_ja.md) | **中文**

面向 Rust 和 Python 的因果非对称 Shapley 值计算库。

`causasv`根据调用方提供的因果 DAG 和价值函数计算非对称 Shapley 值（ASV）。ASV 只在拓扑排序上平均边际贡献，因此原因不会在其后代之后加入联盟。

`causasv`不学习因果图、不训练模型，也不从数据中推断因果关系。它适用于已经拥有因果图，并能在库外说明其假设的场景。

## 安装

PyPI 提供 Linux x86_64、macOS universal2 和 Windows x86_64 wheel：

```bash
pip install causasv
```

Rust：

```toml
[dependencies]
causasv = "0.8"
```

Rust 需要 1.85 或更高版本，Python 需要 3.9 或更高版本。

## ASV 定义

对于 DAG `G`，令`Π(G)`为其全部拓扑排序，`pre(i, π)`为排序`π`中特征`i`之前的特征集合：

```text
φᵢ = 1 / |Π(G)| · Σπ∈Π(G) [v(pre(i,π) ∪ {i}) - v(pre(i,π))]
```

标准 Shapley 值对所有排列求平均；ASV 只使用与给定 DAG 一致的排列。对于加性价值函数，两者相同；当特征交互依赖顺序时，两者可能不同。参见[与 SHAP 的比较](docs/comparison_shap.md)。

## Rust 快速开始

```rust
use causasv::{AsvExplainer, Dag, SamplingConfig};

fn main() -> Result<(), causasv::CausasvError> {
    let mut dag = Dag::new();
    let education = dag.add_node("education");
    let income = dag.add_node("income");
    let risk = dag.add_node("risk_score");
    dag.add_edge(education, income)?;
    dag.add_edge(income, risk)?;

    let result = AsvExplainer::new(dag).auto(
        |coalition| Ok(coalition.len() as f64),
        SamplingConfig::new(10_000).with_seed(42),
    )?;

    println!("{:?}", result.values);
    Ok(())
}
```

`auto()`在可行时选择精确算法，否则回退到带种子的自重要性采样。若近似结果还需要标准误和收敛信息，请使用`auto_quality()`。

## Python 快速开始

推荐使用`explain_quality()`。它先尝试精确算法，无法使用时返回带不确定性诊断的自适应近似结果。

```python
from causasv import ASVExplainer, CausalDAG, explain_quality

dag = CausalDAG.from_edges([
    ("education", "income"),
    ("income", "risk_score"),
])

result = explain_quality(
    ASVExplainer(dag),
    value_fn=lambda features: my_model_score(features),
    seed=42,
    ci=0.95,
)

print(result["values"])
print(result["selected_method"])
print(result["stderr"])
print(result["ci_low"], result["ci_high"])
```

回调接收排序后的`list[str]`，表示联盟中的特征。对于开销较大的模型，可传入`value_fn_batch`代替`value_fn`，在一次 Python 调用中评估多个联盟。

## 选择计算方法

除非正在测试或基准比较特定算法，否则优先使用`auto()`或`auto_quality()`。

| 方法 | 用途 | 限制 |
| --- | --- | --- |
| `exact` | 穷举参考实现 | 实际约为`n <= 8` |
| `exact_tree` | 有根树精确 DP | 受树形状预算约束；位掩码上限`n <= 64` |
| `exact_dag` | 稠密序理想精确 DP | `n <= 20` |
| `exact_dag_sparse` | 稀疏序理想精确 DP | 默认`n <= 28`；配置后最高 63 |
| `uniform_sparse` | 等概率拓扑排序采样 | `n <= 63` |
| `approx` | 重要性采样近似 | 无节点数上限 |

精确计算是否可行还取决于图结构、状态数和内存。正式限制和自动分派规则见[正确性说明](docs/correctness.md#exact-method-bounds)。

使用近似结果时：

- 需要复现时固定随机种子
- 检查标准误、置信区间、收敛状态和回退原因
- 对重要性采样路径检查`ess_ratio`
- 若排序会影响决策，应使用多个种子重复计算

`explain_safe()`会自动执行主要检查，`explain_stability()`报告跨种子的排序稳定性。

## 主要功能

- DAG 验证、拓扑排序和拓扑序枚举
- 穷举、有根树、一般 DAG 和稀疏 DAG 精确 ASV
- 固定样本、自适应、均匀稀疏、并行和批量近似
- 确定性种子、ESS、标准误、置信区间和回退诊断
- Rust 与 Python API
- CPDAG 表示和一致 DAG 扩展
- d-convex 与 strong d-convex 图缩减
- 表格模型和 DAG 敏感性的可选 Python 辅助工具

图缩减同样只处理调用方提供的图，不执行因果发现或效应估计。其假设和独立实现记录见[strong d-convex hull](docs/strong_d_convex_hulls.md)。

## 文档

- [正确性与方法限制](docs/correctness.md)
- [Criterion 基准结果](docs/benchmarks.md)
- [Python 标准基准语料](docs/benchmark_corpus.md)
- [ASV 与 SHAP 比较](docs/comparison_shap.md)
- [strong d-convex hull 假设](docs/strong_d_convex_hulls.md)
- [quietset 不稳定性集成](docs/integrations/quietset_label_instability.md)
- [Rust API](https://docs.rs/causasv)

测量值只适用于文档所列版本、硬件、图和价值函数，不代表一般性能或归因质量保证。

## 当前限制

- 因果图和价值函数必须由调用方提供。
- 即使节点数不大，高度分支的树也可能因组合成本超过`ExactTreeConfig`预算而被`exact_tree`拒绝。
- 稠密和稀疏精确方法受位掩码节点上限约束。重要性采样在 65 个及以上节点时切换到大联盟后端。
- CPDAG 验证检查结构和一致扩展是否存在，但不证明每条有向边都是真实等价类中的必然边。
- 图缩减保证依赖库无法自行验证的分布和图假设。

## 状态

实验性 — v0.8.8。公开 API 在 v1.0 之前可能变化。

小图上的穷举实现是优化算法的正确性参考。已发布和未发布的变更见[CHANGELOG.md](CHANGELOG.md)。

## 构建 Python 扩展

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest
cd py
maturin develop --features python
python -m pytest
```

## 引用

项目引用信息位于[CITATION.cff](CITATION.cff)。ASV 设计受*Beyond Shapley: Efficient Computation of Asymmetric Shapley Values*启发。

图缩减模块独立实现了 Deng、Sun、Li 和 Liu 的“Estimate Collapsibility of Causal Effects in Completed Partial DAGs via Strong d-Convex Hulls,” arXiv:2606.08941 (2026)中的数学定义。参见[引用与适用范围](docs/strong_d_convex_hulls.md)。

## 许可证

可选择[Apache-2.0](LICENSE-APACHE)或[MIT](LICENSE-MIT)许可证。
