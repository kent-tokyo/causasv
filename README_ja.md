# causasv — 非対称 Shapley 値による因果特徴量帰属

[![CI](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml/badge.svg)](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/causasv.svg)](https://crates.io/crates/causasv)
[![PyPI](https://img.shields.io/pypi/v/causasv.svg)](https://pypi.org/project/causasv/)
[![Docs.rs](https://docs.rs/causasv/badge.svg)](https://docs.rs/causasv)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
<br>
[![MSRV](https://img.shields.io/badge/MSRV-1.85%2B-orange.svg)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://doc.rust-lang.org/nomicon/meet-safe-and-unsafe.html)

[English](README.md) | **日本語** | [中文](README_zh.md)

`causasv` はユーザが提供する DAG 上で**非対称 Shapley 値（ASV）**による因果特徴量帰属を計算します。Rust ファーストのエンジンに Python バインディングを備え、既知の因果構造を尊重した特徴量重要度を必要とする XAI ワークフロー向けに設計されています。

## causasv を使うべき場面

**causasv を使う：**
- 特徴量間の既知の因果 DAG がある
- 標準 SHAP が子孫や媒介変数を通じた帰属を行う可能性がある
- 厳密または不確実性付きの近似 ASV が必要
- Rust コアと Python バインディングで CI 付き推定が欲しい

**causasv を使わない：**
- 因果 DAG がない場合（→ SHAP や Captum を使う）
- 因果構造なしの汎用モデル説明可能性が必要な場合
- ディープラーニングの層・ニューロン帰属が必要な場合
- 因果効果の推定や探索が必要な場合（→ DoWhy を使う）

## ASV とは

非対称 Shapley 値（ASV）は、すべての順列ではなく**トポロジー的に有効な順序付け**のみで平均を取ることで Shapley 値を一般化します。因果 DAG G と価値関数 v が与えられたとき：

```
φ_i = (1 / |Π(G)|) Σ_{π ∈ Π(G)} [v(pre(i,π) ∪ {i}) − v(pre(i,π))]
```

ここで Π(G) は G のすべての線形拡張（トポロジー的順序付け）の集合、pre(i,π) は順序付け π において特徴量 i より前に現れる特徴量の集合です。

## ASV と SHAP の違い

標準的な SHAP は因果構造を無視してすべての n! 順列で平均を取ります。ASV は因果 DAG と整合する順列（原因が常に結果より前に来る）のみに平均を制限します。これにより因果性の方向を尊重した帰属が得られます。

## なぜ因果 DAG が重要なのか

特徴量に因果関係がある場合、SHAP はある変数にその子孫を介して媒介された効果の帰属を割り当てることがあります。ASV は有効とみなす順序付けを制限することでこれを防ぎます。

## インストール

```bash
pip install causasv
```

Linux (x86_64 manylinux)・macOS (universal2)・Windows (x86_64) 向け wheel を [PyPI](https://pypi.org/project/causasv/) で公開しています。Rust の場合は `Cargo.toml` に追加してください：

```toml
[dependencies]
causasv = "0.8"
```

## Rust の使用例

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

    // 重要度重み付きトポロジー順序サンプリングによる近似 ASV
    let values = explainer.approximate(
        |coalition| {
            Ok(coalition.len() as f64)
        },
        SamplingConfig::new(10_000).with_seed(42),
    )?;

    for (node, value) in &values.values {
        println!("Node {:?}: ASV = {:.4}", node, value);
    }
    Ok(())
}
```

## Python の使用例

ほとんどのユーザーには `explain_quality()` が推奨入口です。実行可能なら厳密計算、そうでなければ信頼区間付きの近似にフォールバックします：

```python
from causasv import CausalDAG, ASVExplainer, explain_quality

dag = CausalDAG.from_edges([("education", "income"), ("income", "risk_score")])
explainer = ASVExplainer(dag)

info = explain_quality(
    explainer,
    value_fn=lambda feature_names: my_model_score(feature_names),
    ci=0.95,   # 95% 信頼区間を含める
    seed=42,
)
print(info["values"])           # dict[str, float] — 特徴量ごとの ASV 値
print(info["ci_low"])           # dict[str, float] — 95% CI 下限
print(info["ci_high"])          # dict[str, float] — 95% CI 上限
print(info["selected_method"])  # 例: "exact_dag_sparse", "uniform_sparse_adaptive", "uniform_sparse_adaptive_batch"
print(info["stderr"])           # dict[str, float] — 特徴量ごとの標準誤差
```

Python の `value_fn` は連合内の特徴量名のソート済みリストを受け取り、float を返す必要があります。

**低レベル API** — メソッドを明示指定する場合は `explain_with_diagnostics()` を使用：

```python
info = explainer.explain_with_diagnostics(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="approx",
    n_samples=10_000,
    seed=42,
)
print(info["values"])          # dict[str, float]
print(info["ess"])             # float — ESS ≈ n_samples なら信頼できる
print(info["ess_ratio"])       # float — ESS / n_samples ∈ (0, 1]
print(info["method"])          # str — 入力メソッド名
print(info["selected_method"]) # str — auto() が実際に選んだメソッド
print(info["fallback_from"])   # str | None — フォールバック元
```

`explain_adaptive()` で自動収束検出と特徴量ごとの信頼区間を取得：

```python
info = explainer.explain_adaptive(
    value_fn=lambda feature_names: my_model_score(feature_names),
    min_samples=1_000,
    max_samples=100_000,
    batch_size=1_000,
    seed=42,
    ci=0.95,          # 省略可能：ci_low / ci_high を追加
)
print(info["values"])     # dict[str, float]
print(info["stderr"])     # dict[str, float] — 特徴量ごとの IS 標準誤差
print(info["ci_low"])     # dict[str, float] — 95% 信頼区間の下限
print(info["ci_high"])    # dict[str, float] — 95% 信頼区間の上限
print(info["converged"])  # bool — max_samples 前に rel_tol を達成したか
print(info["ess_ratio"])  # float — ESS / n_samples
```

大規模モデルで coalition ごとの呼び出しが遅い場合は `value_fn_batch` を `explain_quality()` に渡します。n ≤ 63 では一様スパース適応バッチサンプリング（ESS = n_samples、IS 分散なし、CI 付き）を使用します：

```python
# value_fn_batch は list[list[str]] を受け取り list[float] を返す
info = explain_quality(
    explainer,
    value_fn_batch=lambda coalitions: [my_model_score(c) for c in coalitions],
    ci=0.95,
    seed=42,
)
print(info["values"])           # dict[str, float]
print(info["ci_low"])           # dict[str, float]
print(info["selected_method"])  # "uniform_sparse_adaptive_batch"（n>63 では "approx_adaptive_batch"）
```

バッチパスは Python GIL の往復を O(n × batch_size) から O(unique_masks_per_batch) に削減します。IS 適応バッチを明示的に使用する場合は `explainer.explain_adaptive_batch()` を直接呼び出してください。

決定論的並列近似には `seed` と `parallel=True` を組み合わせる：

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

`explain_stability()` で近似ランキングの seed 間安定性を確認：

```python
from causasv import explain_stability

result = explain_stability(
    explainer,
    value_fn=lambda feature_names: my_model_score(feature_names),
    seeds=[1, 2, 3, 4, 5],
    method="approx",
    n_samples=10_000,
)
print(result["rank_stability"])  # 平均 Kendall tau；1.0 = 完全安定
print(result["std_values"])      # dict[str, float] — 小さいほど安定
print(result["mean_values"])     # dict[str, float] — seed 間の平均 ASV
```

複数の候補 DAG 間の感度分析には `ASVEnsembleExplainer`：

```python
from causasv import CausalDAG, ASVEnsembleExplainer

dag1 = CausalDAG.from_edges([("A", "B"), ("B", "C")])
dag2 = CausalDAG.from_edges([("A", "B"), ("A", "C")])
ensemble = ASVEnsembleExplainer([dag1, dag2])
result = ensemble.explain_with_sensitivity(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="auto",
)
print(result["mean_values"])     # dict[str, float] — DAG 間の平均 ASV
print(result["std_values"])      # dict[str, float] — DAG 間の標準偏差
print(result["rank_stability"])  # float — 平均 Kendall tau
print(result["per_dag_values"])  # list[dict[str, float]]
```

DAG の操作とエクスポート：

```python
dag.nodes()                     # ["education", "income", "risk_score"]
dag.edges()                     # [("education", "income"), ("income", "risk_score")]
dag.to_dot()                    # 'digraph {\n  education -> income;\n  ...\n}'
dag.to_json()                   # '{"nodes":[...],"edges":[...]}'
dag.ancestors("risk_score")     # ["education", "income"]
dag.descendants("education")    # ["income", "risk_score"]
dag.topological_layers()        # [["education"], ["income"], ["risk_score"]]

# JSON から DAG を復元
dag2 = CausalDAG.from_json(dag.to_json())

# networkx への変換（networkx を別途インストール）
import networkx as nx
G = nx.DiGraph(dag.edges())
```

sklearn 互換モデル向けの高レベル API `TabularExplainer`（numpy が必要）：

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

## 厳密計算 vs 近似計算

| メソッド | 使用場面 | API |
|---------|---------|-----|
| `exact` | 小規模 DAG（n ≤ ~8）：全線形拡張を列挙 | `explainer.exact(value_fn)` |
| `exact_tree` | 有根有向木：順序イデアル DP | `explainer.exact_tree(value_fn)` |
| `exact_dag` | 一般 DAG、n ≤ 20；密な順序イデアル DP | `explainer.exact_dag(value_fn)` |
| `exact_dag_sparse` | 疎な DAG、n ≤ 28；有効な順序イデアルのみ BFS | `explainer.exact_dag_sparse(value_fn)` |
| `approx` | 任意の DAG（n > 28 またはメモリ制限超過）；IS サンプリング | `explainer.approximate(value_fn, SamplingConfig::new(n))` |

`auto` ディスパッチ：n ≤ 8 → `exact`；有根木 → `exact_tree`；n ≤ 20 → edge_count ≤ 2n なら `exact_dag_sparse` そうでなければ `exact_dag`；20 < n ≤ 28 → `exact_dag_sparse`；28 < n ≤ 63 → 順序イデアル数 ≤ 250k なら `exact_dag_sparse`（疎プリフライト）そうでなければ `approx`；n > 63 → `approx`。

`exact_dag_sparse` は有効な順序イデアル（すべてのノードの親も存在する集合）のみを訪問します。疎な DAG では 2^n よりはるかに少ない状態で厳密計算が可能です。`n_order_ideals`・`state_ratio`・`memory_mb` の診断値を返します。

近似推定器は自己正規化重要度サンプリングを使用してフロンティアサンプラーのバイアスを補正するため、近似結果でも効率性公理（Σφ_i = v(V) − v(∅)）が厳密に成立します。

## ステータス

実験的 — v0.8.5。v1.0 以前に公開 API が変更される可能性があります。

## アルゴリズムの状況

| メソッド | 実装 | 備考 |
|---------|------|------|
| `exact` | 全線形拡張を列挙 | 参照オラクル；n ≤ ~8 で実用的 |
| `exact_tree` | 有根木検証 + 順序イデアル DP | 効率的；フック長公式 |
| `exact_dag` | 2^n 状態の順序イデアル DP | 一般 DAG、n ≤ 20；O(2^n × n) |
| `exact_dag_sparse` | 有効な順序イデアルの BFS + 遅延 dp_ind | 疎 DAG、n ≤ 28；メモリ制限付き |
| `approx` | トポロジー的順序付けの自己正規化 IS | 任意 DAG；フロンティアサンプラーのバイアスを補正 |

## フィーチャーマトリクス

| 機能 | Rust | Python | ステータス |
|------|:----:|:------:|-----------|
| 厳密 ASV（ブルートフォース） | ✓ | ✓ | 安定 |
| 有根木の厳密 DP | ✓ | ✓ | 実験的 |
| 一般 DAG の厳密 DP（n ≤ 20） | ✓ | ✓ | 実験的 |
| 疎 DAG の厳密 DP（n ≤ 28） | ✓ | ✓ | 実験的 |
| ESS 付き近似 ASV | ✓ | ✓ | 実験的 |
| 適応型近似 + CI | ✓ | ✓ | 実験的 |
| シード付き決定論的並列近似 | ✓ | ✓ | 実験的 |
| バッチ連合評価 | ✓ | ✓ | 実験的 |
| sklearn / NumPy ヘルパー（TabularExplainer） | — | ✓ | 実験的 |
| DAG アンサンブル / 感度 ASV | — | ✓ | 実験的 |
| DAG 構造検査 | — | ✓ | 実験的 |
| グラフエクスポート（DOT / JSON / networkx） | — | ✓ | 実験的 |

## 論文との対応

*Beyond Shapley: Efficient Computation of Asymmetric Shapley Values*

| アルゴリズムコンポーネント | causasv |
|--------------------------|---------|
| ASV の定義 | ✓ `exact`（ブルートフォースオラクル） |
| 有根木の厳密アルゴリズム | ✓ `exact_tree`（順序イデアル DP + フック長公式） |
| 一般 DAG の厳密 DP | ✓ `exact_dag`（順序イデアル DP、n ≤ 20） |
| 一般 DAG の重要度サンプリング近似 | ✓ `approx` |
| 疎 DAG の厳密 DP | ✓ `exact_dag_sparse`（順序イデアル BFS、n ≤ 28） |
| 因果探索 | — スコープ外 |

## パフォーマンス

Apple M シリーズ（arm64、リリースビルド）での選択結果。`v(S) = |S|`。詳細は [docs/benchmarks.md](docs/benchmarks.md) を参照。

| DAG | n | メソッド | 時間 |
|-----|---|---------|------|
| チェーン | 7 | `exact`（ブルートフォース） | 2.7 µs |
| バランス木 | 15 | `exact_tree`（DP） | 2.8 ms |
| キャタピラー | 10 | `exact_tree`（DP） | 170 µs |
| チェーン | 10 | `exact_dag`（密 DP） | **23 µs** |
| チェーン | 16 | `exact_dag`（密 DP、65k 状態） | 3.0 ms |
| チェーン | 16 | `exact_dag_sparse`（17 順序イデアル、`auto` 経由） | **11 µs**（約280倍速） |
| チェーン | 24 | `exact_dag_sparse` | 15 µs |
| 2本の並列チェーン | 20 | `exact_dag`（密、100万状態） | **55 ms** |
| 2本の並列チェーン | 20 | `exact_dag_sparse`（121状態） | **91 µs**（約600倍速） |
| ダイヤモンド | 10 | `approx` シード付き（10k サンプル） | **16 ms** |
| ダイヤモンド | 10 | `approximate_adaptive_batched`（10k 上限） | 2.4 ms |
| チェーン | 20 | `approx` シード付き直列（10k） | **19 ms** |
| チェーン | 20 | `approx` 並列 4 スレッド（10k） | 7.4 ms |
| バランス木 | 31 | `approx` シード付き（10k サンプル） | 83 ms |

`cargo bench` で再現できます。

## 現在の制限事項

- ブルートフォース exact ASV は線形拡張の数に対して指数的；n ≤ ~8 ノードでのみ実用的。
- `exact_tree` は有根有向木（単一ルート、他の全ノードの入次数が 1）を必要とします。n ≤ 20 の一般 DAG には `exact_dag`、n ≤ 28 の疎 DAG には `exact_dag_sparse`、それより大きい DAG には `approx` を使用してください。
- 組み込みの因果探索、モデル訓練、自動グラフ構築はありません。

## 他ツールとの比較

`causasv` は SHAP の代替や汎用の説明可能性フレームワークではありません。1 つの狭い問題を解決します：

> ユーザが提供する因果 DAG に対する非対称 Shapley 値の計算。

| ツール | 焦点 | ASV / 因果 DAG |
|-------|------|---------------|
| [SHAP](https://github.com/shap/shap) | 汎用 Shapley / SHAP | なし — 標準 Shapley のみ |
| [Captum](https://captum.ai/) | PyTorch モデル解釈可能性 | なし |
| [shapr](https://github.com/NorskRegnesentral/shapr) | 条件付き / 因果 Shapley（R + Python） | あり — より広いスコープ、R ファースト |
| [shapflex](https://pypi.org/project/shapflex/) | 因果知識を用いた ASV（Python アルファ） | あり — 類似コンセプト |
| **causasv** | ユーザ提供の因果 DAG に対する ASV | **コアフォーカス** |

## Python バインディングのビルド

```bash
cd py
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --features python
python -m pytest tests/
```

## 引用

> Fryer, D., Strümke, I., & Nguyen, H. (2021). *Shapley values for feature selection: The good, the bad, and the axioms.* IEEE Access.

非対称定式化と効率的な木計算については、このライブラリにインスピレーションを与えた論文を参照してください：

> Beyond Shapley: Efficient Computation of Asymmetric Shapley Values

## ライセンス

以下のいずれかのライセンスの下で提供されます：

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

お好みのものをご選択ください。
