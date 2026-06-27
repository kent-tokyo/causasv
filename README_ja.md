# causasv

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

[English](README.md) | **日本語** | [中文](README_zh.md)

Rust・Python 向けの高速な因果的非対称 Shapley 値（ASV）計算ライブラリ。

`causasv` は、ユーザが提供する因果 DAG に基づいて非対称 Shapley 値を計算する Rust ファーストのエンジンです。特徴量の帰属（フィーチャーアトリビューション）が既知の因果構造を尊重すべき XAI ワークフロー向けに設計されています。

このクレートは因果グラフを学習しません。ユーザが有効な有向非巡回グラフ（DAG）と価値関数を提供することを前提とします。

## これは何ではないか

- 因果探索ツールではありません — DAG はユーザが用意します
- 汎用 SHAP の代替ではありません — SHAP ではなく ASV を計算します
- モデルの訓練器や特徴量選択器ではありません
- 深層学習向けの説明可能性フレームワークではありません

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
            // ユーザ定義の価値関数：特徴量連合のスコアを返す
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

```python
from causasv import CausalDAG, ASVExplainer

# エッジリストから DAG を構築
dag = CausalDAG.from_edges([("education", "income"), ("income", "risk_score")])

# networkx DiGraph からも構築可能
# import networkx as nx; G = nx.DiGraph(); G.add_edge(...)
# dag = CausalDAG.from_networkx(G)

explainer = ASVExplainer(dag)

values = explainer.explain(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="auto",   # n≤8 なら exact、有根木なら exact_tree、それ以外は approx
    n_samples=10_000,
    seed=42,
)
# values: dict[str, float]  特徴量名 → ASV 値
```

Python の `value_fn` は連合内の特徴量名のソート済みリストを受け取り、float を返す必要があります。

`explain_with_diagnostics()` を使うと ESS・シード・メソッド名も取得できます：

```python
info = explainer.explain_with_diagnostics(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="approx",
    n_samples=10_000,
    seed=42,
)
print(info["values"])     # dict[str, float]
print(info["ess"])        # float | None — ESS ≈ n_samples なら信頼できる
print(info["ess_ratio"])  # float | None — ESS / n_samples ∈ (0, 1]
print(info["n_samples"])  # int
print(info["seed"])       # int | None
print(info["is_exact"])   # bool
print(info["method"])     # str
```

DAG の操作とエクスポート：

```python
dag.nodes()   # ["education", "income", "risk_score"]
dag.edges()   # [("education", "income"), ("income", "risk_score")]
dag.to_dot()  # 'digraph {\n  education -> income;\n  ...}'

# networkx への変換（networkx を別途インストール）
import networkx as nx
G = nx.DiGraph(dag.edges())
```

sklearn モデルを value_fn に変換する `make_tabular_value_fn`（numpy が必要）：

```python
from causasv import make_tabular_value_fn

value_fn = make_tabular_value_fn(
    model=my_classifier,
    x=X_test[0],
    background=X_train,
    feature_names=["education", "income", "risk_score"],
)
values = explainer.explain(value_fn, method="auto")
```

## Exact vs 近似

| メソッド | 使用場面 | API |
|---------|---------|-----|
| `exact` | 小規模 DAG（n ≤ ~8）：全線形拡張を列挙 | `explainer.exact(value_fn)` |
| `exact_tree` | 有根有向木：順序イデアル DP | `explainer.exact_tree(value_fn)` |
| `exact_dag` | 一般 DAG（n ≤ 20）：順序イデアル DP | `explainer.exact_dag(value_fn)` |
| `approx` | 任意の DAG（n > 20）：重要度重み付きサンプリング | `explainer.approximate(value_fn, SamplingConfig::new(n))` |

`auto` ディスパッチ：n ≤ 8 → `exact`；有根木 → `exact_tree`；n ≤ 20 → `exact_dag`；それ以外 → `approx`。

近似推定器は自己正規化重要度サンプリングを使用してフロンティアサンプラーのバイアスを補正します。そのため近似結果でも効率性公理（Σφ_i = v(V) − v(∅)）が厳密に成立します。

結果には `effective_sample_size`（ESS = (Σw)² / Σw²）が含まれます。ESS ≈ n_samples は IS 重みが均一で推定が信頼できることを示し、ESS ≪ n_samples は重みの分散が大きいことを示します。

## ステータス

実験的 — v0.6.0。v1.0 以前に公開 API が変更される可能性があります。

## アルゴリズムの状況

| メソッド | 実装 | 備考 |
|---------|------|------|
| `exact` | 全線形拡張を列挙 | 参照オラクル；n ≤ ~8 で実用的 |
| `exact_tree` | 有根木検証 + 順序イデアル DP | 効率的；フック長公式で重み付け |
| `exact_dag` | 2^n 状態の順序イデアル DP | 一般 DAG、n ≤ 20；O(2^n × n) |
| `approx` | トポロジー的順序付けの自己正規化 IS | フロンティアサンプラーのバイアスを補正 |

ブルートフォース `exact` はすべての他のメソッドのテストにおいて参照オラクルとして使用されます。

`exact_tree` の DP は順序イデアルを介して有効な前集合を列挙し、フック長公式で各集合に重みを付けます。深さ 30 のキャタピラー木ではブルートフォースに対して桁違いの高速化が見られます。

`exact_dag` の DP は全 2^n ビットマスクに対して 2 つのテーブルを計算します：`dp_fwd[S]`（有効な順序イデアル S の順序付け数）と `dp_ind[T]`（任意の誘導部分グラフ G[T] の線形拡張数）。これは木から任意の DAG に一般化された順序イデアル DP です。

## 論文との対応

*Beyond Shapley: Efficient Computation of Asymmetric Shapley Values*

| アルゴリズムコンポーネント | causasv |
|--------------------------|---------|
| ASV の定義 | ✓ `exact`（ブルートフォースオラクル） |
| 有根木の厳密アルゴリズム | ✓ `exact_tree`（順序イデアル DP + フック長公式） |
| 一般 DAG の厳密 DP | ✓ `exact_dag`（順序イデアル DP、n ≤ 20） |
| 一般 DAG の重要度サンプリング近似 | ✓ `approx` |
| 一般 DAG の最適化 DP | 計画中 |
| 因果探索 | — スコープ外 |

## フィーチャーマトリクス

| 機能 | Rust | Python | ステータス |
|------|:----:|:------:|-----------|
| 厳密 ASV（ブルートフォース） | ✓ | ✓ | 安定 |
| 有根木の厳密 DP | ✓ | ✓ | 実験的 |
| 一般 DAG の厳密 DP（n ≤ 20） | ✓ | ✓ | 実験的 |
| ESS 付き近似 ASV | ✓ | ✓ | 実験的 |
| 適応型近似 | ✓ | ✓ | 実験的 |
| sklearn / NumPy ヘルパー | — | ✓ | 実験的 |
| グラフエクスポート（DOT / networkx） | 計画中 | ✓ | 実験的 |

## パフォーマンス

Apple M シリーズ（arm64、リリースビルド）でのベンチマーク。`v(S) = |S|`（加法的価値関数）。

| ベンチマーク | n | L(T) | メソッド | 時間 |
|------------|---|------|---------|------|
| バランス二分木 | 7 | 80 | `exact`（列挙） | ~70 µs |
| バランス二分木 | 7 | 80 | `exact_tree`（DP） | ~145 µs |
| バランス二分木 | 15 | ~22 M | `exact` | — （実行不可能） |
| バランス二分木 | 15 | ~22 M | `exact_tree`（DP） | ~7.8 ms |
| キャタピラー木 | 10 | 945 | `exact_tree`（DP） | ~347 µs |
| 近似（チェーン） | 10 | — | `approx`（1k サンプル） | ~2.9 ms |

> n=15 の厳密列挙では約 2200 万の有効な因果順序を訪問する必要があります；
> `exact_tree` は順序イデアル DP により同じ ASV をミリ秒で計算します。

注意：n ≤ ~8 では `exact` がアロケータのオーバーヘッドが低いため `exact_tree` より速いことが多いです。
`cargo bench` で再現できます。

## 現在の制限事項

- ブルートフォース exact ASV は線形拡張の数に対して指数的；n ≤ ~8 ノードでのみ実用的。
- `exact_tree` は有根有向木（単一ルート、他の全ノードの入次数が 1）を必要とします。一般 DAG には `exact_dag`（n ≤ 20）または `approx` を使用してください。
- Python バインディングは `nodes()`・`edges()`・`to_dot()`・`make_tabular_value_fn` を提供；Rust 側のグラフエクスポートは未実装。
- 組み込みの因果探索、モデル訓練、自動グラフ構築はありません。

## 他ツールとの比較

`causasv` は SHAP の代替や汎用の説明可能性フレームワークではありません。
1 つの狭い問題を解決します：

> ユーザが提供する因果 DAG に対する非対称 Shapley 値の計算。

| ツール | 焦点 | ASV / 因果 DAG |
|-------|------|---------------|
| [SHAP](https://github.com/shap/shap) | 汎用 Shapley / SHAP | なし — 標準 Shapley のみ |
| [Captum](https://captum.ai/) | PyTorch モデル解釈可能性 | なし |
| [shapr](https://github.com/NorskRegnesentral/shapr) | 条件付き / 因果 Shapley（R + Python） | あり — より広いスコープ、R ファースト |
| [shapflex](https://pypi.org/project/shapflex/) | 因果知識を用いた ASV（Python アルファ） | あり — 類似コンセプト |
| **causasv** | ユーザ提供の因果 DAG に対する ASV | **コアフォーカス** |

`shapr` および `shapflex` との主な違い：`causasv` は明示的な因果 DAG と価値関数をユーザが提供することを要求する Rust ファーストエンジンです。因果探索を行わず、データ分布に依存しません。

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
