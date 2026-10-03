# causasv

[![CI](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml/badge.svg)](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/badge/CodeQL-enabled-blue.svg)](https://github.com/kent-tokyo/causasv/security/code-scanning)
[![Crates.io](https://img.shields.io/crates/v/causasv.svg)](https://crates.io/crates/causasv)
[![PyPI](https://img.shields.io/pypi/v/causasv.svg)](https://pypi.org/project/causasv/)
[![Docs.rs](https://docs.rs/causasv/badge.svg)](https://docs.rs/causasv)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

[English](README.md) | **日本語** | [中文](README_zh.md)

RustとPythonで使える、因果構造を考慮した非対称Shapley値（ASV）の計算ライブラリです。

`causasv`は、利用者が用意した因果DAGと値関数からASVを計算します。ASVは位相順序だけを使って限界寄与を平均するため、原因が子孫より後に追加される順序を除外できます。

因果グラフの学習、モデル訓練、データからの因果推論は行いません。グラフを別途用意でき、その前提を利用者側で説明できる場合に使います。

## インストール

Python向けには、Linux x86_64・macOS universal2・Windows x86_64用のwheelを公開しています。

```bash
pip install causasv
```

Rust:

```toml
[dependencies]
causasv = "0.8"
```

Rust 1.85以降、Python 3.9以降に対応します。

## ASVの定義

DAG `G`の位相順序全体を`Π(G)`、順序`π`で特徴量`i`より前に現れる集合を`pre(i, π)`とします。

```text
φᵢ = 1 / |Π(G)| · Σπ∈Π(G) [v(pre(i,π) ∪ {i}) - v(pre(i,π))]
```

通常のShapley値はすべての順列を平均します。ASVは、与えられたDAGと整合する順列だけを使います。値関数が加法的なら両者は一致しますが、相互作用が順序に依存する場合は異なります。具体例は[SHAPとの比較](docs/comparison_shap.md)を参照してください。

## Rustの基本例

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

`auto()`は実行可能なら厳密法を選び、難しい場合はシード付き重要度サンプリングへ切り替えます。近似結果の標準誤差や収束情報も必要なら`auto_quality()`を使います。

## Pythonの基本例

Pythonでは`explain_quality()`を推奨します。まず厳密法を試し、適用できない場合は不確実性情報付きの適応型近似を返します。

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

コールバックには、連合に含まれる特徴量名をソートした`list[str]`が渡されます。モデル評価が重い場合は、`value_fn`の代わりに`value_fn_batch`を渡すと、複数の連合をまとめて評価できます。

## 計算方法の選び方

特定の方式を検証する場合を除き、`auto()`または`auto_quality()`を使ってください。

| 方式 | 用途 | 上限 |
| --- | --- | --- |
| `exact` | 総当たりの参照実装 | 実用上はおおむね`n <= 8` |
| `exact_tree` | 有根木向け厳密DP | 木の形状に応じた予算、ビットマスク上限`n <= 64` |
| `exact_dag` | 密な順序イデアルDP | `n <= 20` |
| `exact_dag_sparse` | 疎な順序イデアルDP | 既定は`n <= 28`、設定変更で63まで |
| `uniform_sparse` | 位相順序の一様サンプリング | `n <= 63` |
| `approx` | 重要度サンプリング近似 | ノード数上限なし |

厳密計算が可能かどうかは、ノード数だけでなくグラフ構造、状態数、メモリにも左右されます。正式な上限と自動選択規則は[正しさと計算方式](docs/correctness.md#exact-method-bounds)にまとめています。

近似結果を使うときは、次を確認してください。

- 再現性が必要ならシードを固定する
- 標準誤差、信頼区間、収束状況、フォールバック理由を確認する
- 重要度サンプリングでは`ess_ratio`を確認する
- 順位を意思決定に使う場合は複数シードで再計算する

`explain_safe()`は主要な確認を自動化します。`explain_stability()`はシード間の順位安定性を返します。

## 主な機能

- DAG検証、位相ソート、位相順序列挙
- 総当たり、有根木、一般DAG、疎DAG向けの厳密ASV
- 固定標本、適応型、一様、並列、バッチ型の近似ASV
- シード再現性、ESS、標準誤差、信頼区間、フォールバック診断
- Rust APIとPython API
- CPDAG表現と整合するDAG拡張
- d-convex hullとstrong d-convex hullによるグラフ縮約
- 表形式モデルとDAG感度分析のPythonヘルパー

グラフ縮約も利用者が用意したグラフを対象とします。因果発見や効果推定は行いません。前提条件と独立実装の記録は[strong d-convex hull](docs/strong_d_convex_hulls.md)にあります。

## ドキュメント

- [正しさと計算方式の上限](docs/correctness.md)
- [Criterionベンチマーク](docs/benchmarks.md)
- [Pythonの標準ベンチマークコーパス](docs/benchmark_corpus.md)
- [ASVとSHAPの比較](docs/comparison_shap.md)
- [strong d-convex hullの前提](docs/strong_d_convex_hulls.md)
- [quietset不安定性連携](docs/integrations/quietset_label_instability.md)
- [Rust API](https://docs.rs/causasv)

計測値が示すのは、記載されたバージョン、ハードウェア、グラフ、値関数での結果です。一般的な速度や帰属品質を保証するものではありません。

## 現在の制限

- 因果グラフと値関数は利用者が用意する必要があります。
- `exact_tree`はノード数が少なくても、分岐が多く組合せコストが`ExactTreeConfig`の予算を超える木を拒否します。
- 密・疎の厳密法にはビットマスク由来のノード数上限があります。重要度サンプリングは65ノード以上で大規模連合用バックエンドへ切り替わります。
- CPDAG検証は構造と整合拡張の存在を確認しますが、すべての有向辺が真の同値類で必須かどうかまでは証明しません。
- グラフ縮約の保証には、ライブラリ側では検証できない分布・グラフ上の前提があります。

## ステータス

実験的 — v0.8.9。v1.0までに公開APIが変わる可能性があります。

小規模グラフでは総当たり実装を、最適化された方式の正しさを確認する基準として使っています。変更履歴は[CHANGELOG.md](CHANGELOG.md)を参照してください。

## Python拡張のビルド

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest
cd py
maturin develop --features python
python -m pytest
```

## 引用

プロジェクトの引用情報は[CITATION.cff](CITATION.cff)にあります。ASV設計は*Beyond Shapley: Efficient Computation of Asymmetric Shapley Values*を参考にしています。

グラフ縮約モジュールは、Deng、Sun、Li、Liuによる“Estimate Collapsibility of Causal Effects in Completed Partial DAGs via Strong d-Convex Hulls,” arXiv:2606.08941 (2026)の数学的定義を独立に実装しています。詳細は[引用と適用範囲](docs/strong_d_convex_hulls.md)を参照してください。

## ライセンス

[Apache-2.0](LICENSE-APACHE)または[MIT](LICENSE-MIT)のどちらかを選択できます。
