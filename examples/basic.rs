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

    // Additive toy model: score = number of features present
    let result = explainer.approximate(
        |coalition| Ok(coalition.len() as f64),
        SamplingConfig::new(10_000).with_seed(42),
    )?;

    println!(
        "Approximate ASV (n_samples={}, seed={:?}):",
        result.n_samples, result.seed
    );
    let names = ["education", "income", "risk_score"];
    for (node, &value) in &result.values {
        println!("  {}: {:.4}", names[node.0 as usize], value);
    }
    Ok(())
}
