use causasv::{AsvExplainer, Dag, SamplingConfig};

fn main() -> Result<(), causasv::CausasvError> {
    // Diamond DAG: a→{b,c}, b→d, c→d (not a tree)
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    let d = dag.add_node("d");
    dag.add_edge(a, b)?;
    dag.add_edge(a, c)?;
    dag.add_edge(b, d)?;
    dag.add_edge(c, d)?;
    dag.validate()?;

    let explainer = AsvExplainer::new(dag);

    // v(S) = |S|^2 (nonlinear, so ASV ≠ equal split)
    let result = explainer.approximate(
        |coalition| Ok((coalition.len() as f64).powi(2)),
        SamplingConfig::new(20_000).with_seed(0),
    )?;

    // Compare with brute-force
    let dag2 = {
        let mut d = Dag::new();
        let a2 = d.add_node("a");
        let b2 = d.add_node("b");
        let c2 = d.add_node("c");
        let d2 = d.add_node("d");
        d.add_edge(a2, b2).unwrap();
        d.add_edge(a2, c2).unwrap();
        d.add_edge(b2, d2).unwrap();
        d.add_edge(c2, d2).unwrap();
        d
    };
    let exact = AsvExplainer::new(dag2).exact(|s| Ok((s.len() as f64).powi(2)))?;

    println!("Diamond DAG, v(S) = |S|²");
    println!("{:<8} {:>10} {:>10}", "node", "approx", "exact");
    let names = ["a", "b", "c", "d"];
    for node in [a, b, c, d] {
        println!(
            "{:<8} {:>10.4} {:>10.4}",
            names[node.0 as usize], result.values[&node], exact.values[&node]
        );
    }
    Ok(())
}
