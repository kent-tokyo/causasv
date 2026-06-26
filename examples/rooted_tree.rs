use causasv::{AsvExplainer, Dag};

fn main() -> Result<(), causasv::CausasvError> {
    // Rooted directed tree: weather → {temperature, humidity}, temperature → comfort
    let mut dag = Dag::new();
    let weather = dag.add_node("weather");
    let temp = dag.add_node("temperature");
    let humidity = dag.add_node("humidity");
    let comfort = dag.add_node("comfort");
    dag.add_edge(weather, temp)?;
    dag.add_edge(weather, humidity)?;
    dag.add_edge(temp, comfort)?;
    dag.validate()?;

    let explainer = AsvExplainer::new(dag);

    // Nonlinear model: comfort depends on temp×humidity interaction
    let result = explainer.exact_tree(|coalition| {
        let has = |name: usize| coalition.iter().any(|n| n.0 as usize == name);
        let score = if has(2) && has(1) {
            4.0
        } else if has(1) {
            2.0
        } else {
            0.0
        };
        Ok(score + coalition.len() as f64 * 0.5)
    })?;

    println!("Exact tree ASV (n_linear_extensions={}):", result.n_samples);
    let names = ["weather", "temperature", "humidity", "comfort"];
    for (node, &value) in &result.values {
        println!("  {}: {:.4}", names[node.0 as usize], value);
    }
    Ok(())
}
