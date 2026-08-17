//! Registry of circuits with trained initial-parameter models.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::circuit::Node;
use crate::nn::{self, NnError};

pub struct Model {
    /// Short alias accepted by `Circuit::new`, e.g. "randles".
    pub name: &'static str,
    /// The topology the weights were trained for.
    pub circuit: &'static str,
    bytes: &'static [u8],
    guesser: OnceLock<Result<nn::Guesser, String>>,
}

impl Model {
    /// Parse the embedded weights, once per process.
    pub fn guesser(&self) -> Result<&nn::Guesser, NnError> {
        match self
            .guesser
            .get_or_init(|| self.parse_and_check().map_err(|e| e.to_string()))
        {
            Ok(g) => Ok(g),
            Err(message) => Err(NnError::BadValue(message.clone())),
        }
    }

    /// Catches a `.eisnn` that was exported for a different circuit than the row it
    /// is registered under, which would otherwise return plausible nonsense.
    fn parse_and_check(&self) -> Result<nn::Guesser, NnError> {
        let guesser = nn::Guesser::from_bytes(self.bytes)?;
        if guesser.circuit() != self.circuit {
            return Err(NnError::BadValue(format!(
                "model {:?} is registered as {:?} but its weights were trained for {:?}",
                self.name,
                self.circuit,
                guesser.circuit()
            )));
        }

        let expected = crate::circuit::parse(self.circuit)
            .map(|topology| crate::circuit::param_names(&topology))
            .map_err(|_| NnError::BadValue(format!("registry circuit {:?}", self.circuit)))?;
        if guesser.param_names() != expected {
            return Err(NnError::BadValue(format!(
                "model {:?} has parameters {:?}, expected {expected:?}",
                self.name,
                guesser.param_names()
            )));
        }
        Ok(guesser)
    }
}

/// Ordered simplest first; this is the order `ml_circuits()` reports.
static MODELS: [Model; 11] = [
    Model {
        name: "rc",
        circuit: "R0-(R1,C1)",
        bytes: include_bytes!("models/rc.eisnn"),
        guesser: OnceLock::new(),
    },
    Model {
        name: "rc_l",
        circuit: "L0-R0-(R1,C1)",
        bytes: include_bytes!("models/rc_l.eisnn"),
        guesser: OnceLock::new(),
    },
    Model {
        name: "rq",
        circuit: "R0-(R1,CPE1)",
        bytes: include_bytes!("models/rq.eisnn"),
        guesser: OnceLock::new(),
    },
    Model {
        name: "rq_l",
        circuit: "L0-R0-(R1,CPE1)",
        bytes: include_bytes!("models/rq_l.eisnn"),
        guesser: OnceLock::new(),
    },
    Model {
        name: "two_rc",
        circuit: "R0-(R1,C1)-(R2,C2)",
        bytes: include_bytes!("models/two_rc.eisnn"),
        guesser: OnceLock::new(),
    },
    Model {
        name: "two_rc_l",
        circuit: "L0-R0-(R1,C1)-(R2,C2)",
        bytes: include_bytes!("models/two_rc_l.eisnn"),
        guesser: OnceLock::new(),
    },
    Model {
        name: "two_rq",
        circuit: "R0-(R1,CPE1)-(R2,CPE2)",
        bytes: include_bytes!("models/two_rq.eisnn"),
        guesser: OnceLock::new(),
    },
    Model {
        name: "two_rq_l",
        circuit: "L0-R0-(R1,CPE1)-(R2,CPE2)",
        bytes: include_bytes!("models/two_rq_l.eisnn"),
        guesser: OnceLock::new(),
    },
    Model {
        name: "randles",
        circuit: "R0-(R1-W1,CPE1)",
        bytes: include_bytes!("models/randles.eisnn"),
        guesser: OnceLock::new(),
    },
    Model {
        name: "sei_randles",
        circuit: "R0-(R1,CPE1)-(R2-W2,CPE2)",
        bytes: include_bytes!("models/sei_randles.eisnn"),
        guesser: OnceLock::new(),
    },
    Model {
        name: "sei_randles_wo",
        circuit: "R0-(R1,CPE1)-(R2-Wo2,CPE2)",
        bytes: include_bytes!("models/sei_randles_wo.eisnn"),
        guesser: OnceLock::new(),
    },
];

/// Every circuit with trained weights, in registry order.
pub fn all() -> &'static [Model] {
    &MODELS
}

/// Aliases accepted by `Circuit::new`, for error messages.
pub fn names() -> Vec<&'static str> {
    all().iter().map(|m| m.name).collect()
}

/// `"randles"` -> `"R0-(R1-W1,CPE1)"`. `None` for anything else, which
/// `Circuit::new` then treats as a topology string.
pub fn resolve_alias(name: &str) -> Option<&'static str> {
    let trimmed = name.trim();
    all()
        .iter()
        .find(|m| m.name.eq_ignore_ascii_case(trimmed))
        .map(|m| m.circuit)
}

/// One node of a parsed tree, annotated with where its parameters sit in that
/// tree's `param_names()` order and with a key describing its shape alone.
struct Shape {
    key: String,
    kind: Kind,
}

enum Kind {
    /// `start..start + len` in `param_names()` order.
    Leaf {
        start: usize,
        len: usize,
    },
    Parallel(Vec<Vec<Shape>>),
}

/// Shapes of `series`' nodes, sorted into canonical order.
///
/// `cursor` walks the tree in the unsorted traversal order `param_names()` uses,
/// so each leaf keeps its original parameter index across the sort.
fn shapes(series: &[Node], cursor: &mut usize) -> Vec<Shape> {
    let mut out: Vec<Shape> = series.iter().map(|node| shape(node, cursor)).collect();
    out.sort_by(|a, b| a.key.cmp(&b.key));
    out
}

fn shape(node: &Node, cursor: &mut usize) -> Shape {
    match node {
        Node::Element(element, _) => {
            let len = element.param_names().len();
            let start = *cursor;
            *cursor += len;
            Shape {
                key: element.type_tag().to_string(),
                kind: Kind::Leaf { start, len },
            }
        }
        Node::Parallel(branches) => {
            let mut branches: Vec<Vec<Shape>> =
                branches.iter().map(|b| shapes(b, cursor)).collect();
            branches.sort_by_cached_key(|b| series_key(b));
            let joined: Vec<String> = branches.iter().map(|b| series_key(b)).collect();
            Shape {
                key: format!("({})", joined.join(",")),
                kind: Kind::Parallel(branches),
            }
        }
    }
}

fn series_key(series: &[Shape]) -> String {
    let joined: Vec<&str> = series.iter().map(|s| s.key.as_str()).collect();
    format!("[{}]", joined.join(","))
}

/// Parameter spans of every leaf, in canonical order.
fn spans(series: &[Shape], out: &mut Vec<(usize, usize)>) {
    for node in series {
        match &node.kind {
            Kind::Leaf { start, len } => out.push((*start, *len)),
            Kind::Parallel(branches) => branches.iter().for_each(|b| spans(b, out)),
        }
    }
}

fn canonical(series: &[Node]) -> (String, Vec<(usize, usize)>) {
    let mut cursor = 0;
    let sorted = shapes(series, &mut cursor);
    let key = series_key(&sorted);
    let mut leaves = Vec::new();
    spans(&sorted, &mut leaves);
    (key, leaves)
}

/// Match the 'trained' and given circuits, so model predictions get mapped
/// back to the ordering and labels supplied by the user.
/// Order does not matter in the circuits: `(R2,C2)-R1` matches `R1-(R2,C2)`.
/// Labels and values do not matter, only the element types and their nesting.
/// For indistinguishable elements, e.g. the two arcs of `R0-(R1,C1)-(R2,C2)`,
/// they pair up in written order.
/// Returns None if they do not match.
pub fn match_topology(trained: &[Node], topology: &[Node]) -> Option<Vec<usize>> {
    let (trained_key, trained_leaves) = canonical(trained);
    let (topology_key, topology_leaves) = canonical(topology);
    if trained_key != topology_key {
        return None;
    }

    let width = topology_leaves.iter().map(|(_, len)| len).sum();
    let mut permutation = vec![0usize; width];
    for (&(from, len), &(to, _)) in trained_leaves.iter().zip(&topology_leaves) {
        for offset in 0..len {
            permutation[to + offset] = from + offset;
        }
    }
    Some(permutation)
}

/// Reorder `values`, given in the trained circuit's order, into the order of the
/// circuit `permutation` was matched against.
pub fn apply_permutation(permutation: &[usize], values: &[f64]) -> Vec<f64> {
    permutation.iter().map(|&i| values[i]).collect()
}

/// A trained model paired with a circuit, and how to get its guess into that
/// circuit's parameter order.
pub struct Match<T> {
    pub model: T,
    /// Trained parameter order -> matched topology's order.
    pub permutation: Vec<usize>,
    /// Whether matching needed `K`/`Zarc` expanded. Guessed values are then in
    /// the expanded circuit's parameters and must be contracted back.
    pub expanded: bool,
}

/// `match_topology`, retried with `K`/`Zarc` expanded to the parallel pairs they
/// abbreviate. E.g. `R-K` expands to `R-(R,C)`, matching a trained circuit.
fn match_or_expand(trained: &[Node], topology: &[Node]) -> Option<(Vec<usize>, bool)> {
    if let Some(permutation) = match_topology(trained, topology) {
        return Some((permutation, false));
    }
    let expanded = crate::circuit::expand_arcs(topology)?;
    Some((match_topology(trained, &expanded)?, true))
}

/// The model trained for `topology`, with the permutation taking its parameter
/// order to `topology`'s. See `match_topology` for what counts as a match.
pub fn find_for_topology(topology: &[Node]) -> Option<Match<&'static Model>> {
    all().iter().find_map(|m| {
        let trained = crate::circuit::parse(m.circuit).ok()?;
        let (permutation, expanded) = match_or_expand(&trained, topology)?;
        Some(Match {
            model: m,
            permutation,
            expanded,
        })
    })
}

/// Weights loaded from a path rather than embedded, keyed by that path.
///
/// Leaked deliberately: a process compares a handful of model files at most, and a
/// `&'static` keeps the borrow simple. `topology` is checked so a file trained for a
/// different circuit cannot be used by accident.
static EXTERNAL: OnceLock<Mutex<HashMap<String, &'static nn::Guesser>>> = OnceLock::new();

pub fn load_external(
    path: &str,
    topology: &[Node],
) -> Result<Match<&'static nn::Guesser>, NnError> {
    let cache = EXTERNAL.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().expect("weights cache poisoned");

    let guesser: &'static nn::Guesser = match map.get(path) {
        Some(g) => g,
        None => {
            let loaded: &'static nn::Guesser = Box::leak(Box::new(nn::Guesser::load(path)?));
            map.insert(path.to_string(), loaded);
            loaded
        }
    };

    let trained = crate::circuit::parse(guesser.circuit())
        .map_err(|_| NnError::BadValue(format!("circuit {:?} in {path}", guesser.circuit())))?;
    let (permutation, expanded) = match_or_expand(&trained, topology).ok_or_else(|| {
        NnError::BadValue(format!(
            "{path} was trained for {:?}, which is a different circuit",
            guesser.circuit()
        ))
    })?;
    Ok(Match {
        model: guesser,
        permutation,
        expanded,
    })
}

/// Message for a circuit that has no trained weights.
pub fn describe_missing() -> String {
    format!(
        "No training data on this circuit. \
         Machine-learning based initial parameters available for {}",
        quoted_names()
    )
}

/// Warning for `fit()` when it would have guessed but has no model to guess with.
pub fn describe_fallback() -> String {
    format!(
        "There is no ML model to guess initial parameters for this circuit. \
         Supply your own with Circuit.with_named_values({{...}}) or Circuit.with_values([...]), \
         or pass guess_init=False to silence this. Models available for {}",
        quoted_names()
    )
}

fn quoted_names() -> String {
    names()
        .iter()
        .map(|n| format!("'{n}'"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::parse;

    #[test]
    fn every_registered_model_loads_and_matches_its_own_circuit() {
        for model in all() {
            let guesser = model.guesser().expect("embedded weights must parse");
            assert_eq!(guesser.circuit(), model.circuit);

            let topology = parse(model.circuit).expect("registry circuit must parse");
            let found = find_for_topology(&topology).unwrap();
            assert!(std::ptr::eq(found.model, model));
            assert!(!found.expanded);
            let permutation = found.permutation;
            assert_eq!(permutation, (0..permutation.len()).collect::<Vec<_>>());
            assert_eq!(
                guesser.param_names(),
                crate::circuit::param_names(&topology)
            );
        }
    }

    #[test]
    fn arc_circuits_reach_the_models_trained_on_written_out_arcs() {
        for (circuit, expected) in [
            ("R0-K1", "rc"),
            ("R0-Zarc1", "rq"),
            ("L0-R0-Zarc1", "rq_l"),
            ("R0-Zarc1-Zarc2", "two_rq"),
            ("R0-Zarc1-(R2,Cpe2)", "two_rq"),
        ] {
            let found = find_for_topology(&parse(circuit).unwrap())
                .unwrap_or_else(|| panic!("{circuit} should reach a model"));
            assert_eq!(found.model.name, expected, "{circuit}");
            assert!(found.expanded, "{circuit}");
        }
    }

    #[test]
    fn alias_resolves_case_insensitively_and_ignores_surrounding_space() {
        assert_eq!(resolve_alias("randles"), Some("R0-(R1-W1,CPE1)"));
        assert_eq!(resolve_alias("Randles"), Some("R0-(R1-W1,CPE1)"));
        assert_eq!(resolve_alias("  randles  "), Some("R0-(R1-W1,CPE1)"));
        assert_eq!(resolve_alias("R0-C1"), None);
    }

    #[test]
    fn topology_match_ignores_labels_and_values() {
        let a = parse("R0-(CPE1,R1-W1)").unwrap();
        let b = parse("R7-(CPE2,R9-W4)").unwrap();
        assert_eq!(match_topology(&a, &b), Some(vec![0, 1, 2, 3, 4]));
    }

    #[test]
    fn topology_match_rejects_different_elements_and_nesting() {
        let trained = parse("R0-(CPE1,R1-W1)").unwrap();

        assert!(match_topology(&trained, &parse("R0-(C1,R1-W1)").unwrap()).is_none());
        assert!(match_topology(&trained, &parse("R0-(CPE1,R1)").unwrap()).is_none());
        assert!(match_topology(&trained, &parse("R0-CPE1-R1-W1").unwrap()).is_none());
    }

    #[test]
    fn topology_match_ignores_order_of_series_and_parallel_siblings() {
        let trained = parse("R0-(R1,C1)").unwrap();
        // param_names: R0.r, R1.r, C1.c
        for (reordered, expected) in [
            ("R0-(C1,R1)", vec![0, 2, 1]),
            ("(R1,C1)-R0", vec![1, 2, 0]),
            ("(C1,R1)-R0", vec![2, 1, 0]),
        ] {
            let topology = parse(reordered).unwrap();
            assert_eq!(
                match_topology(&trained, &topology),
                Some(expected),
                "{reordered}"
            );
        }
    }

    #[test]
    fn permutation_moves_each_value_to_the_parameter_of_the_same_type() {
        let trained = parse("L0-R0-(R1,CPE1)").unwrap();
        let topology = parse("(CPE9,R9)-R8-L8").unwrap();
        let permutation = match_topology(&trained, &topology).unwrap();

        // one distinguishable value per parameter of the trained circuit
        let values: Vec<f64> = (0..crate::circuit::param_count(&trained))
            .map(|i| i as f64)
            .collect();
        let moved = apply_permutation(&permutation, &values);

        let by_name: Vec<(String, f64)> = crate::circuit::param_names(&topology)
            .into_iter()
            .zip(moved)
            .collect();
        let trained_names = crate::circuit::param_names(&trained);
        for (name, value) in by_name {
            let suffix = name.split_once('.').unwrap().1;
            assert!(trained_names[value as usize].ends_with(suffix), "{name}");
        }
    }

    #[test]
    fn identical_siblings_pair_up_in_written_order() {
        let trained = parse("R0-(R1,C1)-(R2,C2)").unwrap();
        let topology = parse("R0-(R1,C1)-(R2,C2)").unwrap();
        let permutation = match_topology(&trained, &topology).unwrap();
        assert_eq!(permutation, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn missing_message_lists_available_names() {
        let message = describe_missing();
        assert!(message.starts_with("No training data on this circuit."));
        for name in names() {
            assert!(message.contains(&format!("'{name}'")));
        }
    }
}
