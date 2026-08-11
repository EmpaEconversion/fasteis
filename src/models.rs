//! Registry of circuits with trained initial-parameter models.

use std::collections::HashMap;
use std::mem::discriminant;
use std::sync::{Mutex, OnceLock};

use crate::circuit::{Node, Series};
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

/// `"randles"` -> `"R0-(CPE1,R1-W1)"`. `None` for anything else, which
/// `Circuit::new` then treats as a topology string.
pub fn resolve_alias(name: &str) -> Option<&'static str> {
    let trimmed = name.trim();
    all()
        .iter()
        .find(|m| m.name.eq_ignore_ascii_case(trimmed))
        .map(|m| m.circuit)
}

/// The model trained for `topology`, if there is one.
///
/// Matching is structural but order-sensitive: `R0-(CPE1,R1-W1)` matches, and so
/// does the same tree with different labels or parameter values, but a
/// topologically equivalent circuit written with its components in another order
/// does not. Labels and values are ignored; only element types and nesting matter.
pub fn find_for_topology(topology: &[Node]) -> Option<&'static Model> {
    all().iter().find(|m| {
        crate::circuit::parse(m.circuit)
            .map(|trained| same_topology(&trained, topology))
            .unwrap_or(false)
    })
}

fn same_topology(a: &[Node], b: &[Node]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| match (x, y) {
            (Node::Element(ea, _), Node::Element(eb, _)) => discriminant(ea) == discriminant(eb),
            (Node::Parallel(ba), Node::Parallel(bb)) => {
                ba.len() == bb.len()
                    && ba
                        .iter()
                        .zip(bb)
                        .all(|(s, t): (&Series, &Series)| same_topology(s, t))
            }
            _ => false,
        })
}

/// Weights loaded from a path rather than embedded, keyed by that path.
///
/// Leaked deliberately: a process compares a handful of model files at most, and a
/// `&'static` keeps the borrow simple. `topology` is checked so a file trained for a
/// different circuit cannot be used by accident.
static EXTERNAL: OnceLock<Mutex<HashMap<String, &'static nn::Guesser>>> = OnceLock::new();

pub fn load_external(path: &str, topology: &[Node]) -> Result<&'static nn::Guesser, NnError> {
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
    if !same_topology(&trained, topology) {
        return Err(NnError::BadValue(format!(
            "{path} was trained for {:?}, which is a different circuit",
            guesser.circuit()
        )));
    }
    Ok(guesser)
}

/// Message for a circuit that has no trained weights.
pub fn describe_missing() -> String {
    let available = names()
        .iter()
        .map(|n| format!("'{n}'"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "No training data on this circuit. \
         Machine-learning based initial parameters available for {available}"
    )
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
            assert!(std::ptr::eq(find_for_topology(&topology).unwrap(), model));
            assert_eq!(
                guesser.param_names(),
                crate::circuit::param_names(&topology)
            );
        }
    }

    #[test]
    fn alias_resolves_case_insensitively_and_ignores_surrounding_space() {
        assert_eq!(resolve_alias("randles"), Some("R0-(CPE1,R1-W1)"));
        assert_eq!(resolve_alias("Randles"), Some("R0-(CPE1,R1-W1)"));
        assert_eq!(resolve_alias("  randles  "), Some("R0-(CPE1,R1-W1)"));
        assert_eq!(resolve_alias("R0-C1"), None);
    }

    #[test]
    fn topology_match_ignores_labels_and_values() {
        let a = parse("R0-(CPE1,R1-W1)").unwrap();
        let b = parse("R7-(CPE2,R9-W4)").unwrap();
        assert!(same_topology(&a, &b));
    }

    #[test]
    fn topology_match_rejects_different_elements_and_ordering() {
        let trained = parse("R0-(CPE1,R1-W1)").unwrap();

        assert!(!same_topology(&trained, &parse("R0-(C1,R1-W1)").unwrap()));
        assert!(!same_topology(&trained, &parse("R0-(R1-W1,CPE1)").unwrap()));
        assert!(!same_topology(&trained, &parse("R0-(CPE1,R1)").unwrap()));
        assert!(!same_topology(&trained, &parse("R0-CPE1-R1-W1").unwrap()));
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
