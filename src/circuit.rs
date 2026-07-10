use num_complex::Complex64;

use crate::elements::Element;

#[derive(Debug, Clone)]
pub enum Node {
    Leaf(Element),
    Series(Vec<Node>),
    Parallel(Vec<Node>),
}

impl Node {
    pub fn impedance(&self, omega: f64) -> Complex64 {
        match self {
            Node::Leaf(e) => e.impedance(omega),
            Node::Series(nodes) => nodes.iter().map(|n| n.impedance(omega)).sum(),
            Node::Parallel(nodes) => {
                let sum_inv: Complex64 = nodes
                    .iter()
                    .map(|n| Complex64::new(1.0, 0.0) / n.impedance(omega))
                    .sum();
                Complex64::new(1.0, 0.0) / sum_inv
            }
        }
    }

    /// Leaf elements in depth-first, pre-order traversal -- the same order `impedance()` walks the tree.
    fn leaves(&self) -> Vec<&Element> {
        let mut out = Vec::new();
        self.collect_leaves(&mut out);
        out
    }

    fn collect_leaves<'a>(&'a self, out: &mut Vec<&'a Element>) {
        match self {
            Node::Leaf(e) => out.push(e),
            Node::Series(nodes) | Node::Parallel(nodes) => {
                for n in nodes {
                    n.collect_leaves(out);
                }
            }
        }
    }

    /// Auto-generated, stable, collision-free parameter names, e.g. "R0.r", "Cpe1.alpha".
    /// Leaves are numbered per element-type in traversal order, regardless of Series/Parallel nesting.
    pub fn param_names(&self) -> Vec<String> {
        let mut counters: std::collections::HashMap<&'static str, usize> = std::collections::HashMap::new();
        self.leaves()
            .into_iter()
            .flat_map(|e| {
                let tag = e.type_tag();
                let idx = counters.entry(tag).or_insert(0);
                let label = format!("{tag}{idx}");
                *idx += 1;
                e.param_names().iter().map(move |n| format!("{label}.{n}")).collect::<Vec<_>>()
            })
            .collect()
    }

    /// Current parameter values, in the same order as `param_names()`.
    pub fn param_values(&self) -> Vec<f64> {
        self.leaves().into_iter().flat_map(|e| e.values()).collect()
    }

    /// Default physical-validity bounds, in the same order as `param_names()`.
    pub fn param_bounds(&self) -> Vec<(f64, f64)> {
        self.leaves().into_iter().flat_map(|e| e.param_bounds()).collect()
    }

    /// Total number of free parameters across all leaves.
    pub fn param_count(&self) -> usize {
        self.leaves().iter().map(|e| e.param_names().len()).sum()
    }

    /// Rebuild the tree with a new flat parameter vector, consumed in the same
    /// traversal order `param_names()`/`param_values()` produced.
    pub fn with_param_values(&self, values: &[f64]) -> Node {
        let mut iter = values.iter().copied();
        let result = self.rebuild(&mut iter);
        debug_assert!(iter.next().is_none(), "with_param_values: too many values supplied");
        result
    }

    fn rebuild(&self, iter: &mut impl Iterator<Item = f64>) -> Node {
        match self {
            Node::Leaf(e) => {
                let n = e.param_names().len();
                let vals: Vec<f64> = iter.by_ref().take(n).collect();
                debug_assert_eq!(vals.len(), n, "with_param_values: not enough values supplied");
                Node::Leaf(e.with_values(&vals))
            }
            Node::Series(nodes) => Node::Series(nodes.iter().map(|n| n.rebuild(iter)).collect()),
            Node::Parallel(nodes) => Node::Parallel(nodes.iter().map(|n| n.rebuild(iter)).collect()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: Complex64, b: Complex64, tol: f64) {
        assert!((a - b).norm() < tol, "{:?} != {:?}", a, b);
    }

    fn r(value: f64) -> Node {
        Node::Leaf(Element::R { r: value })
    }

    #[test]
    fn series_sums_impedance() {
        let circuit = Node::Series(vec![r(3.0), r(4.0)]);
        assert_close(circuit.impedance(1.0), Complex64::new(7.0, 0.0), 1e-12);
    }

    #[test]
    fn parallel_combines_impedance() {
        let circuit = Node::Parallel(vec![r(10.0), r(10.0)]);
        assert_close(circuit.impedance(1.0), Complex64::new(5.0, 0.0), 1e-12);
    }

    #[test]
    fn nested_series_of_parallel() {
        let circuit = Node::Series(vec![Node::Parallel(vec![r(10.0), r(10.0)]), r(5.0)]);
        assert_close(circuit.impedance(1.0), Complex64::new(10.0, 0.0), 1e-12);
    }

    fn cpe(q: f64, alpha: f64) -> Node {
        Node::Leaf(Element::Cpe { q, alpha })
    }

    #[test]
    fn param_names_are_stable_and_type_scoped() {
        let circuit = Node::Series(vec![r(1.0), Node::Parallel(vec![r(2.0), cpe(3.0, 0.5)]), cpe(4.0, 0.9)]);
        assert_eq!(
            circuit.param_names(),
            vec!["R0.r", "R1.r", "Cpe0.q", "Cpe0.alpha", "Cpe1.q", "Cpe1.alpha"]
        );
    }

    #[test]
    fn with_param_values_roundtrips_through_param_values() {
        let circuit = Node::Series(vec![r(1.0), Node::Parallel(vec![r(2.0), cpe(3.0, 0.5)]), cpe(4.0, 0.9)]);
        let values = circuit.param_values();
        let rebuilt = circuit.with_param_values(&values);
        for &omega in &[0.1, 1.0, 100.0] {
            assert_close(circuit.impedance(omega), rebuilt.impedance(omega), 1e-12);
        }

        let mut doubled = values.clone();
        for v in doubled.iter_mut() {
            *v *= 2.0;
        }
        let scaled = circuit.with_param_values(&doubled);
        assert_eq!(scaled.param_values(), doubled);
    }

    #[test]
    fn param_count_matches_flattened_length() {
        let circuit = Node::Series(vec![r(1.0), Node::Parallel(vec![r(2.0), cpe(3.0, 0.5)]), cpe(4.0, 0.9)]);
        assert_eq!(circuit.param_count(), circuit.param_values().len());
        assert_eq!(circuit.param_count(), circuit.param_names().len());
        assert_eq!(circuit.param_count(), 6);
    }
}
