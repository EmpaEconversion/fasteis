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
}
