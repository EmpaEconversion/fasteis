use std::collections::{HashMap, HashSet};

use num_complex::Complex64;

use crate::elements::Element;

/// A leaf carries an optional user-chosen label (e.g. "Cpe1") when it came from
/// a parsed circuit string; `None` for circuits built via the programmatic
/// `Circuit` builder, which keeps today's auto-numbered parameter names.
#[derive(Debug, Clone)]
pub enum Node {
    Element(Element, Option<String>),
    Parallel(Vec<Series>),
}

/// A bare `Vec<Node>` means "these, in series" -- used both as the top-level
/// circuit and as each branch of a `Node::Parallel`.
pub type Series = Vec<Node>;

pub fn impedance(series: &[Node], omega: f64) -> Complex64 {
    series.iter().map(|n| node_impedance(n, omega)).sum()
}

fn node_impedance(node: &Node, omega: f64) -> Complex64 {
    match node {
        Node::Element(e, _) => e.impedance(omega),
        Node::Parallel(branches) => {
            let sum_inv: Complex64 =
                branches.iter().map(|branch| Complex64::new(1.0, 0.0) / impedance(branch, omega)).sum();
            Complex64::new(1.0, 0.0) / sum_inv
        }
    }
}

/// Leaves in depth-first, pre-order traversal -- the same order `impedance()` walks the tree.
fn leaves(series: &[Node]) -> Vec<(&Element, &Option<String>)> {
    let mut out = Vec::new();
    collect_leaves(series, &mut out);
    out
}

fn collect_leaves<'a>(series: &'a [Node], out: &mut Vec<(&'a Element, &'a Option<String>)>) {
    for node in series {
        match node {
            Node::Element(e, label) => out.push((e, label)),
            Node::Parallel(branches) => {
                for branch in branches {
                    collect_leaves(branch, out);
                }
            }
        }
    }
}

/// Parameter names, in the same order as `param_values()`. Labeled leaves (parsed
/// from a circuit string) use their label verbatim, e.g. "Cpe1.q"; unlabeled
/// leaves keep the auto-generated, stable, collision-free scheme, e.g. "R0.r" --
/// numbered per element-type in traversal order, counting only unlabeled leaves
/// of that type.
pub fn param_names(series: &[Node]) -> Vec<String> {
    let mut counters: HashMap<&'static str, usize> = HashMap::new();
    leaves(series)
        .into_iter()
        .flat_map(|(e, label)| {
            let tag = e.type_tag();
            let owned_label;
            let label: &str = match label {
                Some(l) => l,
                None => {
                    let idx = counters.entry(tag).or_insert(0);
                    owned_label = format!("{tag}{idx}");
                    *idx += 1;
                    &owned_label
                }
            };
            e.param_names().iter().map(move |n| format!("{label}.{n}")).collect::<Vec<_>>()
        })
        .collect()
}

/// Current parameter values, in the same order as `param_names()`.
pub fn param_values(series: &[Node]) -> Vec<f64> {
    leaves(series).into_iter().flat_map(|(e, _)| e.values()).collect()
}

/// Default physical-validity bounds, in the same order as `param_names()`.
pub fn param_bounds(series: &[Node]) -> Vec<(f64, f64)> {
    leaves(series).into_iter().flat_map(|(e, _)| e.param_bounds()).collect()
}

/// Total number of free parameters across all leaves.
pub fn param_count(series: &[Node]) -> usize {
    leaves(series).iter().map(|(e, _)| e.param_names().len()).sum()
}

/// Rebuild the series with a new flat parameter vector, consumed in the same
/// traversal order `param_names()`/`param_values()` produced.
pub fn with_param_values(series: &[Node], values: &[f64]) -> Series {
    let mut iter = values.iter().copied();
    let result = rebuild(series, &mut iter);
    debug_assert!(iter.next().is_none(), "with_param_values: too many values supplied");
    result
}

fn rebuild(series: &[Node], iter: &mut impl Iterator<Item = f64>) -> Series {
    series
        .iter()
        .map(|node| match node {
            Node::Element(e, label) => {
                let n = e.param_names().len();
                let vals: Vec<f64> = iter.by_ref().take(n).collect();
                debug_assert_eq!(vals.len(), n, "with_param_values: not enough values supplied");
                Node::Element(e.with_values(&vals), label.clone())
            }
            Node::Parallel(branches) => {
                Node::Parallel(branches.iter().map(|b| rebuild(b, iter)).collect())
            }
        })
        .collect()
}

/// Errors produced while parsing a circuit string such as `"R0-p(R1,Cpe1)"`.
#[derive(Debug, PartialEq)]
pub enum ParseError {
    UnexpectedChar(usize),
    UnexpectedEnd,
    UnknownElementCode(String, usize),
    TrailingInput(usize),
    DuplicateLabel(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedChar(pos) => write!(f, "unexpected character at position {pos}"),
            ParseError::UnexpectedEnd => write!(f, "unexpected end of input"),
            ParseError::UnknownElementCode(code, pos) => {
                write!(f, "unknown element code {code:?} at position {pos}")
            }
            ParseError::TrailingInput(pos) => write!(f, "unexpected trailing input at position {pos}"),
            ParseError::DuplicateLabel(label) => write!(f, "duplicate element label {label:?}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse a circuit topology string, e.g. `"R0-p(R1,Cpe1)"` or `"R0-p(R1-C1,R2-Cpe2)"`.
///
/// Grammar:
/// ```text
/// series   := term ('-' term)*
/// term     := parallel | element
/// parallel := 'p' '(' series (',' series)* ')'
/// element  := code digits
/// ```
/// `code` is one of the known element codes (matched against the longest run of
/// letters, so e.g. "Tlmq5" and "T5" are unambiguous). The string carries no
/// parameter values -- every parsed element gets a sensible placeholder default
/// (see `Element::default_for_code`); real values are supplied afterward via
/// `with_param_values` or a name-keyed lookup against `param_names()`.
pub fn parse(input: &str) -> Result<Series, ParseError> {
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0;
    let series = parse_series(&chars, &mut pos)?;
    if pos != chars.len() {
        return Err(ParseError::TrailingInput(pos));
    }

    let mut seen = HashSet::new();
    for (_, label) in leaves(&series) {
        if let Some(l) = label {
            if !seen.insert(l.clone()) {
                return Err(ParseError::DuplicateLabel(l.clone()));
            }
        }
    }

    Ok(series)
}

fn parse_series(chars: &[char], pos: &mut usize) -> Result<Series, ParseError> {
    let mut nodes = vec![parse_term(chars, pos)?];
    while *pos < chars.len() && chars[*pos] == '-' {
        *pos += 1;
        nodes.push(parse_term(chars, pos)?);
    }
    Ok(nodes)
}

fn parse_term(chars: &[char], pos: &mut usize) -> Result<Node, ParseError> {
    if *pos < chars.len() && chars[*pos] == 'p' && chars.get(*pos + 1) == Some(&'(') {
        *pos += 2;
        let mut branches = vec![parse_series(chars, pos)?];
        while *pos < chars.len() && chars[*pos] == ',' {
            *pos += 1;
            branches.push(parse_series(chars, pos)?);
        }
        if *pos >= chars.len() || chars[*pos] != ')' {
            return Err(ParseError::UnexpectedEnd);
        }
        *pos += 1;
        Ok(Node::Parallel(branches))
    } else {
        parse_element(chars, pos)
    }
}

fn parse_element(chars: &[char], pos: &mut usize) -> Result<Node, ParseError> {
    let start = *pos;
    while *pos < chars.len() && chars[*pos].is_ascii_alphabetic() {
        *pos += 1;
    }
    if *pos == start {
        return if *pos >= chars.len() {
            Err(ParseError::UnexpectedEnd)
        } else {
            Err(ParseError::UnexpectedChar(*pos))
        };
    }
    let code: String = chars[start..*pos].iter().collect();

    let digits_start = *pos;
    while *pos < chars.len() && chars[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if *pos == digits_start {
        return Err(ParseError::UnexpectedChar(*pos));
    }
    let digits: String = chars[digits_start..*pos].iter().collect();
    let label = format!("{code}{digits}");

    let element = Element::default_for_code(&code).ok_or(ParseError::UnknownElementCode(code, start))?;
    Ok(Node::Element(element, Some(label)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: Complex64, b: Complex64, tol: f64) {
        assert!((a - b).norm() < tol, "{:?} != {:?}", a, b);
    }

    fn r(value: f64) -> Node {
        Node::Element(Element::R { r: value }, None)
    }

    fn cpe(q: f64, alpha: f64) -> Node {
        Node::Element(Element::Cpe { q, alpha }, None)
    }

    #[test]
    fn series_sums_impedance() {
        let circuit = vec![r(3.0), r(4.0)];
        assert_close(impedance(&circuit, 1.0), Complex64::new(7.0, 0.0), 1e-12);
    }

    #[test]
    fn parallel_combines_impedance() {
        let circuit = vec![Node::Parallel(vec![vec![r(10.0)], vec![r(10.0)]])];
        assert_close(impedance(&circuit, 1.0), Complex64::new(5.0, 0.0), 1e-12);
    }

    #[test]
    fn nested_series_of_parallel() {
        let circuit = vec![Node::Parallel(vec![vec![r(10.0)], vec![r(10.0)]]), r(5.0)];
        assert_close(impedance(&circuit, 1.0), Complex64::new(10.0, 0.0), 1e-12);
    }

    #[test]
    fn param_names_are_stable_and_type_scoped() {
        let circuit = vec![r(1.0), Node::Parallel(vec![vec![r(2.0)], vec![cpe(3.0, 0.5)]]), cpe(4.0, 0.9)];
        assert_eq!(param_names(&circuit), vec!["R0.r", "R1.r", "Cpe0.q", "Cpe0.alpha", "Cpe1.q", "Cpe1.alpha"]);
    }

    #[test]
    fn with_param_values_roundtrips_through_param_values() {
        let circuit = vec![r(1.0), Node::Parallel(vec![vec![r(2.0)], vec![cpe(3.0, 0.5)]]), cpe(4.0, 0.9)];
        let values = param_values(&circuit);
        let rebuilt = with_param_values(&circuit, &values);
        for &omega in &[0.1, 1.0, 100.0] {
            assert_close(impedance(&circuit, omega), impedance(&rebuilt, omega), 1e-12);
        }

        let mut doubled = values.clone();
        for v in doubled.iter_mut() {
            *v *= 2.0;
        }
        let scaled = with_param_values(&circuit, &doubled);
        assert_eq!(param_values(&scaled), doubled);
    }

    #[test]
    fn param_count_matches_flattened_length() {
        let circuit = vec![r(1.0), Node::Parallel(vec![vec![r(2.0)], vec![cpe(3.0, 0.5)]]), cpe(4.0, 0.9)];
        assert_eq!(param_count(&circuit), param_values(&circuit).len());
        assert_eq!(param_count(&circuit), param_names(&circuit).len());
        assert_eq!(param_count(&circuit), 6);
    }

    #[test]
    fn parses_flat_series() {
        let circuit = parse("R0-C1").unwrap();
        assert_eq!(param_names(&circuit), vec!["R0.r", "C1.c"]);
    }

    #[test]
    fn parses_series_ending_in_parallel() {
        let circuit = parse("R0-p(R1,Cpe1)").unwrap();
        assert_eq!(param_names(&circuit), vec!["R0.r", "R1.r", "Cpe1.q", "Cpe1.alpha"]);
    }

    #[test]
    fn parses_series_inside_parallel_branches() {
        let circuit = parse("R0-p(R1-C1,R2-Cpe2)").unwrap();
        assert_eq!(
            param_names(&circuit),
            vec!["R0.r", "R1.r", "C1.c", "R2.r", "Cpe2.q", "Cpe2.alpha"]
        );
    }

    #[test]
    fn parses_nested_parallel() {
        let circuit = parse("p(R0,p(R1,C1))").unwrap();
        assert_eq!(param_names(&circuit), vec!["R0.r", "R1.r", "C1.c"]);
    }

    #[test]
    fn rejects_unknown_element_code() {
        assert!(matches!(parse("Q0"), Err(ParseError::UnknownElementCode(code, _)) if code == "Q"));
    }

    #[test]
    fn rejects_missing_label_digits() {
        assert!(matches!(parse("R-C1"), Err(ParseError::UnexpectedChar(_))));
    }

    #[test]
    fn rejects_unbalanced_parens() {
        assert!(matches!(parse("R0-p(R1,C1"), Err(ParseError::UnexpectedEnd)));
    }

    #[test]
    fn rejects_trailing_input() {
        assert!(matches!(parse("R0)"), Err(ParseError::TrailingInput(_))));
    }

    #[test]
    fn rejects_duplicate_labels() {
        assert!(matches!(parse("R0-R0"), Err(ParseError::DuplicateLabel(l)) if l == "R0"));
    }

    #[test]
    fn accepts_alias_codes_matching_python_static_method_casing() {
        let circuit = parse("CPE0-TLMQ1").unwrap();
        assert_eq!(param_names(&circuit), vec!["CPE0.q", "CPE0.alpha", "TLMQ1.r_ion", "TLMQ1.qs", "TLMQ1.gamma"]);
    }
}
