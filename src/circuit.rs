// Copyright © 2026, Empa.
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
            let sum_inv: Complex64 = branches
                .iter()
                .map(|branch| Complex64::new(1.0, 0.0) / impedance(branch, omega))
                .sum();
            Complex64::new(1.0, 0.0) / sum_inv
        }
    }
}

/// Impedance of `series` evaluated with a parameter vector `params`
/// (`param_names()`/`param_values()` order), without rebuilding a the whole
/// `Series`/`Node` tree. Speeds up fit loops.
pub fn impedance_with_params(series: &[Node], params: &[f64], omega: f64) -> Complex64 {
    let mut iter = params.iter().copied();
    let z = series_impedance_with_params(series, &mut iter, omega);
    debug_assert!(
        iter.next().is_none(),
        "impedance_with_params: too many values supplied"
    );
    z
}

fn series_impedance_with_params(
    series: &[Node],
    iter: &mut impl Iterator<Item = f64>,
    omega: f64,
) -> Complex64 {
    series
        .iter()
        .map(|node| node_impedance_with_params(node, iter, omega))
        .sum()
}

fn node_impedance_with_params(
    node: &Node,
    iter: &mut impl Iterator<Item = f64>,
    omega: f64,
) -> Complex64 {
    match node {
        Node::Element(e, _) => e.impedance_from_iter(iter, omega),
        Node::Parallel(branches) => {
            let sum_inv: Complex64 = branches
                .iter()
                .map(|branch| {
                    Complex64::new(1.0, 0.0) / series_impedance_with_params(branch, iter, omega)
                })
                .sum();
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
            e.param_names()
                .iter()
                .map(move |n| format!("{label}.{n}"))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Current parameter values, in the same order as `param_names()`.
pub fn param_values(series: &[Node]) -> Vec<f64> {
    leaves(series)
        .into_iter()
        .flat_map(|(e, _)| e.values())
        .collect()
}

/// Circuit with `K` and `Zarc` expanded to (R,C) and (R,Cpe), so they can still
/// be used with ML guesses. `None` when there is nothing to rewrite. Labels
/// dropped since the matching ignores them.
pub fn expand_arcs(series: &[Node]) -> Option<Series> {
    let mut rewrote = false;
    let expanded = expand_series(series, &mut rewrote);
    rewrote.then_some(expanded)
}

fn expand_series(series: &[Node], rewrote: &mut bool) -> Series {
    series
        .iter()
        .map(|node| match node {
            Node::Element(e, _) => match e.as_parallel_pair() {
                Some((r, shunt)) => {
                    *rewrote = true;
                    Node::Parallel(vec![
                        vec![Node::Element(r, None)],
                        vec![Node::Element(shunt, None)],
                    ])
                }
                None => node.clone(),
            },
            Node::Parallel(branches) => {
                Node::Parallel(branches.iter().map(|b| expand_series(b, rewrote)).collect())
            }
        })
        .collect()
}

/// Map guessed parameter values back to `K` or `Zarc`.
pub fn contract_arc_values(series: &[Node], expanded: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(param_count(series));
    let mut cursor = 0;
    for (element, _) in leaves(series) {
        match element.as_parallel_pair() {
            Some((r, shunt)) => {
                let width = r.param_names().len() + shunt.param_names().len();
                let rebuilt = element
                    .with_parallel_pair_values(&expanded[cursor..cursor + width])
                    .expect("both pair methods cover the same variants");
                out.extend(rebuilt.values());
                cursor += width;
            }
            None => {
                let width = element.param_names().len();
                out.extend_from_slice(&expanded[cursor..cursor + width]);
                cursor += width;
            }
        }
    }
    out
}

/// Default physical-validity bounds, in the same order as `param_names()`.
pub fn param_bounds(series: &[Node]) -> Vec<(f64, f64)> {
    leaves(series)
        .into_iter()
        .flat_map(|(e, _)| e.param_bounds())
        .collect()
}

/// Physical units, in the same order as `param_names()`.
pub fn param_units(series: &[Node]) -> Vec<&'static str> {
    leaves(series)
        .into_iter()
        .flat_map(|(e, _)| e.param_units().iter().copied())
        .collect()
}

/// Total number of free parameters across all leaves.
pub fn param_count(series: &[Node]) -> usize {
    leaves(series)
        .iter()
        .map(|(e, _)| e.param_names().len())
        .sum()
}

/// Format numbers for printing: plain decimal for "normal-sized" numbers
fn fmt_num(x: f64) -> String {
    if x.is_infinite() || x == 0.0 || (1e-4..1e6).contains(&x.abs()) {
        format!("{x}")
    } else {
        format!("{x:e}")
    }
}

/// Human-readable "name = value [unit]  bounds (lo, hi)" table
/// Used by `Circuit::__repr__` and `describe_param_error`.
pub fn describe_params(
    names: &[String],
    values: &[f64],
    units: &[&str],
    bounds: &[(f64, f64)],
) -> String {
    let width = names.iter().map(String::len).max().unwrap_or(0);
    names
        .iter()
        .zip(values)
        .zip(units)
        .zip(bounds)
        .map(|(((name, &value), unit), &(lo, hi))| {
            format!(
                "  {name:width$} = {:<10} [{unit}]  bounds ({}, {})",
                fmt_num(value),
                fmt_num(lo),
                fmt_num(hi)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Standard Levenshtein edit distance between two strings.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for j in 0..=b.len() {
        dp[0][j] = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[a.len()][b.len()]
}

/// Closest known parameter name to `name`, if any candidate is close enough to
/// plausibly be a typo of it (edit distance <= max(2, name.len() / 2)).
pub fn closest_param_name<'a>(name: &str, candidates: &'a [String]) -> Option<&'a str> {
    let threshold = (name.chars().count() / 2).max(2);
    candidates
        .iter()
        .map(|c| (c.as_str(), levenshtein(name, c)))
        .min_by_key(|&(_, dist)| dist)
        .filter(|&(_, dist)| dist <= threshold)
        .map(|(c, _)| c)
}

/// Error body for a bad named-parameter dict: one line per unknown key (with a
/// "did you mean" suggestion when applicable), one line listing any missing
/// required keys, then the full valid-parameter table (name, unit, bounds).
pub fn describe_param_error(
    names: &[String],
    units: &[&str],
    bounds: &[(f64, f64)],
    unknown: &[&str],
    missing: &[&str],
) -> String {
    let mut lines = Vec::new();
    for &key in unknown {
        match closest_param_name(key, names) {
            Some(suggestion) => lines.push(format!(
                "unknown parameter {key:?} (did you mean {suggestion:?}?)"
            )),
            None => lines.push(format!("unknown parameter {key:?}")),
        }
    }
    if !missing.is_empty() {
        lines.push(format!("missing parameter(s): {missing:?}"));
    }
    lines.push(String::new());
    lines.push("valid parameters for this circuit:".to_string());
    let width = names.iter().map(String::len).max().unwrap_or(0);
    for ((name, unit), &(lo, hi)) in names.iter().zip(units).zip(bounds) {
        lines.push(format!(
            "  {name:width$}  [{unit}]  bounds ({}, {})",
            fmt_num(lo),
            fmt_num(hi)
        ));
    }
    lines.join("\n")
}

/// Rebuild the series with a new flat parameter vector, consumed in the same
/// traversal order `param_names()`/`param_values()` produced.
pub fn with_param_values(series: &[Node], values: &[f64]) -> Series {
    let mut iter = values.iter().copied();
    let result = rebuild(series, &mut iter);
    debug_assert!(
        iter.next().is_none(),
        "with_param_values: too many values supplied"
    );
    result
}

fn rebuild(series: &[Node], iter: &mut impl Iterator<Item = f64>) -> Series {
    series
        .iter()
        .map(|node| match node {
            Node::Element(e, label) => {
                let n = e.param_names().len();
                let vals: Vec<f64> = iter.by_ref().take(n).collect();
                debug_assert_eq!(
                    vals.len(),
                    n,
                    "with_param_values: not enough values supplied"
                );
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
            ParseError::TrailingInput(pos) => {
                write!(f, "unexpected trailing input at position {pos}")
            }
            ParseError::DuplicateLabel(label) => write!(f, "duplicate element label {label:?}"),
        }
    }
}

impl std::error::Error for ParseError {}

impl ParseError {
    /// Character offset into the original input this error points at, if any
    /// (`DuplicateLabel` has no single offset -- the label may appear twice at
    /// unrelated positions).
    fn position(&self) -> Option<usize> {
        match self {
            ParseError::UnexpectedChar(pos)
            | ParseError::UnknownElementCode(_, pos)
            | ParseError::TrailingInput(pos) => Some(*pos),
            ParseError::UnexpectedEnd => None,
            ParseError::DuplicateLabel(_) => None,
        }
    }
}

/// Full help text for a failed `parse()` call: the specific problem, a caret
/// pointing at the offending position in `input` (when there is one), a syntax
/// refresher, and a table of every element code `parse()` accepts.
pub fn describe_parse_error(input: &str, err: &ParseError) -> String {
    let mut lines = vec![err.to_string()];

    if let Some(pos) = err.position() {
        lines.push(String::new());
        lines.push(format!("  {input}"));
        lines.push(format!("  {}^", " ".repeat(pos)));
    }

    lines.push(String::new());
    lines.push("syntax:".to_string());
    lines.push(
        "  - every element needs a numeric label, e.g. \"R0\" and \"C1\", not \"R\" and \"C\""
            .to_string(),
    );
    lines.push("  - connect elements in series with '-', e.g. \"R0-C1\"".to_string());
    lines.push("  - connect elements in parallel with '(...)' or 'p(...)', e.g. \"R0-(R1,C1)\" or \"R0-p(R1,C1)\"".to_string());
    lines.push(String::new());
    lines.push("available elements (code, parameters, [units]):".to_string());
    lines.push(Element::describe_codes());

    lines.join("\n")
}

/// Parse a circuit topology string, e.g. `"R0-p(R1,Cpe1)"`, `"R0-(R1,Cpe1)"`, or
/// `"R0-p(R1-C1,R2-Cpe2)"`.
///
/// Grammar:
/// ```text
/// series   := term ('-' term)*
/// term     := parallel | element
/// parallel := 'p'? '(' series (',' series)* ')'
/// element  := code digits
/// ```
/// A parallel group can be written `(...)` or `p(...)`
/// `code` is one of the known element codes (matched against the longest run of
/// letters, so e.g. "Tlmq5" and "T5" are unambiguous). The string carries no
/// parameter values -- every parsed element gets a placeholder default
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
        return parse_parallel_body(chars, pos);
    }
    if *pos < chars.len() && chars[*pos] == '(' {
        *pos += 1;
        return parse_parallel_body(chars, pos);
    }
    parse_element(chars, pos)
}

/// The `series (',' series)* ')'` tail shared by both `p(...)` and bare `(...)`
/// parallel syntax -- called just after the opening paren has been consumed.
fn parse_parallel_body(chars: &[char], pos: &mut usize) -> Result<Node, ParseError> {
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

    let element =
        Element::default_for_code(&code).ok_or(ParseError::UnknownElementCode(code, start))?;
    Ok(Node::Element(element, Some(label)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: Complex64, b: Complex64, tol: f64) {
        assert!((a - b).norm() < tol, "{:?} != {:?}", a, b);
    }

    #[test]
    fn expanding_arcs_leaves_impedance_unchanged() {
        let arcs = [
            Element::K {
                r: 25.0,
                tau_k: 1e-3,
            },
            Element::Zarc {
                r: 40.0,
                tau_k: 2e-4,
                gamma: 0.82,
            },
            // gamma = 1 is where Zarc and K coincide
            Element::Zarc {
                r: 5.0,
                tau_k: 1.0,
                gamma: 1.0,
            },
        ];
        for arc in arcs {
            let written = vec![Node::Element(arc, None)];
            let expanded = expand_arcs(&written).expect("K and Zarc both expand");
            for decade in -4..8 {
                let omega = 10f64.powi(decade);
                let want = impedance(&written, omega);
                let got = impedance(&expanded, omega);
                assert!(
                    (want - got).norm() < 1e-9 * want.norm(),
                    "{arc:?} at omega={omega}: {want:?} != {got:?}"
                );
            }
        }
    }

    #[test]
    fn contract_arc_values_inverts_the_expansion() {
        let written = with_param_values(
            &parse("R0-Zarc1-K2-Cpe3").unwrap(),
            &[3.0, 40.0, 2e-4, 0.82, 25.0, 1e-3, 5e-5, 0.9],
        );
        let expanded = expand_arcs(&written).expect("circuit contains K and Zarc");
        let round_tripped = contract_arc_values(&written, &param_values(&expanded));
        for (got, want) in round_tripped.iter().zip(param_values(&written)) {
            assert!(
                (got - want).abs() < 1e-9 * want.abs(),
                "{round_tripped:?} != {:?}",
                param_values(&written)
            );
        }
    }

    #[test]
    fn expand_arcs_is_none_when_there_is_nothing_to_rewrite() {
        assert!(expand_arcs(&parse("R0-(R1,Cpe1)").unwrap()).is_none());
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
        let circuit = vec![
            r(1.0),
            Node::Parallel(vec![vec![r(2.0)], vec![cpe(3.0, 0.5)]]),
            cpe(4.0, 0.9),
        ];
        assert_eq!(
            param_names(&circuit),
            vec![
                "R0.r",
                "R1.r",
                "Cpe0.q",
                "Cpe0.alpha",
                "Cpe1.q",
                "Cpe1.alpha"
            ]
        );
    }

    #[test]
    fn with_param_values_roundtrips_through_param_values() {
        let circuit = vec![
            r(1.0),
            Node::Parallel(vec![vec![r(2.0)], vec![cpe(3.0, 0.5)]]),
            cpe(4.0, 0.9),
        ];
        let values = param_values(&circuit);
        let rebuilt = with_param_values(&circuit, &values);
        for &omega in &[0.1, 1.0, 100.0] {
            assert_close(
                impedance(&circuit, omega),
                impedance(&rebuilt, omega),
                1e-12,
            );
        }

        let mut doubled = values.clone();
        for v in doubled.iter_mut() {
            *v *= 2.0;
        }
        let scaled = with_param_values(&circuit, &doubled);
        assert_eq!(param_values(&scaled), doubled);
    }

    #[test]
    fn impedance_with_params_matches_with_param_values_then_impedance() {
        let circuit = vec![
            r(1.0),
            Node::Parallel(vec![vec![r(2.0)], vec![cpe(3.0, 0.5)]]),
            cpe(4.0, 0.9),
        ];
        let values = param_values(&circuit);
        let mut perturbed = values.clone();
        for v in perturbed.iter_mut() {
            *v *= 1.7;
        }
        let rebuilt = with_param_values(&circuit, &perturbed);
        for &omega in &[0.1, 1.0, 100.0] {
            assert_close(
                impedance_with_params(&circuit, &perturbed, omega),
                impedance(&rebuilt, omega),
                1e-12,
            );
        }
    }

    #[test]
    fn param_count_matches_flattened_length() {
        let circuit = vec![
            r(1.0),
            Node::Parallel(vec![vec![r(2.0)], vec![cpe(3.0, 0.5)]]),
            cpe(4.0, 0.9),
        ];
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
        assert_eq!(
            param_names(&circuit),
            vec!["R0.r", "R1.r", "Cpe1.q", "Cpe1.alpha"]
        );
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
    fn bare_parens_are_equivalent_to_p_parens() {
        let bare = parse("R0-(R1,Cpe1)").unwrap();
        let with_p = parse("R0-p(R1,Cpe1)").unwrap();
        assert_eq!(param_names(&bare), param_names(&with_p));
        for &omega in &[0.1, 1.0, 100.0] {
            assert_close(impedance(&bare, omega), impedance(&with_p, omega), 1e-12);
        }
    }

    #[test]
    fn bare_parens_nest_like_p_parens() {
        let circuit = parse("(R0,(R1,C1))").unwrap();
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
        assert!(matches!(
            parse("R0-p(R1,C1"),
            Err(ParseError::UnexpectedEnd)
        ));
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
        assert_eq!(
            param_names(&circuit),
            vec![
                "CPE0.q",
                "CPE0.alpha",
                "TLMQ1.r_ion",
                "TLMQ1.qs",
                "TLMQ1.gamma"
            ]
        );
    }

    #[test]
    fn describe_params_contains_every_name_value_unit_and_bound() {
        let circuit = parse("R0-Cpe1").unwrap();
        let names = param_names(&circuit);
        let values = param_values(&circuit);
        let units = param_units(&circuit);
        let bounds = param_bounds(&circuit);
        let text = describe_params(&names, &values, &units, &bounds);
        for name in &names {
            assert!(text.contains(name.as_str()), "missing {name} in {text}");
        }
        assert!(text.contains("ohm"));
        assert!(text.contains("ohm^-1*s^alpha"));
    }

    #[test]
    fn closest_param_name_suggests_near_miss_typo() {
        let names = vec![
            "R0.r".to_string(),
            "Cpe1.q".to_string(),
            "Cpe1.alpha".to_string(),
        ];
        assert_eq!(closest_param_name("Cpe1.alph", &names), Some("Cpe1.alpha"));
    }

    #[test]
    fn closest_param_name_gives_no_suggestion_for_unrelated_key() {
        let names = vec![
            "R0.r".to_string(),
            "Cpe1.q".to_string(),
            "Cpe1.alpha".to_string(),
        ];
        assert_eq!(closest_param_name("bogus", &names), None);
    }

    #[test]
    fn describe_param_error_lists_all_valid_names_and_suggests_typo_fix() {
        let circuit = parse("R0-Cpe1").unwrap();
        let names = param_names(&circuit);
        let units = param_units(&circuit);
        let bounds = param_bounds(&circuit);
        let text = describe_param_error(&names, &units, &bounds, &["Cpe1.alph"], &[]);
        assert!(text.contains("did you mean \"Cpe1.alpha\"?"));
        for name in &names {
            assert!(text.contains(name.as_str()), "missing {name} in {text}");
        }
    }

    #[test]
    fn describe_parse_error_points_at_offending_position() {
        let err = parse("R").unwrap_err();
        let text = describe_parse_error("R", &err);
        // "R" is at column 2 (after the "  " prefix), so the offending position
        // (1, right after the code with no digits) lines up one column further.
        assert!(text.contains("  R\n"), "missing input line in:\n{text}");
        assert!(
            text.contains("   ^"),
            "caret not aligned under position 1 in:\n{text}"
        );
    }

    #[test]
    fn describe_parse_error_lists_syntax_and_element_table() {
        let err = parse("Q0").unwrap_err();
        let text = describe_parse_error("Q0", &err);
        assert!(text.contains("series"));
        assert!(text.contains("parallel"));
        assert!(text.contains("CPE"));
        assert!(text.contains("Zarc"));
        for &code in Element::CODES {
            assert!(
                text.contains(code),
                "{code} missing from parse-error help:\n{text}"
            );
        }
    }

    #[test]
    fn describe_param_error_lists_missing_keys() {
        let circuit = parse("R0-C1").unwrap();
        let names = param_names(&circuit);
        let units = param_units(&circuit);
        let bounds = param_bounds(&circuit);
        let text = describe_param_error(&names, &units, &bounds, &[], &["C1.c"]);
        assert!(text.contains("missing parameter(s)"));
        assert!(text.contains("C1.c"));
    }
}
