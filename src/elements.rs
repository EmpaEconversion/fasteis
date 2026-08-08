use num_complex::Complex64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Element {
    R {
        r: f64,
    },
    C {
        c: f64,
    },
    L {
        l: f64,
    },
    La {
        l: f64,
        alpha: f64,
    },
    Cpe {
        q: f64,
        alpha: f64,
    },
    W {
        aw: f64,
    },
    Wo {
        z0: f64,
        tau: f64,
    },
    Ws {
        z0: f64,
        tau: f64,
    },
    G {
        rg: f64,
        tg: f64,
    },
    Gs {
        rg: f64,
        tg: f64,
        phi: f64,
    },
    K {
        r: f64,
        tau_k: f64,
    },
    Zarc {
        r: f64,
        tau_k: f64,
        gamma: f64,
    },
    Tlmq {
        r_ion: f64,
        qs: f64,
        gamma: f64,
    },
    T {
        a_coeff: f64,
        b_coeff: f64,
        a_param: f64,
        b_param: f64,
    },
}

/// `z.powf(exponent)`, special-cased for exponents of 1.0 and 0.5.
/// Can skip round-trips to_polar and from_polar.
#[inline]
fn complex_powf(z: Complex64, exponent: f64) -> Complex64 {
    if exponent == 1.0 {
        z // skip completey
    } else if exponent == 0.5 {
        z.sqrt() // faster shortcut for pure real or pure imaginary z
    } else {
        z.powf(exponent)
    }
}

/// `tanh` can overflow if the input is too large.
/// Fix it to 1 above tanh(20), which is within f64 precision anyway.
#[inline]
fn stable_tanh(z: Complex64) -> Complex64 {
    const SATURATES: f64 = 20.0;
    if z.re.abs() > SATURATES {
        Complex64::new(z.re.signum(), 0.0)
    } else {
        z.tanh()
    }
}

/// `cosh(z)/sinh(z)`, saturating for the same reason as `stable_tanh`.
#[inline]
fn stable_coth(z: Complex64) -> Complex64 {
    const SATURATES: f64 = 20.0;
    if z.re.abs() > SATURATES {
        Complex64::new(z.re.signum(), 0.0)
    } else {
        z.cosh() / z.sinh()
    }
}

/// `1/sinh(z)`, saturates at zero instead of overflowing.
#[inline]
fn stable_cosech(z: Complex64) -> Complex64 {
    const SATURATES: f64 = 20.0;
    if z.re.abs() > SATURATES {
        Complex64::new(0.0, 0.0)
    } else {
        z.sinh().inv()
    }
}

impl Element {
    pub fn impedance(&self, omega: f64) -> Complex64 {
        let j = Complex64::new(0.0, 1.0);
        let jw = j * omega;
        match *self {
            Element::R { r } => Complex64::new(r, 0.0),
            Element::C { c } => Complex64::new(1.0, 0.0) / (c * jw),
            Element::L { l } => l * jw,
            Element::La { l, alpha } => complex_powf(l * jw, alpha),
            Element::Cpe { q, alpha } => (q * complex_powf(jw, alpha)).inv(),
            Element::W { aw } => {
                aw * (Complex64::new(1.0, 0.0) - j) / Complex64::new(omega.sqrt(), 0.0)
            }
            Element::Wo { z0, tau } => {
                let x = (jw * tau).sqrt();
                z0 / (x * stable_tanh(x))
            }
            Element::Ws { z0, tau } => {
                let x = (jw * tau).sqrt();
                z0 * stable_tanh(x) / x
            }
            Element::G { rg, tg } => rg / (Complex64::new(1.0, 0.0) + jw * tg).sqrt(),
            Element::Gs { rg, tg, phi } => {
                let s = (Complex64::new(1.0, 0.0) + jw * tg).sqrt();
                rg / (s * stable_tanh(s * phi))
            }
            Element::K { r, tau_k } => r / (Complex64::new(1.0, 0.0) + jw * tau_k),
            Element::Zarc { r, tau_k, gamma } => {
                r / (Complex64::new(1.0, 0.0) + complex_powf(jw * tau_k, gamma))
            }
            Element::Tlmq { r_ion, qs, gamma } => {
                let zs = (qs * complex_powf(jw, gamma)).inv();
                let y = (r_ion / zs).sqrt();
                (r_ion * zs).sqrt() / stable_tanh(y)
            }
            Element::T {
                a_coeff,
                b_coeff,
                a_param,
                b_param,
            } => {
                let beta = (Complex64::new(a_param, 0.0) + jw * b_param).sqrt();
                a_coeff * stable_coth(beta) / beta + b_coeff * stable_cosech(beta) / beta
            }
        }
    }

    /// Same as `impedance()`, but takes iterates straight through parameters
    /// instead of looking up the fields. This lets a fit's inner loop use a
    /// fixed circuit topology and calculate impedance directly from a parameter
    /// vector. Using `with_values()` + `impedance()` would rebuild the whole
    /// `Element`/`Node` tree every evaluation. Gets a ~5% perf boost.
    pub fn impedance_from_iter(
        &self,
        iter: &mut impl Iterator<Item = f64>,
        omega: f64,
    ) -> Complex64 {
        let j = Complex64::new(0.0, 1.0);
        let jw = j * omega;
        let mut next = || {
            iter.next()
                .expect("impedance_from_iter: not enough values supplied")
        };
        match *self {
            Element::R { .. } => Complex64::new(next(), 0.0),
            Element::C { .. } => Complex64::new(1.0, 0.0) / (next() * jw),
            Element::L { .. } => next() * jw,
            Element::La { .. } => {
                let l = next();
                let alpha = next();
                complex_powf(l * jw, alpha)
            }
            Element::Cpe { .. } => {
                let q = next();
                let alpha = next();
                (q * complex_powf(jw, alpha)).inv()
            }
            Element::W { .. } => {
                next() * (Complex64::new(1.0, 0.0) - j) / Complex64::new(omega.sqrt(), 0.0)
            }
            Element::Wo { .. } => {
                let z0 = next();
                let tau = next();
                let x = (jw * tau).sqrt();
                z0 / (x * stable_tanh(x))
            }
            Element::Ws { .. } => {
                let z0 = next();
                let tau = next();
                let x = (jw * tau).sqrt();
                z0 * stable_tanh(x) / x
            }
            Element::G { .. } => {
                let rg = next();
                let tg = next();
                rg / (Complex64::new(1.0, 0.0) + jw * tg).sqrt()
            }
            Element::Gs { .. } => {
                let rg = next();
                let tg = next();
                let phi = next();
                let s = (Complex64::new(1.0, 0.0) + jw * tg).sqrt();
                rg / (s * stable_tanh(s * phi))
            }
            Element::K { .. } => {
                let r = next();
                let tau_k = next();
                r / (Complex64::new(1.0, 0.0) + jw * tau_k)
            }
            Element::Zarc { .. } => {
                let r = next();
                let tau_k = next();
                let gamma = next();
                r / (Complex64::new(1.0, 0.0) + complex_powf(jw * tau_k, gamma))
            }
            Element::Tlmq { .. } => {
                let r_ion = next();
                let qs = next();
                let gamma = next();
                let zs = (qs * complex_powf(jw, gamma)).inv();
                let y = (r_ion / zs).sqrt();
                (r_ion * zs).sqrt() / stable_tanh(y)
            }
            Element::T { .. } => {
                let a_coeff = next();
                let b_coeff = next();
                let a_param = next();
                let b_param = next();
                let beta = (Complex64::new(a_param, 0.0) + jw * b_param).sqrt();
                a_coeff * stable_coth(beta) / beta + b_coeff * stable_cosech(beta) / beta
            }
        }
    }
}

impl Element {
    /// Short tag used for auto-generated parameter names (e.g. "R", "Cpe", "Zarc").
    /// This mirrors Rust variant names, not python.rs's constructor names, which
    /// intentionally diverge in case for a couple of elements (CPE -> Cpe, TLMQ -> Tlmq).
    pub fn type_tag(&self) -> &'static str {
        match self {
            Element::R { .. } => "R",
            Element::C { .. } => "C",
            Element::L { .. } => "L",
            Element::La { .. } => "La",
            Element::Cpe { .. } => "Cpe",
            Element::W { .. } => "W",
            Element::Wo { .. } => "Wo",
            Element::Ws { .. } => "Ws",
            Element::G { .. } => "G",
            Element::Gs { .. } => "Gs",
            Element::K { .. } => "K",
            Element::Zarc { .. } => "Zarc",
            Element::Tlmq { .. } => "Tlmq",
            Element::T { .. } => "T",
        }
    }

    /// Field names in the same order `values()`/`with_values()` use.
    pub fn param_names(&self) -> &'static [&'static str] {
        match self {
            Element::R { .. } => &["r"],
            Element::C { .. } => &["c"],
            Element::L { .. } => &["l"],
            Element::La { .. } => &["l", "alpha"],
            Element::Cpe { .. } => &["q", "alpha"],
            Element::W { .. } => &["aw"],
            Element::Wo { .. } => &["z0", "tau"],
            Element::Ws { .. } => &["z0", "tau"],
            Element::G { .. } => &["rg", "tg"],
            Element::Gs { .. } => &["rg", "tg", "phi"],
            Element::K { .. } => &["r", "tau_k"],
            Element::Zarc { .. } => &["r", "tau_k", "gamma"],
            Element::Tlmq { .. } => &["r_ion", "qs", "gamma"],
            Element::T { .. } => &["a_coeff", "b_coeff", "a_param", "b_param"],
        }
    }

    /// Current field values, same order as `param_names()`.
    pub fn values(&self) -> Vec<f64> {
        match *self {
            Element::R { r } => vec![r],
            Element::C { c } => vec![c],
            Element::L { l } => vec![l],
            Element::La { l, alpha } => vec![l, alpha],
            Element::Cpe { q, alpha } => vec![q, alpha],
            Element::W { aw } => vec![aw],
            Element::Wo { z0, tau } => vec![z0, tau],
            Element::Ws { z0, tau } => vec![z0, tau],
            Element::G { rg, tg } => vec![rg, tg],
            Element::Gs { rg, tg, phi } => vec![rg, tg, phi],
            Element::K { r, tau_k } => vec![r, tau_k],
            Element::Zarc { r, tau_k, gamma } => vec![r, tau_k, gamma],
            Element::Tlmq { r_ion, qs, gamma } => vec![r_ion, qs, gamma],
            Element::T {
                a_coeff,
                b_coeff,
                a_param,
                b_param,
            } => vec![a_coeff, b_coeff, a_param, b_param],
        }
    }

    /// Rebuild this variant with new values. `values.len()` must equal `param_names().len()`.
    pub fn with_values(&self, values: &[f64]) -> Element {
        match *self {
            Element::R { .. } => Element::R { r: values[0] },
            Element::C { .. } => Element::C { c: values[0] },
            Element::L { .. } => Element::L { l: values[0] },
            Element::La { .. } => Element::La {
                l: values[0],
                alpha: values[1],
            },
            Element::Cpe { .. } => Element::Cpe {
                q: values[0],
                alpha: values[1],
            },
            Element::W { .. } => Element::W { aw: values[0] },
            Element::Wo { .. } => Element::Wo {
                z0: values[0],
                tau: values[1],
            },
            Element::Ws { .. } => Element::Ws {
                z0: values[0],
                tau: values[1],
            },
            Element::G { .. } => Element::G {
                rg: values[0],
                tg: values[1],
            },
            Element::Gs { .. } => Element::Gs {
                rg: values[0],
                tg: values[1],
                phi: values[2],
            },
            Element::K { .. } => Element::K {
                r: values[0],
                tau_k: values[1],
            },
            Element::Zarc { .. } => Element::Zarc {
                r: values[0],
                tau_k: values[1],
                gamma: values[2],
            },
            Element::Tlmq { .. } => Element::Tlmq {
                r_ion: values[0],
                qs: values[1],
                gamma: values[2],
            },
            Element::T { .. } => Element::T {
                a_coeff: values[0],
                b_coeff: values[1],
                a_param: values[2],
                b_param: values[3],
            },
        }
    }

    /// Physical units per parameter, `"-"` = dimensionless
    pub fn param_units(&self) -> &'static [&'static str] {
        match self {
            Element::R { .. } => &["ohm"],
            Element::C { .. } => &["F"],
            Element::L { .. } => &["H"],
            Element::La { .. } => &["H*s", "-"],
            Element::Cpe { .. } => &["ohm^-1*s^alpha", "-"],
            Element::W { .. } => &["ohm*s^-0.5"],
            Element::Wo { .. } => &["ohm", "s"],
            Element::Ws { .. } => &["ohm", "s"],
            Element::G { .. } => &["ohm", "s"],
            Element::Gs { .. } => &["ohm", "s", "-"],
            Element::K { .. } => &["ohm", "s"],
            Element::Zarc { .. } => &["ohm", "s", "-"],
            Element::Tlmq { .. } => &["ohm", "F*s^(gamma-1)", "-"],
            Element::T { .. } => &["ohm*m^2", "ohm*m^2", "-", "s"],
        }
    }

    /// Default physical-validity bounds per parameter, derived from `param_names()`:
    /// fields named "alpha" or "gamma" are fractional exponents bounded to [0, 1];
    /// everything else is a positive magnitude/time-constant bounded to (~0, inf).
    pub fn param_bounds(&self) -> Vec<(f64, f64)> {
        self.param_names()
            .iter()
            .map(|&name| match name {
                "alpha" | "gamma" => (0.0, 1.0),
                _ => (1e-12, f64::INFINITY),
            })
            .collect()
    }

    /// All element codes accepted by `circuit::parse()`, in the canonical spelling
    /// used in help/error messages -- matches python.rs's static-method names
    /// (e.g. "CPE" not "Cpe", "TLMQ" not "Tlmq"), since that's the spelling users
    /// see everywhere else. `default_for_code` also accepts the lowercase-tail
    /// variants ("Cpe", "Tlmq") for backward compatibility with older strings.
    pub const CODES: &'static [&'static str] = &[
        "R", "C", "L", "La", "CPE", "W", "Wo", "Ws", "G", "Gs", "K", "Zarc", "TLMQ", "T",
    ];

    /// One line per known element code: the code, then each parameter name and
    /// unit. Used to help users discover valid circuit-string syntax after a
    /// parse error.
    pub fn describe_codes() -> String {
        let width = Element::CODES.iter().map(|c| c.len()).max().unwrap_or(0);
        Element::CODES
            .iter()
            .map(|&code| {
                let element = Element::default_for_code(code)
                    .expect("Element::CODES entries must all be valid");
                let params: Vec<String> = element
                    .param_names()
                    .iter()
                    .zip(element.param_units())
                    .map(|(name, unit)| format!("{name} [{unit}]"))
                    .collect();
                format!("  {code:width$}  {}", params.join(", "))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Build a placeholder-valued element for the given code, used when parsing
    /// a circuit topology string (which carries no parameter values).
    /// Accepts both the internal spelling ("Cpe", "Tlmq") and python.rs's
    /// static-method spelling ("CPE", "TLMQ").
    pub fn default_for_code(code: &str) -> Option<Element> {
        let element = match code {
            "R" => Element::R { r: 1.0 },
            "C" => Element::C { c: 1e-3 },
            "L" => Element::L { l: 1e-6 },
            "La" => Element::La {
                l: 1e-6,
                alpha: 0.8,
            },
            "Cpe" | "CPE" => Element::Cpe {
                q: 1e-3,
                alpha: 0.8,
            },
            "W" => Element::W { aw: 1.0 },
            // solid-state diffusion takes seconds to minutes for micron particles
            "Wo" => Element::Wo { z0: 1.0, tau: 10.0 },
            "Ws" => Element::Ws { z0: 1.0, tau: 10.0 },
            "G" => Element::G { rg: 1.0, tg: 1.0 },
            "Gs" => Element::Gs {
                rg: 1.0,
                tg: 1.0,
                phi: 1.0,
            },
            "K" => Element::K {
                r: 1.0,
                tau_k: 1e-3,
            },
            "Zarc" => Element::Zarc {
                r: 1.0,
                tau_k: 1e-3,
                gamma: 0.8,
            },
            "Tlmq" | "TLMQ" => Element::Tlmq {
                r_ion: 1.0,
                qs: 1e-3,
                gamma: 0.8,
            },
            "T" => Element::T {
                a_coeff: 1.0,
                b_coeff: 1.0,
                a_param: 1.0,
                b_param: 1.0,
            },
            _ => return None,
        };
        Some(element)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: Complex64, b: Complex64, tol: f64) {
        assert!((a - b).norm() < tol, "{:?} != {:?}", a, b);
    }

    #[test]
    fn complex_powf_boundary_cases_match_general_powf() {
        // Complex z, including the pure-imaginary jw shape that
        // Cpe/La/Zarc/Tlmq actually feed in.
        let zs = [
            Complex64::new(3.0, 4.0),
            Complex64::new(-2.0, 7.5),
            Complex64::new(0.0, 1e-3), // jw at a small omega
            Complex64::new(0.0, 1e6),  // jw at a large omega
            Complex64::new(-1.0, -1.0),
        ];
        for &z in &zs {
            assert_close(complex_powf(z, 1.0), z.powf(1.0), 1e-9);
            assert_close(complex_powf(z, 0.5), z.sqrt(), 1e-9);
            assert_close(complex_powf(z, 0.5), z.powf(0.5), 1e-9);
        }
    }

    #[test]
    fn resistor_is_frequency_independent() {
        let r = Element::R { r: 5.0 };
        for &omega in &[0.0, 1.0, 1000.0] {
            assert_close(r.impedance(omega), Complex64::new(5.0, 0.0), 1e-12);
        }
    }

    #[test]
    fn capacitor_matches_hand_calc() {
        let c = Element::C { c: 1e-6 };
        assert_close(c.impedance(1000.0), Complex64::new(0.0, -1000.0), 1e-9);
    }

    #[test]
    fn inductor_matches_hand_calc() {
        let l = Element::L { l: 2.0 };
        assert_close(l.impedance(3.0), Complex64::new(0.0, 6.0), 1e-12);
    }

    #[test]
    fn modified_inductance_matches_inductor_at_alpha_one() {
        let l = Element::L { l: 4.0 };
        let la = Element::La { l: 4.0, alpha: 1.0 };
        assert_close(l.impedance(7.0), la.impedance(7.0), 1e-9);
    }

    #[test]
    fn cpe_matches_capacitor_and_resistor_at_boundary_alphas() {
        let q = 3.0;
        let cpe_cap = Element::Cpe { q, alpha: 1.0 };
        let cap = Element::C { c: q };
        assert_close(cpe_cap.impedance(50.0), cap.impedance(50.0), 1e-9);

        let cpe_res = Element::Cpe { q, alpha: 0.0 };
        let res = Element::R { r: 1.0 / q };
        assert_close(cpe_res.impedance(50.0), res.impedance(50.0), 1e-9);
    }

    #[test]
    fn warburg_matches_hand_calc() {
        let w = Element::W { aw: 1.0 };
        assert_close(w.impedance(1.0), Complex64::new(1.0, -1.0), 1e-9);
    }

    #[test]
    fn gerischer_matches_resistor_at_zero_tg() {
        let g = Element::G { rg: 10.0, tg: 0.0 };
        let r = Element::R { r: 10.0 };
        assert_close(g.impedance(100.0), r.impedance(100.0), 1e-9);
    }

    #[test]
    fn zarc_matches_k_at_gamma_one() {
        let zarc = Element::Zarc {
            r: 20.0,
            tau_k: 0.5,
            gamma: 1.0,
        };
        let k = Element::K {
            r: 20.0,
            tau_k: 0.5,
        };
        assert_close(zarc.impedance(10.0), k.impedance(10.0), 1e-9);
    }

    #[test]
    fn warburg_open_and_short_differ() {
        let wo = Element::Wo { z0: 1.0, tau: 1.0 };
        let ws = Element::Ws { z0: 1.0, tau: 1.0 };
        let zo = wo.impedance(1.0);
        let zs = ws.impedance(1.0);
        assert!(
            (zo - zs).norm() > 1e-6,
            "Wo and Ws must not coincide: {:?} vs {:?}",
            zo,
            zs
        );

        // Reference values computed independently via Python:
        //   x = cmath.sqrt(1j*1.0*1.0)
        //   wo = 1.0/(x*cmath.tanh(x)); ws = 1.0*cmath.tanh(x)/x
        assert_close(zo, Complex64::new(0.3312380920, -1.0220127244), 1e-6);
        assert_close(zs, Complex64::new(0.8854508123, -0.2869778728), 1e-6);
    }

    #[test]
    fn t_and_tlmq_are_finite() {
        let t = Element::T {
            a_coeff: 1.0,
            b_coeff: 2.0,
            a_param: 0.5,
            b_param: 0.1,
        };
        let z = t.impedance(10.0);
        assert!(z.re.is_finite() && z.im.is_finite());

        let tlmq = Element::Tlmq {
            r_ion: 5.0,
            qs: 1e-4,
            gamma: 0.8,
        };
        let z2 = tlmq.impedance(10.0);
        assert!(z2.re.is_finite() && z2.im.is_finite());
    }

    #[test]
    fn param_names_values_with_values_roundtrip_all_variants() {
        let samples = [
            Element::R { r: 5.0 },
            Element::C { c: 1e-6 },
            Element::L { l: 2.0 },
            Element::La { l: 4.0, alpha: 0.9 },
            Element::Cpe {
                q: 3.0,
                alpha: 0.85,
            },
            Element::W { aw: 1.0 },
            Element::Wo { z0: 1.0, tau: 1.0 },
            Element::Ws { z0: 1.0, tau: 1.0 },
            Element::G { rg: 10.0, tg: 0.5 },
            Element::Gs {
                rg: 10.0,
                tg: 0.5,
                phi: 0.3,
            },
            Element::K {
                r: 20.0,
                tau_k: 0.5,
            },
            Element::Zarc {
                r: 20.0,
                tau_k: 0.5,
                gamma: 0.9,
            },
            Element::Tlmq {
                r_ion: 5.0,
                qs: 1e-4,
                gamma: 0.8,
            },
            Element::T {
                a_coeff: 1.0,
                b_coeff: 2.0,
                a_param: 0.5,
                b_param: 0.1,
            },
        ];
        for e in samples {
            assert_eq!(e.param_names().len(), e.values().len());
            assert_eq!(e.with_values(&e.values()), e);
        }
    }

    #[test]
    fn param_units_lengths_match_param_names_for_all_variants() {
        let samples = [
            Element::R { r: 5.0 },
            Element::C { c: 1e-6 },
            Element::L { l: 2.0 },
            Element::La { l: 4.0, alpha: 0.9 },
            Element::Cpe {
                q: 3.0,
                alpha: 0.85,
            },
            Element::W { aw: 1.0 },
            Element::Wo { z0: 1.0, tau: 1.0 },
            Element::Ws { z0: 1.0, tau: 1.0 },
            Element::G { rg: 10.0, tg: 0.5 },
            Element::Gs {
                rg: 10.0,
                tg: 0.5,
                phi: 0.3,
            },
            Element::K {
                r: 20.0,
                tau_k: 0.5,
            },
            Element::Zarc {
                r: 20.0,
                tau_k: 0.5,
                gamma: 0.9,
            },
            Element::Tlmq {
                r_ion: 5.0,
                qs: 1e-4,
                gamma: 0.8,
            },
            Element::T {
                a_coeff: 1.0,
                b_coeff: 2.0,
                a_param: 0.5,
                b_param: 0.1,
            },
        ];
        for e in samples {
            assert_eq!(e.param_units().len(), e.param_names().len());
        }
    }

    #[test]
    fn param_units_exact_values() {
        assert_eq!(Element::R { r: 1.0 }.param_units(), &["ohm"]);
        assert_eq!(
            Element::Cpe { q: 1.0, alpha: 0.5 }.param_units(),
            &["ohm^-1*s^alpha", "-"]
        );
        assert_eq!(
            Element::Zarc {
                r: 1.0,
                tau_k: 1.0,
                gamma: 0.5
            }
            .param_units(),
            &["ohm", "s", "-"]
        );
    }

    #[test]
    fn every_code_is_a_valid_default_for_code_and_appears_in_describe_codes() {
        let described = Element::describe_codes();
        for &code in Element::CODES {
            assert!(
                Element::default_for_code(code).is_some(),
                "{code} has no default_for_code entry"
            );
            assert!(
                described.contains(code),
                "{code} missing from describe_codes() output:\n{described}"
            );
        }
    }

    #[test]
    fn hyperbolic_elements_stay_finite_across_a_wide_sweep() {
        // sqrt(j w tau) grows without bound with frequency, and the naive
        // sinh/cosh form of tanh overflows to inf, then inf/inf = NaN.
        let elements = [
            Element::Wo { z0: 1.0, tau: 10.0 },
            Element::Ws { z0: 1.0, tau: 10.0 },
            Element::Gs {
                rg: 1.0,
                tg: 10.0,
                phi: 5.0,
            },
            Element::Tlmq {
                r_ion: 1.0,
                qs: 1e-3,
                gamma: 0.8,
            },
            Element::T {
                a_coeff: 1.0,
                b_coeff: 1.0,
                a_param: 1.0,
                b_param: 10.0,
            },
        ];
        for element in elements {
            for decade in -6..12 {
                let omega = 10f64.powi(decade);
                let z = element.impedance(omega);
                assert!(
                    z.re.is_finite() && z.im.is_finite(),
                    "{element:?} gave {z:?} at omega=1e{decade}"
                );
            }
        }
    }

    #[test]
    fn param_bounds_clamps_alpha_and_gamma_to_unit_interval() {
        let cpe = Element::Cpe { q: 3.0, alpha: 0.5 };
        assert_eq!(cpe.param_bounds(), vec![(1e-12, f64::INFINITY), (0.0, 1.0)]);

        let zarc = Element::Zarc {
            r: 1.0,
            tau_k: 1.0,
            gamma: 0.5,
        };
        assert_eq!(
            zarc.param_bounds(),
            vec![(1e-12, f64::INFINITY), (1e-12, f64::INFINITY), (0.0, 1.0)]
        );

        // Gs.phi is a tanh-argument scale factor, not a fractional exponent -- must not be unit-clamped.
        let gs = Element::Gs {
            rg: 1.0,
            tg: 1.0,
            phi: 0.5,
        };
        assert_eq!(
            gs.param_bounds(),
            vec![
                (1e-12, f64::INFINITY),
                (1e-12, f64::INFINITY),
                (1e-12, f64::INFINITY)
            ]
        );
    }
}
