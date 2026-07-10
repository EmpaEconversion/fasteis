use num_complex::Complex64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Element {
    R { r: f64 },
    C { c: f64 },
    L { l: f64 },
    La { l: f64, alpha: f64 },
    Cpe { q: f64, alpha: f64 },
    W { aw: f64 },
    Wo { z0: f64, tau: f64 },
    Ws { z0: f64, tau: f64 },
    G { rg: f64, tg: f64 },
    Gs { rg: f64, tg: f64, phi: f64 },
    K { r: f64, tau_k: f64 },
    Zarc { r: f64, tau_k: f64, gamma: f64 },
    Tlmq { r_ion: f64, qs: f64, gamma: f64 },
    T { a_coeff: f64, b_coeff: f64, a_param: f64, b_param: f64 },
}

/// `z.powf(exponent)`, special-cased for exponents of 1.0 and 0.5.
/// Can skip round-trips to_polar and from_polar.
#[inline]
fn complex_powf(z: Complex64, exponent: f64) -> Complex64 {
    if exponent == 1.0 {
        z  // skip completey
    } else if exponent == 0.5 {
        z.sqrt()  // faster shortcut for pure real or pure imaginary z
    } else {
        z.powf(exponent)
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
            Element::W { aw } => aw * (Complex64::new(1.0, 0.0) - j) / Complex64::new(omega.sqrt(), 0.0),
            Element::Wo { z0, tau } => {
                let x = (jw * tau).sqrt();
                z0 / (x * x.tanh())
            }
            Element::Ws { z0, tau } => {
                let x = (jw * tau).sqrt();
                z0 * x.tanh() / x
            }
            Element::G { rg, tg } => rg / (Complex64::new(1.0, 0.0) + jw * tg).sqrt(),
            Element::Gs { rg, tg, phi } => {
                let s = (Complex64::new(1.0, 0.0) + jw * tg).sqrt();
                rg / (s * (s * phi).tanh())
            }
            Element::K { r, tau_k } => r / (Complex64::new(1.0, 0.0) + jw * tau_k),
            Element::Zarc { r, tau_k, gamma } => r / (Complex64::new(1.0, 0.0) + complex_powf(jw * tau_k, gamma)),
            Element::Tlmq { r_ion, qs, gamma } => {
                let zs = (qs * complex_powf(jw, gamma)).inv();
                let y = (r_ion / zs).sqrt();
                (r_ion * zs).sqrt() / y.tanh()
            }
            Element::T { a_coeff, b_coeff, a_param, b_param } => {
                let beta = (Complex64::new(a_param, 0.0) + jw * b_param).sqrt();
                a_coeff * (beta.cosh() / beta.sinh()) / beta + b_coeff / (beta * beta.sinh())
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

    #[test]
    fn complex_powf_boundary_cases_match_general_powf() {
        // Genuinely complex z (not real, not imaginary-zero), including the
        // pure-imaginary jw shape that Cpe/La/Zarc/Tlmq actually feed in.
        let zs = [
            Complex64::new(3.0, 4.0),
            Complex64::new(-2.0, 7.5),
            Complex64::new(0.0, 1e-3),   // jw at a small omega
            Complex64::new(0.0, 1e6),    // jw at a large omega
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
        let zarc = Element::Zarc { r: 20.0, tau_k: 0.5, gamma: 1.0 };
        let k = Element::K { r: 20.0, tau_k: 0.5 };
        assert_close(zarc.impedance(10.0), k.impedance(10.0), 1e-9);
    }

    #[test]
    fn warburg_open_and_short_differ() {
        let wo = Element::Wo { z0: 1.0, tau: 1.0 };
        let ws = Element::Ws { z0: 1.0, tau: 1.0 };
        let zo = wo.impedance(1.0);
        let zs = ws.impedance(1.0);
        assert!((zo - zs).norm() > 1e-6, "Wo and Ws must not coincide: {:?} vs {:?}", zo, zs);

        // Reference values computed independently via Python:
        //   x = cmath.sqrt(1j*1.0*1.0)
        //   wo = 1.0/(x*cmath.tanh(x)); ws = 1.0*cmath.tanh(x)/x
        assert_close(zo, Complex64::new(0.3312380920, -1.0220127244), 1e-6);
        assert_close(zs, Complex64::new(0.8854508123, -0.2869778728), 1e-6);
    }

    #[test]
    fn t_and_tlmq_are_finite() {
        let t = Element::T { a_coeff: 1.0, b_coeff: 2.0, a_param: 0.5, b_param: 0.1 };
        let z = t.impedance(10.0);
        assert!(z.re.is_finite() && z.im.is_finite());

        let tlmq = Element::Tlmq { r_ion: 5.0, qs: 1e-4, gamma: 0.8 };
        let z2 = tlmq.impedance(10.0);
        assert!(z2.re.is_finite() && z2.im.is_finite());
    }
}
