//! Neural-network initial-parameter guessing.
//!
//! Loads a `.eisnn` file produced by `training/export.py` to run inference.
//! The network reads a normalised impedance spectrum, predicts normalized
//! parameters, then scales back to real units.

use std::collections::HashMap;
use std::f64::consts::{PI, TAU};
use std::fs;
use std::io;
use std::path::Path;

use num_complex::Complex64;

const MAGIC: &[u8; 8] = b"EISNN001";

/// Tensor element encodings, matching `training/serialize_weights.py`. Weights are ~99.8% of
/// the file, so this is the only thing that meaningfully changes its size.
const DTYPE_F32: u8 = 0;
const DTYPE_F16: u8 = 1;
const DTYPE_INT8: u8 = 2;

/// Scale estimator used by `scales()`, needs to match rescaling here.
const SUPPORTED_ESTIMATORS: [&str; 1] = ["reactive_centroid"];

#[derive(Debug)]
pub enum NnError {
    Io(io::Error),
    BadMagic,
    Truncated,
    MissingKey(String),
    BadValue(String),
    UnsupportedEstimator(String),
    UnsupportedDtype(u8),
    ShapeMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    TooFewPoints(usize),
}

impl std::fmt::Display for NnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NnError::Io(e) => write!(f, "reading weights: {e}"),
            NnError::BadMagic => write!(f, "not an eisnn weight file (bad magic bytes)"),
            NnError::Truncated => write!(f, "weight file is truncated"),
            NnError::MissingKey(k) => write!(f, "weight file is missing {k:?}"),
            NnError::BadValue(k) => write!(f, "weight file has an unparsable value for {k:?}"),
            NnError::UnsupportedEstimator(name) => write!(
                f,
                "weight file was normalised with the {name:?} scale estimator, which this \
                 build does not implement (supported: {})",
                SUPPORTED_ESTIMATORS.join(", ")
            ),
            NnError::UnsupportedDtype(tag) => {
                write!(
                    f,
                    "weight file uses tensor encoding {tag}, which this build does not know"
                )
            }
            NnError::ShapeMismatch {
                name,
                expected,
                got,
            } => {
                write!(f, "tensor {name:?} has {got} elements, expected {expected}")
            }
            NnError::TooFewPoints(n) => {
                write!(f, "need at least 2 frequency points to guess, got {n}")
            }
        }
    }
}

impl std::error::Error for NnError {}

impl From<io::Error> for NnError {
    fn from(e: io::Error) -> Self {
        NnError::Io(e)
    }
}

/// A 2-D tensor stored row-major. 1-D tensors use `rows == 1`.
#[derive(Debug, Clone)]
struct Tensor {
    dims: Vec<usize>,
    data: Vec<f64>,
}

impl Tensor {
    fn len(&self) -> usize {
        self.data.len()
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], NnError> {
        let end = self.pos.checked_add(n).ok_or(NnError::Truncated)?;
        let chunk = self.data.get(self.pos..end).ok_or(NnError::Truncated)?;
        self.pos = end;
        Ok(chunk)
    }

    fn u8(&mut self) -> Result<u8, NnError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, NnError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn string(&mut self) -> Result<String, NnError> {
        let n = self.u32()? as usize;
        let raw = self.take(n)?;
        String::from_utf8(raw.to_vec()).map_err(|_| NnError::Truncated)
    }

    fn f32(&mut self) -> Result<f32, NnError> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// `n` elements of the given encoding, widened to f64.
    fn values(&mut self, dtype: u8, n: usize) -> Result<Vec<f64>, NnError> {
        match dtype {
            DTYPE_F32 => Ok(self
                .take(4 * n)?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect()),
            DTYPE_F16 => Ok(self
                .take(2 * n)?
                .chunks_exact(2)
                .map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            DTYPE_INT8 => {
                // scale precedes the payload; symmetric, so no zero point
                let scale = self.f32()? as f64;
                Ok(self
                    .take(n)?
                    .iter()
                    .map(|&b| b as i8 as f64 * scale)
                    .collect())
            }
            other => Err(NnError::UnsupportedDtype(other)),
        }
    }
}

/// IEEE 754 binary16 -> f64, including subnormals, infinities and NaN.
fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x3ff) as f64;

    match exponent {
        // subnormal: no implicit leading 1, fixed exponent of -14
        0 => sign * mantissa * 2f64.powi(-24),
        0x1f if mantissa == 0.0 => sign * f64::INFINITY,
        0x1f => f64::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15),
    }
}

/// How one parameter picks up the two scales:
/// `physical = normalised * k^a * w_c^(b + c * params[index])`.
#[derive(Debug, Clone, Copy)]
struct Scaling {
    a: f64,
    b: f64,
    c: f64,
    index: i32,
}

/// A trained guesser for one fixed circuit.
pub struct Guesser {
    circuit: String,
    param_names: Vec<String>,
    n_grid: usize,
    channels: usize,
    kernel: usize,
    groups: usize,
    dilations: Vec<usize>,
    alpha_range: (f64, f64),
    target_mean: Vec<f64>,
    target_std: Vec<f64>,
    log_params: Vec<bool>,
    scaling: Vec<Scaling>,
    tensors: HashMap<String, Tensor>,
}

fn meta<'a>(m: &'a HashMap<String, String>, key: &str) -> Result<&'a str, NnError> {
    m.get(key)
        .map(String::as_str)
        .ok_or_else(|| NnError::MissingKey(key.to_string()))
}

fn parse<T: std::str::FromStr>(m: &HashMap<String, String>, key: &str) -> Result<T, NnError> {
    meta(m, key)?
        .parse()
        .map_err(|_| NnError::BadValue(key.to_string()))
}

impl Guesser {
    /// Load a `.eisnn` container from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Guesser, NnError> {
        Self::from_bytes(&fs::read(path)?)
    }

    /// Parse a `.eisnn` container already in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Guesser, NnError> {
        let mut r = Reader {
            data: bytes,
            pos: 0,
        };
        if r.take(MAGIC.len())? != MAGIC {
            return Err(NnError::BadMagic);
        }

        let mut metadata = HashMap::new();
        for _ in 0..r.u32()? {
            let key = r.string()?;
            metadata.insert(key, r.string()?);
        }

        let mut tensors = HashMap::new();
        for _ in 0..r.u32()? {
            let name = r.string()?;
            let dtype = r.u8()?;
            let ndim = r.u8()? as usize;
            let dims: Vec<usize> = (0..ndim)
                .map(|_| r.u32().map(|d| d as usize))
                .collect::<Result<_, _>>()?;
            let count = dims.iter().product::<usize>().max(1);
            tensors.insert(
                name,
                Tensor {
                    dims,
                    data: r.values(dtype, count)?,
                },
            );
        }

        let take_vec = |name: &str| -> Result<Vec<f64>, NnError> {
            tensors
                .get(name)
                .map(|t| t.data.clone())
                .ok_or_else(|| NnError::MissingKey(name.to_string()))
        };

        let target_mean = take_vec("target_mean")?;
        let target_std = take_vec("target_std")?;
        let log_params: Vec<bool> = take_vec("log_params")?.iter().map(|&v| v != 0.0).collect();

        let scaling_raw = tensors
            .get("scaling")
            .ok_or_else(|| NnError::MissingKey("scaling".to_string()))?;
        if scaling_raw.len() != 4 * target_mean.len() {
            return Err(NnError::ShapeMismatch {
                name: "scaling".to_string(),
                expected: 4 * target_mean.len(),
                got: scaling_raw.len(),
            });
        }
        let scaling: Vec<Scaling> = scaling_raw
            .data
            .chunks_exact(4)
            .map(|c| Scaling {
                a: c[0],
                b: c[1],
                c: c[2],
                index: c[3].round() as i32,
            })
            .collect();

        let dilations: Vec<usize> = meta(&metadata, "dilations")?
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<usize>()
                    .map_err(|_| NnError::BadValue("dilations".into()))
            })
            .collect::<Result<_, _>>()?;

        let estimator = meta(&metadata, "estimator")?.to_string();
        if !SUPPORTED_ESTIMATORS.contains(&estimator.as_str()) {
            return Err(NnError::UnsupportedEstimator(estimator));
        }

        Ok(Guesser {
            circuit: meta(&metadata, "circuit")?.to_string(),
            param_names: meta(&metadata, "param_names")?
                .split(',')
                .map(str::to_string)
                .collect(),
            n_grid: parse(&metadata, "n_grid")?,
            channels: parse(&metadata, "channels")?,
            kernel: parse(&metadata, "kernel")?,
            groups: parse(&metadata, "groups")?,
            dilations,
            alpha_range: (
                parse(&metadata, "alpha_min")?,
                parse(&metadata, "alpha_max")?,
            ),
            target_mean,
            target_std,
            log_params,
            scaling,
            tensors,
        })
    }

    /// The circuit string this guesser was trained for.
    pub fn circuit(&self) -> &str {
        &self.circuit
    }

    /// Parameter names, matching `Circuit::param_names()` for `circuit()`.
    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    fn tensor(&self, name: &str) -> Result<&Tensor, NnError> {
        self.tensors
            .get(name)
            .ok_or_else(|| NnError::MissingKey(name.to_string()))
    }

    /// Starting parameters for `circuit()`, in `param_names()` order.
    /// `frequencies` are in Hz, may be unsorted.
    pub fn guess(
        &self,
        frequencies: &[f64],
        impedances: &[Complex64],
    ) -> Result<Vec<f64>, NnError> {
        if frequencies.len() < 2 || impedances.len() < 2 {
            return Err(NnError::TooFewPoints(
                frequencies.len().min(impedances.len()),
            ));
        }

        let (k, w_c) = self.scales(frequencies, impedances);
        let (grid, scalars) = self.features(frequencies, impedances, k, w_c);
        let mu = self.forward(&grid, &scalars)?;

        let mut params = vec![0.0; self.param_names.len()];
        for (i, &m) in mu.iter().enumerate() {
            let target = m * self.target_std[i] + self.target_mean[i];
            params[i] = if self.log_params[i] {
                10f64.powf(target)
            } else {
                target
            };
        }
        for (i, log_scaled) in self.log_params.iter().enumerate() {
            if !log_scaled {
                params[i] = params[i].clamp(self.alpha_range.0, self.alpha_range.1);
            }
        }

        // scale factors read alpha from the normalised vector, where it is invariant
        let normalised = params.clone();
        for (i, s) in self.scaling.iter().enumerate() {
            let exponent = s.b
                + if s.index >= 0 {
                    s.c * normalised[s.index as usize]
                } else {
                    0.0
                };
            params[i] = normalised[i] * k.powf(s.a) * w_c.powf(exponent);
        }
        Ok(params)
    }

    /// Impedance and frequency scales, weighting by `max(-sin(phase), 0)` so points in
    /// a featureless part of the sweep carry no weight.
    fn scales(&self, frequencies: &[f64], impedances: &[Complex64]) -> (f64, f64) {
        let mut total = 0.0;
        let mut log_k = 0.0;
        let mut log_w = 0.0;
        for (&f, z) in frequencies.iter().zip(impedances) {
            let magnitude = z.norm();
            if magnitude <= 0.0 || f <= 0.0 {
                continue;
            }
            let u = (-z.im / magnitude).max(0.0);
            total += u;
            log_k += u * magnitude.ln();
            log_w += u * (TAU * f).ln();
        }

        if total < 1e-9 {
            // no reactive response to weight by; fall back to the sweep itself
            let n = frequencies.len() as f64;
            let mean_log_w = frequencies.iter().map(|f| (TAU * f).ln()).sum::<f64>() / n;
            let mut magnitudes: Vec<f64> = impedances.iter().map(|z| z.norm()).collect();
            magnitudes.sort_by(f64::total_cmp);
            return (magnitudes[magnitudes.len() / 2], mean_log_w.exp());
        }
        ((log_k / total).exp(), (log_w / total).exp())
    }

    /// Resample onto `n_grid` log-spaced points across the measured range and
    /// normalise. Channels are log10|Z_hat|, phase/(pi/2) and log10(w_hat)/4.
    fn features(
        &self,
        frequencies: &[f64],
        impedances: &[Complex64],
        k: f64,
        w_c: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut order: Vec<usize> = (0..frequencies.len()).collect();
        order.sort_by(|&a, &b| frequencies[a].total_cmp(&frequencies[b]));

        let log_f: Vec<f64> = order.iter().map(|&i| frequencies[i].log10()).collect();
        let log_mag: Vec<f64> = order
            .iter()
            .map(|&i| impedances[i].norm().log10())
            .collect();

        // unwrapped phase, matching numpy.unwrap
        let mut phase = Vec::with_capacity(order.len());
        let mut previous = impedances[order[0]].arg();
        phase.push(previous);
        for &i in &order[1..] {
            let raw = impedances[i].arg();
            let mut delta = raw - previous;
            delta -= TAU * ((delta + PI) / TAU).floor();
            if delta == -PI && raw > previous {
                delta = PI;
            }
            previous += delta;
            phase.push(previous);
        }

        let (lo, hi) = (log_f[0], log_f[log_f.len() - 1]);
        let step = if self.n_grid > 1 {
            (hi - lo) / (self.n_grid - 1) as f64
        } else {
            0.0
        };

        let mut grid = vec![0.0; 3 * self.n_grid];
        let log_w_c = (w_c / TAU).log10();
        let log_k = k.log10();
        for g in 0..self.n_grid {
            let x = lo + step * g as f64;
            grid[g] = interpolate(&log_f, &log_mag, x) - log_k;
            grid[self.n_grid + g] = interpolate(&log_f, &phase, x) / (PI / 2.0);
            grid[2 * self.n_grid + g] = (x - log_w_c) / 4.0;
        }

        let scalars = vec![(hi - lo) / 8.0, (frequencies.len() as f64).log10() / 2.0];
        (grid, scalars)
    }

    fn forward(&self, grid: &[f64], scalars: &[f64]) -> Result<Vec<f64>, NnError> {
        let mut h = self.conv1d(grid, 3, "stem", 1)?;

        for (block, &dilation) in self.dilations.iter().enumerate() {
            let p = format!("blocks.{block}.");
            let mut r = self.conv1d(&h, self.channels, &format!("{p}conv1"), dilation)?;
            self.group_norm(&mut r, &format!("{p}norm1"))?;
            gelu(&mut r);
            let mut r2 = self.conv1d(&r, self.channels, &format!("{p}conv2"), dilation)?;
            self.group_norm(&mut r2, &format!("{p}norm2"))?;
            for (a, b) in h.iter_mut().zip(&r2) {
                *a += b;
            }
            gelu(&mut h);
        }

        // mean and max pooling over the length axis, then the scalars
        let mut pooled = Vec::with_capacity(2 * self.channels + scalars.len());
        for c in 0..self.channels {
            let row = &h[c * self.n_grid..(c + 1) * self.n_grid];
            pooled.push(row.iter().sum::<f64>() / self.n_grid as f64);
        }
        for c in 0..self.channels {
            let row = &h[c * self.n_grid..(c + 1) * self.n_grid];
            pooled.push(row.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        }
        pooled.extend_from_slice(scalars);

        let mut x = self.linear(&pooled, "head.0")?;
        gelu(&mut x);
        let mut x = self.linear(&x, "head.2")?;
        gelu(&mut x);
        let out = self.linear(&x, "head.4")?;

        // the second half is log-variance, unused here
        Ok(out[..self.param_names.len()].to_vec())
    }

    /// Same-padded 1-D convolution over `(channels_in, n_grid)`, row-major.
    fn conv1d(
        &self,
        x: &[f64],
        c_in: usize,
        name: &str,
        dilation: usize,
    ) -> Result<Vec<f64>, NnError> {
        let weight = self.tensor(&format!("w.{name}.weight"))?;
        let bias = self.tensor(&format!("w.{name}.bias"))?;
        let c_out = weight.dims[0];
        let k = self.kernel;
        let pad = (dilation * (k - 1) / 2) as isize;
        let n = self.n_grid;

        // Accumulate one (output channel, input channel, tap) at a time so the inner
        // loop walks both slices contiguously and autovectorises. Iterating taps
        // innermost instead costs ~3x here.
        let mut out = vec![0.0; c_out * n];
        for o in 0..c_out {
            out[o * n..(o + 1) * n].fill(bias.data[o]);
        }
        for o in 0..c_out {
            let out_row = &mut out[o * n..(o + 1) * n];
            for i in 0..c_in {
                let x_row = &x[i * n..(i + 1) * n];
                for tap in 0..k {
                    let w = weight.data[(o * c_in + i) * k + tap];
                    if w == 0.0 {
                        continue;
                    }
                    let offset = (tap * dilation) as isize - pad;
                    // clip to where the shifted input still lies inside the row
                    let lo = (-offset).max(0) as usize;
                    let hi = (n as isize - offset).min(n as isize).max(0) as usize;
                    for t in lo..hi {
                        out_row[t] += w * x_row[(t as isize + offset) as usize];
                    }
                }
            }
        }
        Ok(out)
    }

    fn group_norm(&self, x: &mut [f64], name: &str) -> Result<(), NnError> {
        let weight = self.tensor(&format!("w.{name}.weight"))?;
        let bias = self.tensor(&format!("w.{name}.bias"))?;
        let per_group = self.channels / self.groups;
        let span = per_group * self.n_grid;

        for g in 0..self.groups {
            let slice = &mut x[g * span..(g + 1) * span];
            let mean = slice.iter().sum::<f64>() / span as f64;
            let var = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / span as f64;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for (j, v) in slice.iter_mut().enumerate() {
                let channel = g * per_group + j / self.n_grid;
                *v = (*v - mean) * inv * weight.data[channel] + bias.data[channel];
            }
        }
        Ok(())
    }

    fn linear(&self, x: &[f64], name: &str) -> Result<Vec<f64>, NnError> {
        let weight = self.tensor(&format!("w.{name}.weight"))?;
        let bias = self.tensor(&format!("w.{name}.bias"))?;
        let (rows, cols) = (weight.dims[0], weight.dims[1]);
        if cols != x.len() {
            return Err(NnError::ShapeMismatch {
                name: name.to_string(),
                expected: cols,
                got: x.len(),
            });
        }
        Ok((0..rows)
            .map(|r| {
                bias.data[r]
                    + (0..cols)
                        .map(|c| weight.data[r * cols + c] * x[c])
                        .sum::<f64>()
            })
            .collect())
    }
}

/// Linear interpolation of `y` against ascending `xs`, clamped at both ends.
fn interpolate(xs: &[f64], y: &[f64], x: f64) -> f64 {
    if x <= xs[0] {
        return y[0];
    }
    if x >= xs[xs.len() - 1] {
        return y[y.len() - 1];
    }
    let i = xs.partition_point(|&v| v <= x).max(1) - 1;
    let (x0, x1) = (xs[i], xs[i + 1]);
    if x1 == x0 {
        return y[i];
    }
    y[i] + (y[i + 1] - y[i]) * (x - x0) / (x1 - x0)
}

/// GELU, tanh approximation -- matches `nn.GELU(approximate="tanh")`.
fn gelu(x: &mut [f64]) {
    const C: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + (C * (*v + 0.044715 * *v * *v * *v)).tanh());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolate_clamps_outside_the_range() {
        let xs = [0.0, 1.0, 2.0];
        let y = [10.0, 20.0, 30.0];
        assert_eq!(interpolate(&xs, &y, -1.0), 10.0);
        assert_eq!(interpolate(&xs, &y, 3.0), 30.0);
        assert_eq!(interpolate(&xs, &y, 0.5), 15.0);
        assert_eq!(interpolate(&xs, &y, 1.5), 25.0);
    }

    #[test]
    fn gelu_matches_known_values() {
        let mut x = [-1.0, 0.0, 1.0, 2.0];
        gelu(&mut x);
        assert!((x[0] - -0.158_808).abs() < 1e-5);
        assert_eq!(x[1], 0.0);
        assert!((x[2] - 0.841_192).abs() < 1e-5);
        assert!((x[3] - 1.954_598).abs() < 1e-5);
    }

    #[test]
    fn f16_decodes_normals_subnormals_and_specials() {
        assert_eq!(f16_to_f64(0x0000), 0.0);
        assert_eq!(f16_to_f64(0x8000), 0.0); // negative zero
        assert_eq!(f16_to_f64(0x3c00), 1.0);
        assert_eq!(f16_to_f64(0xbc00), -1.0);
        assert_eq!(f16_to_f64(0x4000), 2.0);
        assert_eq!(f16_to_f64(0x3555), 0.333_251_953_125); // nearest f16 to 1/3

        // largest normal and smallest positive subnormal
        assert_eq!(f16_to_f64(0x7bff), 65504.0);
        assert!((f16_to_f64(0x0001) - 5.960_464_477_539_063e-8).abs() < 1e-20);
        // largest subnormal is just under the smallest normal
        assert!(f16_to_f64(0x03ff) < f16_to_f64(0x0400));

        assert_eq!(f16_to_f64(0x7c00), f64::INFINITY);
        assert_eq!(f16_to_f64(0xfc00), f64::NEG_INFINITY);
        assert!(f16_to_f64(0x7e00).is_nan());
    }

    #[test]
    fn rejects_a_file_that_is_not_eisnn() {
        assert!(matches!(
            Guesser::from_bytes(b"not a model"),
            Err(NnError::BadMagic)
        ));
    }
}
