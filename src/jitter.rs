//! Jitter / randomness abstraction.
//!
//! Adding randomness to backoff delays is the standard defence against the
//! *thundering herd*: if a thousand peers all fail at the same instant and all
//! back off by exactly the same amount, they will all retry at the same instant
//! too. Jitter spreads them out.
//!
//! Crucially, the randomness is **injected**, not pulled from a global RNG.
//! Inject [`NoJitter`] for fully deterministic behaviour, a seeded
//! [`SplitMix64`] for reproducible-yet-spread-out behaviour (ideal for P2P
//! simulation and consensus tests), or your own host RNG in production.

use core::time::Duration;

use crate::util::saturating_mul_f64;

/// A source of uniform randomness used to perturb backoff delays.
///
/// The only required method is [`next_unit_f64`], which yields a value in
/// `[0.0, 1.0)`. The provided [`apply`] method turns that into a symmetric
/// jitter around a base delay.
///
/// [`next_unit_f64`]: Jitter::next_unit_f64
/// [`apply`]: Jitter::apply
pub trait Jitter {
    /// Return the next pseudo-random value in the half-open range `[0.0, 1.0)`.
    fn next_unit_f64(&mut self) -> f64;

    /// Apply symmetric jitter of `factor` around `base`.
    ///
    /// With `factor == 0.0` the `base` is returned unchanged. With
    /// `factor == r` the result is uniformly distributed in
    /// `[base * (1 - r), base * (1 + r))`. `factor` is clamped to `[0.0, 1.0]`.
    ///
    /// The conversion is saturating and therefore panic-free.
    #[inline]
    fn apply(&mut self, base: Duration, factor: f64) -> Duration {
        if factor <= 0.0 || base.is_zero() {
            return base;
        }
        let factor = factor.min(1.0);
        let r = self.next_unit_f64();
        // scale ranges over [1 - factor, 1 + factor).
        let scale = (1.0 - factor) + r * (2.0 * factor);
        saturating_mul_f64(base, scale)
    }
}

impl<J: Jitter + ?Sized> Jitter for &mut J {
    #[inline]
    fn next_unit_f64(&mut self) -> f64 {
        (**self).next_unit_f64()
    }
}

/// A [`Jitter`] that adds no randomness at all.
///
/// [`apply`](Jitter::apply) always returns the base delay unchanged, so a
/// policy using `NoJitter` is fully deterministic. This is the default jitter
/// for [`ExponentialBackoff`].
///
/// [`ExponentialBackoff`]: crate::ExponentialBackoff
#[derive(Debug, Clone, Copy, Default)]
pub struct NoJitter;

impl Jitter for NoJitter {
    /// Returns the midpoint `0.5`, so any symmetric jitter collapses to the
    /// base delay (`(1 - f) + 0.5 * 2f == 1`).
    #[inline]
    fn next_unit_f64(&mut self) -> f64 {
        0.5
    }
}

/// A tiny, fast, deterministic PRNG (Steele et al.'s SplitMix64).
///
/// This is **not** cryptographically secure, but it is excellent for jitter:
/// it is allocation-free, `const`-constructible, has good statistical spread,
/// and is perfectly reproducible from a seed. That reproducibility is the whole
/// point in P2P/consensus testing.
///
/// # Example
///
/// ```
/// use tardigrade::{Jitter, SplitMix64};
///
/// let mut a = SplitMix64::new(42);
/// let mut b = SplitMix64::new(42);
/// // Same seed => identical stream => reproducible tests.
/// assert_eq!(a.next_unit_f64(), b.next_unit_f64());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a new generator from a 64-bit seed.
    #[inline]
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the generator and return the next raw 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl Default for SplitMix64 {
    #[inline]
    fn default() -> Self {
        // An arbitrary non-zero default seed.
        Self::new(0x2545_F491_4F6C_DD1D)
    }
}

impl Jitter for SplitMix64 {
    #[inline]
    fn next_unit_f64(&mut self) -> f64 {
        // Use the top 53 bits to fill an f64 mantissa, giving a uniform
        // value in [0, 1). 2^-53 is exactly representable.
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 * SCALE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_jitter_is_identity() {
        let mut j = NoJitter;
        let base = Duration::from_millis(100);
        assert_eq!(j.apply(base, 0.5), base);
        assert_eq!(j.apply(base, 1.0), base);
    }

    #[test]
    fn unit_values_are_in_range() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..10_000 {
            let v = rng.next_unit_f64();
            assert!((0.0..1.0).contains(&v), "value out of range: {v}");
        }
    }

    #[test]
    fn jitter_stays_within_symmetric_bounds() {
        let mut rng = SplitMix64::new(99);
        let base = Duration::from_millis(1000);
        for _ in 0..10_000 {
            let d = rng.apply(base, 0.5);
            assert!(d >= Duration::from_millis(500));
            assert!(d < Duration::from_millis(1500));
        }
    }

    #[test]
    fn seeded_streams_are_reproducible() {
        let mut a = SplitMix64::new(123);
        let mut b = SplitMix64::new(123);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
