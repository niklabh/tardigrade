//! Internal numeric helpers.
//!
//! These exist mainly to make `f64`-to-[`Duration`] conversions *total*: they
//! can never panic, regardless of overflow, `NaN`, or infinity. This matters a
//! great deal in deterministic state machines (consensus, P2P) where a panic is
//! a consensus fault, and in `no_std` targets where unwinding may be disabled.

use core::time::Duration;

/// Multiply a [`Duration`] by an `f64` factor, saturating instead of panicking.
///
/// [`Duration::from_secs_f64`] panics on negative, non-finite, or overflowing
/// inputs. This helper is the panic-free equivalent:
///
/// * `factor <= 0.0`, `NaN`, or a zero `base` yields [`Duration::ZERO`].
/// * Any result that would overflow a `Duration` saturates to [`Duration::MAX`].
#[inline]
#[must_use]
pub(crate) fn saturating_mul_f64(base: Duration, factor: f64) -> Duration {
    // NaN and non-positive factors are meaningless for a delay: collapse to zero.
    if factor.is_nan() || factor <= 0.0 || base.is_zero() {
        return Duration::ZERO;
    }
    // A positive-infinite factor saturates to the maximum representable delay.
    if factor.is_infinite() {
        return Duration::MAX;
    }

    let secs = base.as_secs_f64() * factor;
    if secs <= 0.0 {
        return Duration::ZERO;
    }
    if !secs.is_finite() {
        return Duration::MAX;
    }

    // `try_from_secs_f64` returns `Err` on overflow rather than panicking,
    // which is exactly the behaviour we want for an untrusted multiplier.
    match Duration::try_from_secs_f64(secs) {
        Ok(d) => d,
        Err(_) => Duration::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_pathological_factors() {
        let base = Duration::from_secs(1);
        assert_eq!(saturating_mul_f64(base, f64::NAN), Duration::ZERO);
        assert_eq!(saturating_mul_f64(base, f64::INFINITY), Duration::MAX);
        assert_eq!(saturating_mul_f64(base, -1.0), Duration::ZERO);
        assert_eq!(saturating_mul_f64(base, 0.0), Duration::ZERO);
        assert_eq!(saturating_mul_f64(Duration::ZERO, 10.0), Duration::ZERO);
    }

    #[test]
    fn saturates_at_max_instead_of_panicking() {
        // Repeatedly growing the max duration must never panic.
        let grown = saturating_mul_f64(Duration::MAX, 2.0);
        assert_eq!(grown, Duration::MAX);
    }

    #[test]
    fn ordinary_multiplication_is_exact_enough() {
        let d = saturating_mul_f64(Duration::from_millis(100), 1.5);
        assert_eq!(d, Duration::from_millis(150));
    }
}
