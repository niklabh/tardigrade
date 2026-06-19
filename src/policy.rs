//! Backoff policies: pure, allocation-free state machines.
//!
//! A [`BackoffPolicy`] answers exactly one question: *"given that the last
//! attempt failed, how long should I wait before the next one (if at all)?"*
//! It never sleeps, never reads a clock on its own, and never performs the
//! operation. That separation is what makes it portable across sync, async,
//! WASM, and deterministic consensus executors.
//!
//! Policies compose. Start with a base generator like [`ExponentialBackoff`] or
//! [`Constant`], then layer caps on top using [`PolicyExt`]:
//!
//! ```
//! use core::time::Duration;
//! use tardigrade::{ExponentialBackoff, PolicyExt};
//!
//! let policy = ExponentialBackoff::new(Duration::from_millis(100), 2.0)
//!     .with_max_delay(Duration::from_secs(10)) // never wait longer than 10s
//!     .max_attempts(5);                         // give up after 5 retries
//! # let _ = policy;
//! ```

use core::time::Duration;

use crate::clock::Clock;
use crate::jitter::{Jitter, NoJitter};
use crate::util::saturating_mul_f64;

/// The core retry-timing state machine.
///
/// Implementations are *mutable* state machines driven entirely by
/// [`next_delay`] and rewound by [`reset`]. They contain no I/O.
///
/// [`next_delay`]: BackoffPolicy::next_delay
/// [`reset`]: BackoffPolicy::reset
pub trait BackoffPolicy {
    /// Advance the state machine and return the delay to wait before the next
    /// attempt.
    ///
    /// Returning [`None`] means "stop retrying" — the policy is exhausted
    /// (e.g. attempt cap reached or elapsed-time budget exceeded).
    fn next_delay(&mut self) -> Option<Duration>;

    /// Rewind the state machine to its initial condition so it can be reused.
    fn reset(&mut self);
}

impl<P: BackoffPolicy + ?Sized> BackoffPolicy for &mut P {
    #[inline]
    fn next_delay(&mut self) -> Option<Duration> {
        (**self).next_delay()
    }

    #[inline]
    fn reset(&mut self) {
        (**self).reset();
    }
}

/// A policy that always returns the same fixed delay, forever.
///
/// Useful as a base for composition (e.g. `Constant + max_attempts`) or for
/// simple fixed-interval polling.
///
/// ```
/// use core::time::Duration;
/// use tardigrade::{BackoffPolicy, Constant};
///
/// let mut p = Constant::new(Duration::from_millis(250));
/// assert_eq!(p.next_delay(), Some(Duration::from_millis(250)));
/// assert_eq!(p.next_delay(), Some(Duration::from_millis(250)));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Constant {
    delay: Duration,
}

impl Constant {
    /// Create a policy that always waits `delay`.
    #[inline]
    #[must_use]
    pub const fn new(delay: Duration) -> Self {
        Self { delay }
    }
}

impl BackoffPolicy for Constant {
    #[inline]
    fn next_delay(&mut self) -> Option<Duration> {
        Some(self.delay)
    }

    #[inline]
    fn reset(&mut self) {}
}

/// Classic exponential backoff with optional injected jitter.
///
/// The delay starts at `initial` and is multiplied by `multiplier` after every
/// call to [`next_delay`](BackoffPolicy::next_delay). On its own it retries
/// forever; pair it with [`MaxAttempts`], [`WithMaxDelay`], or
/// [`MaxElapsedTime`] (via [`PolicyExt`]) to bound it.
///
/// # Overflow safety
///
/// The internal multiplication is *saturating*: no matter how large the
/// interval or multiplier grows, it can never panic — it simply pins at
/// [`Duration::MAX`]. This is essential in `no_std`/consensus contexts where a
/// panic is unacceptable.
///
/// # Jitter
///
/// The type parameter `J` selects the randomness source and defaults to
/// [`NoJitter`] (fully deterministic). Use [`with_jitter`] to opt into spread.
///
/// [`with_jitter`]: ExponentialBackoff::with_jitter
///
/// ```
/// use core::time::Duration;
/// use tardigrade::{BackoffPolicy, ExponentialBackoff};
///
/// let mut p = ExponentialBackoff::new(Duration::from_millis(100), 2.0);
/// assert_eq!(p.next_delay(), Some(Duration::from_millis(100)));
/// assert_eq!(p.next_delay(), Some(Duration::from_millis(200)));
/// assert_eq!(p.next_delay(), Some(Duration::from_millis(400)));
/// p.reset();
/// assert_eq!(p.next_delay(), Some(Duration::from_millis(100)));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ExponentialBackoff<J = NoJitter> {
    initial: Duration,
    current: Duration,
    multiplier: f64,
    randomization_factor: f64,
    jitter: J,
}

impl ExponentialBackoff<NoJitter> {
    /// Create an exponential backoff starting at `initial`, scaled by
    /// `multiplier` after each attempt, with no jitter.
    ///
    /// A `multiplier` of `<= 0.0` or non-finite is treated as "no growth".
    #[inline]
    #[must_use]
    pub const fn new(initial: Duration, multiplier: f64) -> Self {
        Self {
            initial,
            current: initial,
            multiplier,
            randomization_factor: 0.0,
            jitter: NoJitter,
        }
    }
}

impl<J: Jitter> ExponentialBackoff<J> {
    /// Replace the jitter source and set the symmetric randomization factor.
    ///
    /// `randomization_factor` is clamped to `[0.0, 1.0]`; a value of `0.3`
    /// means each delay is spread uniformly within ±30% of its nominal value.
    ///
    /// ```
    /// use core::time::Duration;
    /// use tardigrade::{BackoffPolicy, ExponentialBackoff, SplitMix64};
    ///
    /// let mut p = ExponentialBackoff::new(Duration::from_millis(1000), 2.0)
    ///     .with_jitter(SplitMix64::new(42), 0.5);
    ///
    /// // First nominal delay is 1000ms; jitter keeps it within [500ms, 1500ms).
    /// let d = p.next_delay().unwrap();
    /// assert!(d >= Duration::from_millis(500) && d < Duration::from_millis(1500));
    /// ```
    #[inline]
    #[must_use]
    pub fn with_jitter<J2: Jitter>(
        self,
        jitter: J2,
        randomization_factor: f64,
    ) -> ExponentialBackoff<J2> {
        ExponentialBackoff {
            initial: self.initial,
            current: self.current,
            multiplier: self.multiplier,
            randomization_factor: randomization_factor.clamp(0.0, 1.0),
            jitter,
        }
    }
}

impl<J: Jitter> BackoffPolicy for ExponentialBackoff<J> {
    fn next_delay(&mut self) -> Option<Duration> {
        // Jitter is applied to the *current* nominal interval...
        let delay = self.jitter.apply(self.current, self.randomization_factor);
        // ...then the nominal interval grows for next time (saturating).
        self.current = saturating_mul_f64(self.current, self.multiplier);
        Some(delay)
    }

    #[inline]
    fn reset(&mut self) {
        self.current = self.initial;
    }
}

impl Default for ExponentialBackoff<NoJitter> {
    /// Sensible defaults: 500ms initial interval, 1.5x growth, no jitter.
    #[inline]
    fn default() -> Self {
        Self::new(Duration::from_millis(500), 1.5)
    }
}

/// Caps the total number of delays an inner policy may produce.
///
/// After `max` delays have been handed out, [`next_delay`] returns [`None`],
/// signalling the executor to stop retrying. Construct via
/// [`PolicyExt::max_attempts`].
///
/// [`next_delay`]: BackoffPolicy::next_delay
#[derive(Debug, Clone, Copy)]
pub struct MaxAttempts<P> {
    inner: P,
    max: u32,
    count: u32,
}

impl<P> MaxAttempts<P> {
    /// Wrap `inner`, allowing at most `max` retries.
    #[inline]
    #[must_use]
    pub const fn new(inner: P, max: u32) -> Self {
        Self {
            inner,
            max,
            count: 0,
        }
    }

    /// Borrow the wrapped policy.
    #[inline]
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Consume the wrapper and return the inner policy.
    #[inline]
    pub fn into_inner(self) -> P {
        self.inner
    }
}

impl<P: BackoffPolicy> BackoffPolicy for MaxAttempts<P> {
    #[inline]
    fn next_delay(&mut self) -> Option<Duration> {
        if self.count >= self.max {
            return None;
        }
        let delay = self.inner.next_delay()?;
        self.count += 1;
        Some(delay)
    }

    #[inline]
    fn reset(&mut self) {
        self.count = 0;
        self.inner.reset();
    }
}

/// Caps the maximum delay returned by an inner policy.
///
/// Any delay the inner policy produces is clamped down to `max_delay`. This is
/// the idiomatic way to bound an otherwise unbounded [`ExponentialBackoff`].
/// Construct via [`PolicyExt::with_max_delay`].
#[derive(Debug, Clone, Copy)]
pub struct WithMaxDelay<P> {
    inner: P,
    max_delay: Duration,
}

impl<P> WithMaxDelay<P> {
    /// Wrap `inner`, clamping every delay to at most `max_delay`.
    #[inline]
    #[must_use]
    pub const fn new(inner: P, max_delay: Duration) -> Self {
        Self { inner, max_delay }
    }

    /// Borrow the wrapped policy.
    #[inline]
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Consume the wrapper and return the inner policy.
    #[inline]
    pub fn into_inner(self) -> P {
        self.inner
    }
}

impl<P: BackoffPolicy> BackoffPolicy for WithMaxDelay<P> {
    #[inline]
    fn next_delay(&mut self) -> Option<Duration> {
        self.inner.next_delay().map(|d| d.min(self.max_delay))
    }

    #[inline]
    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Stops retrying once a wall-clock budget has elapsed.
///
/// On the first call to [`next_delay`], the wrapper snapshots the current
/// instant from the injected [`Clock`]. Once `max_elapsed` has passed since
/// that snapshot, it returns [`None`]. Construct via
/// [`PolicyExt::max_elapsed_time`].
///
/// Because the time source is injected, this works identically against a real
/// system clock and a fully virtual test clock.
///
/// [`next_delay`]: BackoffPolicy::next_delay
#[derive(Debug, Clone, Copy)]
pub struct MaxElapsedTime<P, C: Clock> {
    inner: P,
    clock: C,
    max_elapsed: Duration,
    started_at: Option<C::Instant>,
}

impl<P, C: Clock> MaxElapsedTime<P, C> {
    /// Wrap `inner`, giving up after `max_elapsed` measured by `clock`.
    #[inline]
    pub const fn new(inner: P, clock: C, max_elapsed: Duration) -> Self {
        Self {
            inner,
            clock,
            max_elapsed,
            started_at: None,
        }
    }

    /// Borrow the wrapped policy.
    #[inline]
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Consume the wrapper and return the inner policy.
    #[inline]
    pub fn into_inner(self) -> P {
        self.inner
    }
}

impl<P: BackoffPolicy, C: Clock> BackoffPolicy for MaxElapsedTime<P, C> {
    fn next_delay(&mut self) -> Option<Duration> {
        let now = self.clock.now();
        let started_at = *self.started_at.get_or_insert(now);
        if self.clock.duration_since(started_at, now) >= self.max_elapsed {
            return None;
        }
        self.inner.next_delay()
    }

    #[inline]
    fn reset(&mut self) {
        self.started_at = None;
        self.inner.reset();
    }
}

/// Ergonomic, zero-cost combinators for any [`BackoffPolicy`].
///
/// Blanket-implemented for every policy, so you can fluently layer caps:
///
/// ```
/// use core::time::Duration;
/// use tardigrade::{Constant, PolicyExt};
///
/// let policy = Constant::new(Duration::from_millis(100))
///     .with_max_delay(Duration::from_secs(1))
///     .max_attempts(3);
/// # let _ = policy;
/// ```
pub trait PolicyExt: BackoffPolicy + Sized {
    /// Cap the number of retries to `max`. See [`MaxAttempts`].
    #[inline]
    #[must_use]
    fn max_attempts(self, max: u32) -> MaxAttempts<Self> {
        MaxAttempts::new(self, max)
    }

    /// Clamp every produced delay to at most `max_delay`. See [`WithMaxDelay`].
    #[inline]
    #[must_use]
    fn with_max_delay(self, max_delay: Duration) -> WithMaxDelay<Self> {
        WithMaxDelay::new(self, max_delay)
    }

    /// Stop retrying after `max_elapsed` measured by `clock`. See
    /// [`MaxElapsedTime`].
    #[inline]
    #[must_use]
    fn max_elapsed_time<C: Clock>(
        self,
        clock: C,
        max_elapsed: Duration,
    ) -> MaxElapsedTime<Self, C> {
        MaxElapsedTime::new(self, clock, max_elapsed)
    }
}

impl<P: BackoffPolicy> PolicyExt for P {}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

    #[test]
    fn exponential_grows_and_resets() {
        let mut p = ExponentialBackoff::new(Duration::from_millis(100), 2.0);
        assert_eq!(p.next_delay(), Some(Duration::from_millis(100)));
        assert_eq!(p.next_delay(), Some(Duration::from_millis(200)));
        assert_eq!(p.next_delay(), Some(Duration::from_millis(400)));
        p.reset();
        assert_eq!(p.next_delay(), Some(Duration::from_millis(100)));
    }

    #[test]
    fn exponential_never_panics_on_overflow() {
        let mut p = ExponentialBackoff::new(Duration::from_secs(u64::MAX / 2), 4.0);
        for _ in 0..1000 {
            let _ = p.next_delay();
        }
        assert_eq!(p.next_delay(), Some(Duration::MAX));
    }

    #[test]
    fn max_attempts_stops() {
        let mut p = Constant::new(Duration::from_millis(10)).max_attempts(3);
        assert!(p.next_delay().is_some());
        assert!(p.next_delay().is_some());
        assert!(p.next_delay().is_some());
        assert_eq!(p.next_delay(), None);
        p.reset();
        assert!(p.next_delay().is_some());
    }

    #[test]
    fn with_max_delay_clamps() {
        let mut p = ExponentialBackoff::new(Duration::from_millis(100), 10.0)
            .with_max_delay(Duration::from_millis(500));
        assert_eq!(p.next_delay(), Some(Duration::from_millis(100)));
        assert_eq!(p.next_delay(), Some(Duration::from_millis(500))); // would be 1000
        assert_eq!(p.next_delay(), Some(Duration::from_millis(500)));
    }

    struct VirtualClock {
        now_ms: Cell<u64>,
    }
    impl Clock for VirtualClock {
        type Instant = u64;
        fn now(&self) -> u64 {
            self.now_ms.get()
        }
        fn duration_since(&self, earlier: u64, now: u64) -> Duration {
            Duration::from_millis(now.saturating_sub(earlier))
        }
    }

    #[test]
    fn max_elapsed_time_stops() {
        let clock = VirtualClock { now_ms: Cell::new(0) };
        let mut p = Constant::new(Duration::from_millis(100))
            .max_elapsed_time(&clock, Duration::from_millis(250));

        assert!(p.next_delay().is_some()); // t=0
        clock.now_ms.set(100);
        assert!(p.next_delay().is_some()); // t=100
        clock.now_ms.set(250);
        assert_eq!(p.next_delay(), None); // budget exhausted
    }
}
