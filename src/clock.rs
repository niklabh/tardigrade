//! Time abstraction.
//!
//! `tardigrade` never reads the wall clock itself. Instead, any policy that
//! needs a notion of "how much time has elapsed" (e.g. [`MaxElapsedTime`]) asks
//! the host for it through a [`Clock`]. This is what makes the crate usable
//! inside a WebAssembly engine, a deterministic blockchain state machine, or a
//! bare-metal `no_std` target that has no `std::time::Instant`.
//!
//! [`MaxElapsedTime`]: crate::MaxElapsedTime

use core::time::Duration;

/// A source of monotonic time supplied by the host environment.
///
/// Implementors decide what an "instant" is. It can be a real
/// `std::time::Instant`, a `u64` of milliseconds from a virtual clock, a block
/// height, or a logical tick counter. The only requirement is that
/// [`duration_since`] returns the elapsed [`Duration`] between two instants
/// produced by [`now`].
///
/// [`now`]: Clock::now
/// [`duration_since`]: Clock::duration_since
///
/// # Example: a deterministic virtual clock
///
/// ```
/// use core::cell::Cell;
/// use core::time::Duration;
/// use tardigrade::Clock;
///
/// /// A clock the test fully controls, measured in milliseconds.
/// struct VirtualClock {
///     now_ms: Cell<u64>,
/// }
///
/// impl VirtualClock {
///     fn advance(&self, by: Duration) {
///         self.now_ms.set(self.now_ms.get() + by.as_millis() as u64);
///     }
/// }
///
/// impl Clock for VirtualClock {
///     type Instant = u64;
///
///     fn now(&self) -> u64 {
///         self.now_ms.get()
///     }
///
///     fn duration_since(&self, earlier: u64, now: u64) -> Duration {
///         Duration::from_millis(now.saturating_sub(earlier))
///     }
/// }
///
/// let clock = VirtualClock { now_ms: Cell::new(0) };
/// let start = clock.now();
/// clock.advance(Duration::from_millis(250));
/// assert_eq!(clock.duration_since(start, clock.now()), Duration::from_millis(250));
/// ```
pub trait Clock {
    /// The host's representation of a point in time.
    ///
    /// Must be [`Copy`] so policies can cheaply snapshot a start instant.
    type Instant: Copy;

    /// Return the current instant.
    fn now(&self) -> Self::Instant;

    /// Return the elapsed time from `earlier` to `now`.
    ///
    /// Implementations should saturate (never panic) if `now` precedes
    /// `earlier`, returning [`Duration::ZERO`] in that case.
    fn duration_since(&self, earlier: Self::Instant, now: Self::Instant) -> Duration;
}

impl<C: Clock + ?Sized> Clock for &C {
    type Instant = C::Instant;

    #[inline]
    fn now(&self) -> Self::Instant {
        (**self).now()
    }

    #[inline]
    fn duration_since(&self, earlier: Self::Instant, now: Self::Instant) -> Duration {
        (**self).duration_since(earlier, now)
    }
}

/// A monotonic [`Clock`] backed by `std::time::Instant`.
///
/// Only available with the `std` feature. The default build is `no_std` and
/// expects you to bring your own clock.
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl Clock for SystemClock {
    type Instant = std::time::Instant;

    #[inline]
    fn now(&self) -> Self::Instant {
        std::time::Instant::now()
    }

    #[inline]
    fn duration_since(&self, earlier: Self::Instant, now: Self::Instant) -> Duration {
        now.saturating_duration_since(earlier)
    }
}
