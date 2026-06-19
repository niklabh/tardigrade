//! # tardigrade
//!
//! A generic, **execution-agnostic** and **time-agnostic** backoff & retry
//! utility, built as a pure state machine.
//!
//! Tardigrades (water bears) survive vacuum, radiation, and deep freeze by
//! entering a dormant state and waiting it out. This crate does the digital
//! equivalent: it computes *when* to wake up and try again, and nothing else.
//!
//! ## Why another retry crate?
//!
//! Most retry crates (e.g. `backoff`) bake in `std::time::Instant` and
//! `std::thread::sleep`, coupling *policy* (how long to wait) with *execution*
//! (how to wait). That falls apart in:
//!
//! * **WebAssembly engines** executing Wasm directly, with no host threads;
//! * **Deterministic blockchain / consensus** state machines, where wall-clock
//!   reads and panics are forbidden;
//! * **`no_std` embedded** targets with no allocator and no `std::time`.
//!
//! `tardigrade` solves this by being a pure state machine:
//!
//! * `#![no_std]`, zero allocation, no `dyn` (unless you opt in).
//! * Time is just [`core::time::Duration`]; instants come from your [`Clock`].
//! * Randomness/jitter is injected via [`Jitter`] (deterministic by default).
//! * Execution is driven by *your* `sleep` — sync closure or async future.
//!
//! ## 30-second tour
//!
//! ```
//! use core::ops::ControlFlow;
//! use core::time::Duration;
//! use tardigrade::{retry_sync, ExponentialBackoff, PolicyExt, RetryError};
//!
//! // 1. Build a policy by composing pure state machines.
//! let policy = ExponentialBackoff::new(Duration::from_millis(50), 2.0)
//!     .with_max_delay(Duration::from_secs(5))
//!     .max_attempts(4);
//!
//! // 2. Drive it with your own operation + sleep. Here `sleep` just advances
//! //    a virtual clock, so the example runs instantly and deterministically.
//! let mut tries = 0u32;
//! let mut elapsed = Duration::ZERO;
//! let result: Result<&str, RetryError<&str>> = retry_sync(
//!     policy,
//!     || {
//!         tries += 1;
//!         if tries >= 3 { ControlFlow::Break("connected") }
//!         else { ControlFlow::Continue("connection refused") }
//!     },
//!     |delay| elapsed += delay, // plug in your real sleep in production
//! );
//!
//! assert_eq!(result, Ok("connected"));
//! ```
//!
//! ## Feature flags
//!
//! | feature | default | adds |
//! |---------|---------|------|
//! | `alloc` | no      | [`BoxedPolicy`], `impl BackoffPolicy for Box<dyn …>` |
//! | `std`   | no      | [`SystemClock`] backed by `std::time::Instant`        |
//!
//! The default build pulls in neither, keeping it pure-`core` and
//! allocation-free.

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "alloc")]
extern crate alloc;

mod clock;
mod exec;
mod jitter;
mod policy;
mod util;

pub use clock::Clock;
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub use clock::SystemClock;

pub use jitter::{Jitter, NoJitter, SplitMix64};

pub use policy::{
    BackoffPolicy, Constant, ExponentialBackoff, MaxAttempts, MaxElapsedTime, PolicyExt,
    WithMaxDelay,
};

pub use exec::{retry_async, retry_sync, RetryError};

/// Optional dynamic-dispatch support, gated behind the `alloc` feature.
///
/// Static dispatch (generics) is the default and the recommended path — it is
/// what keeps the crate allocation-free. But sometimes you genuinely need to
/// store policies of differing concrete types in one place (e.g. a config-driven
/// table of policies). For those cases only, enable `alloc` and use a
/// [`BoxedPolicy`].
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
mod boxed {
    use crate::policy::BackoffPolicy;
    use alloc::boxed::Box;
    use core::time::Duration;

    /// A heap-allocated, type-erased [`BackoffPolicy`].
    ///
    /// ```
    /// # extern crate alloc;
    /// use core::time::Duration;
    /// use tardigrade::{BackoffPolicy, BoxedPolicy, Constant, ExponentialBackoff, PolicyExt};
    ///
    /// // Two different concrete policy types behind one storable type.
    /// let policies: [BoxedPolicy; 2] = [
    ///     alloc::boxed::Box::new(Constant::new(Duration::from_millis(10))),
    ///     alloc::boxed::Box::new(
    ///         ExponentialBackoff::new(Duration::from_millis(10), 2.0).max_attempts(3),
    ///     ),
    /// ];
    /// for mut p in policies {
    ///     assert!(p.next_delay().is_some());
    /// }
    /// ```
    pub type BoxedPolicy<'a> = Box<dyn BackoffPolicy + 'a>;

    impl<P: BackoffPolicy + ?Sized> BackoffPolicy for Box<P> {
        #[inline]
        fn next_delay(&mut self) -> Option<Duration> {
            (**self).next_delay()
        }

        #[inline]
        fn reset(&mut self) {
            (**self).reset();
        }
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub use boxed::BoxedPolicy;
