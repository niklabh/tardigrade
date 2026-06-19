//! Execution drivers.
//!
//! These functions glue a [`BackoffPolicy`] to *your* notion of "do the work"
//! and "wait". They are deliberately tiny and own no I/O of their own:
//!
//! * [`retry_sync`] takes a blocking `sleep` closure.
//! * [`retry_async`] takes futures for both the operation and the sleep, and is
//!   runtime-agnostic — it never mentions Tokio, async-std, or `embassy`.
//!
//! # The `ControlFlow` contract
//!
//! Your operation returns a [`ControlFlow<B, C>`]:
//!
//! * [`ControlFlow::Break`]`(value)` — **terminal**. The retry loop stops and
//!   hands `value` back as `Ok(value)`. Encode *both* success and fatal errors
//!   here, e.g. by returning `ControlFlow::Break(Result<T, E>)`.
//! * [`ControlFlow::Continue`]`(state)` — **transient**. The operation should be
//!   retried after a backoff delay. The `state` you carry is preserved and, if
//!   the policy gives up, returned inside [`RetryError::Exhausted`] so you can
//!   report the last transient failure.

use core::ops::ControlFlow;

use crate::policy::BackoffPolicy;

/// The reason a retry loop stopped without a terminal [`ControlFlow::Break`].
///
/// The generic parameter `C` is the [`ControlFlow::Continue`] payload type, so
/// the caller always gets back the *last* transient state observed before the
/// policy gave up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryError<C> {
    /// The [`BackoffPolicy`] returned [`None`]: it ran out of attempts, ran out
    /// of its elapsed-time budget, or was otherwise exhausted. Carries the last
    /// transient state from [`ControlFlow::Continue`].
    Exhausted(C),
}

impl<C> RetryError<C> {
    /// Return a reference to the last transient state.
    #[inline]
    pub fn last_state(&self) -> &C {
        match self {
            RetryError::Exhausted(c) => c,
        }
    }

    /// Consume the error and return the last transient state.
    #[inline]
    pub fn into_last_state(self) -> C {
        match self {
            RetryError::Exhausted(c) => c,
        }
    }
}

impl<C: core::fmt::Display> core::fmt::Display for RetryError<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RetryError::Exhausted(c) => write!(f, "retry policy exhausted (last state: {c})"),
        }
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<C: core::fmt::Debug + core::fmt::Display> std::error::Error for RetryError<C> {}

/// Drive an operation to completion synchronously, backing off between tries.
///
/// The loop:
/// 1. runs `operation`;
/// 2. on [`ControlFlow::Break`]`(b)` returns `Ok(b)`;
/// 3. on [`ControlFlow::Continue`]`(c)` asks `policy` for the next delay;
/// 4. if there is one, calls `sleep(delay)` and loops; otherwise returns
///    `Err(`[`RetryError::Exhausted`]`(c))`.
///
/// `sleep` is *yours* — it can call `std::thread::sleep`, busy-spin a virtual
/// clock in a test, or do nothing at all.
///
/// # Example: dummy clock + no-op sleep
///
/// ```
/// use core::ops::ControlFlow;
/// use core::time::Duration;
/// use tardigrade::{retry_sync, ExponentialBackoff, PolicyExt, RetryError};
///
/// let policy = ExponentialBackoff::default().max_attempts(5);
///
/// let mut attempts = 0u32;
/// let mut virtual_clock = Duration::ZERO; // our "host" time, advanced by sleep
///
/// let result: Result<u32, RetryError<&'static str>> = retry_sync(
///     policy,
///     || {
///         attempts += 1;
///         if attempts < 3 {
///             ControlFlow::Continue("service warming up")
///         } else {
///             ControlFlow::Break(attempts) // terminal success
///         }
///     },
///     |delay| virtual_clock += delay, // "sleep" by advancing virtual time
/// );
///
/// assert_eq!(result, Ok(3));
/// assert_eq!(attempts, 3);
/// assert!(virtual_clock > Duration::ZERO);
/// ```
///
/// # Example: giving up
///
/// ```
/// use core::ops::ControlFlow;
/// use core::time::Duration;
/// use tardigrade::{retry_sync, Constant, PolicyExt, RetryError};
///
/// let policy = Constant::new(Duration::from_millis(1)).max_attempts(2);
/// let result: Result<(), RetryError<i32>> = retry_sync(
///     policy,
///     || ControlFlow::Continue(-1), // never succeeds
///     |_d| {},
/// );
/// assert_eq!(result, Err(RetryError::Exhausted(-1)));
/// ```
pub fn retry_sync<P, B, C, Op, Sleep>(
    mut policy: P,
    mut operation: Op,
    mut sleep: Sleep,
) -> Result<B, RetryError<C>>
where
    P: BackoffPolicy,
    Op: FnMut() -> ControlFlow<B, C>,
    Sleep: FnMut(core::time::Duration),
{
    loop {
        match operation() {
            ControlFlow::Break(value) => return Ok(value),
            ControlFlow::Continue(state) => match policy.next_delay() {
                Some(delay) => sleep(delay),
                None => return Err(RetryError::Exhausted(state)),
            },
        }
    }
}

/// Drive an operation to completion asynchronously, backing off between tries.
///
/// Identical in spirit to [`retry_sync`], but `operation` and `sleep` each
/// produce a [`Future`]. This function is a plain `async fn` over `core`'s
/// [`Future`] and is **not** tied to any executor: plug in Tokio's
/// `tokio::time::sleep`, `embassy_time::Timer`, a WASM timer, or a virtual one.
///
/// [`Future`]: core::future::Future
///
/// # Example: runtime-free, deterministic async
///
/// This example uses a trivial hand-rolled executor and an instantly-ready
/// sleep future, proving the function needs no real runtime.
///
/// ```
/// use core::future::Future;
/// use core::ops::ControlFlow;
/// use core::pin::Pin;
/// use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
/// use core::time::Duration;
/// use tardigrade::{retry_async, ExponentialBackoff, PolicyExt, RetryError};
///
/// // A future that is ready immediately (a stand-in for a real timer).
/// async fn ready() {}
///
/// async fn run() -> Result<u32, RetryError<&'static str>> {
///     let policy = ExponentialBackoff::default().max_attempts(5);
///     let mut attempts = 0u32;
///     retry_async(
///         policy,
///         || {
///             attempts += 1;
///             async move {
///                 if attempts < 3 {
///                     ControlFlow::Continue("warming up")
///                 } else {
///                     ControlFlow::Break(attempts)
///                 }
///             }
///         },
///         |_delay| ready(), // your runtime's sleep goes here
///     )
///     .await
/// }
///
/// // Minimal block_on so the doctest needs no async runtime dependency.
/// fn block_on<F: Future>(mut fut: F) -> F::Output {
///     fn noop(_: *const ()) {}
///     fn clone(_: *const ()) -> RawWaker { RawWaker::new(core::ptr::null(), &VTABLE) }
///     static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
///     let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) };
///     let mut cx = Context::from_waker(&waker);
///     // Safety: `fut` is not moved after being pinned.
///     let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
///     loop {
///         if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
///             return v;
///         }
///     }
/// }
///
/// assert_eq!(block_on(run()), Ok(3));
/// ```
pub async fn retry_async<P, B, C, Op, OpFut, Sleep, SleepFut>(
    mut policy: P,
    mut operation: Op,
    mut sleep: Sleep,
) -> Result<B, RetryError<C>>
where
    P: BackoffPolicy,
    Op: FnMut() -> OpFut,
    OpFut: core::future::Future<Output = ControlFlow<B, C>>,
    Sleep: FnMut(core::time::Duration) -> SleepFut,
    SleepFut: core::future::Future<Output = ()>,
{
    loop {
        match operation().await {
            ControlFlow::Break(value) => return Ok(value),
            ControlFlow::Continue(state) => match policy.next_delay() {
                Some(delay) => sleep(delay).await,
                None => return Err(RetryError::Exhausted(state)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Constant, ExponentialBackoff, PolicyExt};
    use core::time::Duration;

    #[test]
    fn sync_succeeds_after_retries() {
        let policy = ExponentialBackoff::new(Duration::from_millis(1), 2.0).max_attempts(10);
        let mut attempts = 0u32;
        let mut slept = Duration::ZERO;

        let out: Result<&str, RetryError<u32>> = retry_sync(
            policy,
            || {
                attempts += 1;
                if attempts < 4 {
                    ControlFlow::Continue(attempts)
                } else {
                    ControlFlow::Break("ok")
                }
            },
            |d| slept += d,
        );

        assert_eq!(out, Ok("ok"));
        assert_eq!(attempts, 4);
        // 3 sleeps of 1ms, 2ms, 4ms.
        assert_eq!(slept, Duration::from_millis(7));
    }

    #[test]
    fn sync_exhausts_and_returns_last_state() {
        let policy = Constant::new(Duration::from_millis(1)).max_attempts(3);
        let mut attempts = 0u32;
        let out: Result<(), RetryError<u32>> = retry_sync(
            policy,
            || {
                attempts += 1;
                ControlFlow::Continue(attempts)
            },
            |_d| {},
        );
        // 1 initial try + 3 retries = 4 operation calls; last state is 4.
        assert_eq!(out, Err(RetryError::Exhausted(4)));
        assert_eq!(attempts, 4);
    }

    #[test]
    fn sync_breaks_immediately_without_sleeping() {
        let policy = Constant::new(Duration::from_secs(99));
        let mut slept = false;
        let out: Result<i32, RetryError<()>> =
            retry_sync(policy, || ControlFlow::Break(7), |_d| slept = true);
        assert_eq!(out, Ok(7));
        assert!(!slept);
    }
}
