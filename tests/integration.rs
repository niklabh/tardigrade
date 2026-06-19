//! Integration tests exercising the public API as an external consumer.
//!
//! These run as a normal `std` test binary, but they only touch the crate's
//! public `core`-based surface — proving the API is ergonomic from the outside.

use core::cell::Cell;
use core::future::Future;
use core::ops::ControlFlow;
use core::pin::Pin;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use core::time::Duration;

use tardigrade::{
    retry_async, retry_sync, BackoffPolicy, Clock, ExponentialBackoff, PolicyExt, RetryError,
    SplitMix64,
};

/// A fully deterministic virtual clock measured in milliseconds.
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
fn elapsed_time_budget_against_virtual_clock() {
    let clock = VirtualClock { now_ms: Cell::new(0) };
    // Sleep advances the same virtual clock the policy reads from.
    let mut policy = ExponentialBackoff::new(Duration::from_millis(100), 2.0)
        .max_elapsed_time(&clock, Duration::from_millis(1_000));

    let mut attempts = 0u32;
    let out: Result<(), RetryError<u32>> = retry_sync(
        &mut policy,
        || {
            attempts += 1;
            ControlFlow::Continue(attempts)
        },
        |delay| clock.now_ms.set(clock.now_ms.get() + delay.as_millis() as u64),
    );

    assert!(matches!(out, Err(RetryError::Exhausted(_))));
    // The loop must have given up once the 1s budget elapsed.
    assert!(clock.now_ms.get() >= 1_000);
}

#[test]
fn jitter_is_reproducible_for_a_fixed_seed() {
    fn collect(seed: u64) -> Vec<Duration> {
        let mut p = ExponentialBackoff::new(Duration::from_millis(1_000), 2.0)
            .with_jitter(SplitMix64::new(seed), 0.5);
        (0..8).map(|_| p.next_delay().unwrap()).collect()
    }
    // Same seed => identical delay sequence (critical for P2P sim tests).
    assert_eq!(collect(0xDEAD_BEEF), collect(0xDEAD_BEEF));
    // Different seed => (almost certainly) different sequence.
    assert_ne!(collect(1), collect(2));
}

// --- Minimal no-runtime executor so we can test `retry_async` deterministically. ---

fn block_on<F: Future>(mut fut: F) -> F::Output {
    fn noop(_: *const ()) {}
    fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(core::ptr::null(), &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);

    let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
    }
}

#[test]
fn async_retry_without_a_real_runtime() {
    let policy = ExponentialBackoff::new(Duration::from_millis(1), 2.0).max_attempts(10);
    let slept = Cell::new(Duration::ZERO);

    let result = block_on(retry_async(
        policy,
        {
            let mut attempts = 0u32;
            move || {
                attempts += 1;
                async move {
                    if attempts < 5 {
                        ControlFlow::Continue("retry me")
                    } else {
                        ControlFlow::Break(attempts)
                    }
                }
            }
        },
        |delay| {
            slept.set(slept.get() + delay);
            async {}
        },
    ));

    assert_eq!(result, Ok(5));
    // 4 sleeps: 1 + 2 + 4 + 8 = 15ms.
    assert_eq!(slept.get(), Duration::from_millis(15));
}
