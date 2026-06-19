# tardigrade

> A generic, execution-agnostic and time-agnostic backoff & retry utility — a pure state machine for `no_std`, zero-allocation environments.

Tardigrades (water bears) survive vacuum, radiation, and deep freeze by going
dormant and waiting it out. This crate does the digital equivalent: it computes
*when* to retry, and nothing else.

## Why?

Most retry crates bake in `std::time::Instant` and `std::thread::sleep`, welding
*policy* (how long to wait) to *execution* (how to wait). That breaks down in:

- **WebAssembly engines** executing Wasm directly, with no host threads;
- **Deterministic blockchain / consensus** state machines, where reading the
  wall clock or panicking is a fault;
- **`no_std` embedded** targets with no allocator and no `std::time`.

`tardigrade` is a pure state machine instead:

- `#![no_std]`, **zero allocation**, no `dyn` (unless you opt in via `alloc`).
- Time is just `core::time::Duration`; instants come from *your* `Clock`.
- Randomness is injected via `Jitter` (deterministic by default) — great for
  reproducible P2P tests and avoiding thundering herds.
- Execution is driven by *your* `sleep` — a sync closure or an async future,
  with **no runtime lock-in** (no Tokio dependency).
- `#![forbid(unsafe_code)]`.

## Quick start

```rust
use core::ops::ControlFlow;
use core::time::Duration;
use tardigrade::{retry_sync, ExponentialBackoff, PolicyExt, RetryError};

// Compose pure state-machine policies.
let policy = ExponentialBackoff::new(Duration::from_millis(50), 2.0)
    .with_max_delay(Duration::from_secs(5))
    .max_attempts(4);

// Drive it with your own operation + sleep.
let mut tries = 0u32;
let result: Result<&str, RetryError<&str>> = retry_sync(
    policy,
    || {
        tries += 1;
        if tries >= 3 { ControlFlow::Break("connected") }
        else { ControlFlow::Continue("connection refused") }
    },
    |delay| { /* std::thread::sleep(delay) — or advance a virtual clock */ },
);

assert_eq!(result, Ok("connected"));
```

### The `ControlFlow` contract

Your operation returns `core::ops::ControlFlow<B, C>`:

- `Break(value)` — **terminal**. The loop stops and returns `Ok(value)`. Encode
  both success *and* fatal errors here (e.g. `Break(Result<T, E>)`).
- `Continue(state)` — **transient**. Retry after a backoff delay. If the policy
  gives up, the last `state` is returned in `RetryError::Exhausted`.

### Async, runtime-free

```rust,ignore
retry_async(
    policy,
    || async { /* ... */ ControlFlow::Continue("transient") },
    |delay| tokio::time::sleep(delay), // or embassy_time::Timer, or a WASM timer
).await
```

`retry_async` is a plain `async fn` over `core::future::Future`; it never names a
specific executor.

## Building blocks

| Item | Role |
|------|------|
| `BackoffPolicy` | trait: `next_delay() -> Option<Duration>` + `reset()` |
| `ExponentialBackoff<J>` | exponential growth, saturating (never panics) |
| `Constant` | fixed-interval delay |
| `MaxAttempts<P>` | caps the number of retries |
| `WithMaxDelay<P>` | clamps the maximum delay |
| `MaxElapsedTime<P, C>` | gives up after a time budget (uses a `Clock`) |
| `PolicyExt` | fluent combinators: `.max_attempts(..)`, `.with_max_delay(..)`, `.max_elapsed_time(..)` |
| `Clock` | host-provided time source |
| `Jitter` / `NoJitter` / `SplitMix64` | injected randomness |
| `retry_sync` / `retry_async` | execution drivers |

## Feature flags

| feature | default | adds |
|---------|---------|------|
| `alloc` | off | `BoxedPolicy` + `impl BackoffPolicy for Box<dyn …>` |
| `std`   | off | `SystemClock` backed by `std::time::Instant` |

The default build pulls in neither: pure `core`, allocation-free.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
