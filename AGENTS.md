# AGENTS.md

`stime` is a small Rust GPU task runtime built on the CUDA driver API via
[`cudarc`](https://crates.io/crates/cudarc) (`driver` + `dynamic-loading`, CUDA 12.9 bindings).

## Commands

```bash
cargo build
cargo test                        # all integration tests; needs a working CUDA GPU
cargo test --test caching_allocator  # one test file
```

- Tests run against the real GPU. There is no mock driver, so a test failure may be a device or
  driver issue. Check `nvidia-smi -L` before debugging code.
- `tests/memory.rs` is by far the slowest test file.
- `clippy` and `rustfmt` are not installed on this toolchain. Install with
  `rustup component add clippy rustfmt` before using them. Until then, match existing formatting
  by hand (default rustfmt style).

## Layout

Each module wraps one CUDA concept. `src/lib.rs` re-exports the whole public API flat
(`stime::Device`, `stime::Task`, ...), so add new public types to the `pub use` list there.

| Path | Contents |
|------|----------|
| `src/device.rs`, `stream.rs`, `event.rs` | Thin wrappers over CUDA device, stream, event |
| `src/kernel.rs` | `Module` (loaded from PTX), `Kernel`, `KernelArg`, `LaunchConfig` |
| `src/task.rs` | `Task<A>`: kernel + launch config + owned args (`KernelArgs`), with `TaskState` |
| `src/worker.rs` | `Worker`: dedicated host thread + bounded queue launching tasks on one stream |
| `src/runtime.rs` | `Runtime`: round-robins tasks over several workers/streams |
| `src/memory/` | `Allocator` trait, `SystemAllocator`, `CachingAllocator`, `MemoryPool`, `DeviceBuffer` |
| `src/error.rs` | Crate `Error` enum and `Result` alias |
| `tests/` | Integration tests, one file per module. Shared helpers in `tests/common/mod.rs` |

## Key invariants

- A `Task` owns its buffers until its kernel finishes. That is why an `Allocator` may reuse a
  freed block immediately; don't break this ownership.
- Host copies and memsets synchronize the default stream so non-blocking worker streams never
  see stale data.
- Tasks reject arguments on a different device than the stream (`Error::DeviceMismatch`).
- `Allocator` is an `unsafe trait`: `alloc` must return exclusively owned, suitably aligned
  memory on `device()`, and `free` gets the exact `(ptr, bytes)` from `alloc`.

## Code conventions

- Every `unsafe` block has a `// SAFETY:` comment, and every `unsafe fn`/`unsafe trait` has a
  `# Safety` doc section stating the caller's contract.
- Every public item has a doc comment. Larger modules open with a `//!` overview, sometimes
  with an ASCII diagram (see `worker.rs`, `runtime.rs`).
- Errors go through `crate::Error`. When adding a variant, update `Display` and `source`. Those
  matches list variants explicitly (no `_` wildcard) so new variants fail to compile until
  handled. Keep it that way.
- Convert driver errors with `?` (there is `From<DriverError> for Error`).
- Imports go at the top of the file, not inline.
- Comments explain constraints the code can't show, not what the next line does.

## Tests

- Kernels are inline PTX strings (`ADD_PTX`, `FILL_PTX` in `tests/common/mod.rs`, targeting
  `sm_52`). Reuse `add_task`, `fill_task` and `config` rather than duplicating setup.
- Allocator logic can be tested without real allocations by wrapping a fake `Allocator`
  (see `FakeAllocator` in `tests/caching_allocator.rs`).
- Add or update tests in the matching `tests/<module>.rs` with any behavior change.

## Commits

Imperative one-line summary (e.g. "Add CachingAllocator and split memory into a module
folder"), then a short body explaining what changed and why.
