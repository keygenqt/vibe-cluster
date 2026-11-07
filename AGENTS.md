# AGENTS.md — vibe-cluster

Multi-provider LLM cluster library for Rust; rules of the crate.

## Project

The crate distributes a stream of prompts across a pool of LLM
connections. It is a public crate: the public API is stable and
changes deliberately.

A provider is a user-facing connection kind, not an implementation.
There are three client implementations: the native Ollama API, the
shared OpenAI-compatible one (OpenAI, DeepSeek, Qwen, Moonshot,
Zhipu, Yandex) and GigaChat with its OAuth wrapper. Adding a provider
does not necessarily mean a new client.

The layering logic: the cluster orchestrates only through the `Client`
trait, concrete clients are created by the factory, data types are
kept apart from the machinery. Capability detection is its own layer,
consumed by the clients and the cluster.

## Invariants

- The public surface is the root lib.rs: modules are private, an item
  becomes public only through an explicit re-export; paths like
  `crate::data::...` do not exist outside the crate.
- All failures funnel into `ClusterError`; the retry and cancellation
  policy is centralized in the trait's execute — a client's call
  returns a raw result and knows nothing about retry logic.
- Cancellation never loses results: cancelled prompts of a batch get
  a result with `ClusterError::Cancelled`, the result buffer is never
  abandoned.
- Image payload normalization is the client's concern: providers want
  different formats (raw base64 / data URI), the shared helper lives
  in support.
- Model capabilities are detected by rules, not hardcoded in Rust:
  the default rule tables ship embedded and are maintained by hand,
  the user extends or replaces them at runtime. A broken rule set is
  rejected at load, never at match time. Ollama detects live via its
  API and only falls back to the rules.
- One file — one entity; mod.rs holds only imports and a doc header
  with a `# Submodules` section; every file starts with an SPDX header.
- Validation is a contract, not a side effect: a type with invariants
  implements the validate trait, validation runs after parsing and
  before the data is applied — invalid data never reaches the engine.
- Tests live in the tests/ directory only, split by task into separate
  binaries; no `#[cfg(test)]` tests inside the crate.
