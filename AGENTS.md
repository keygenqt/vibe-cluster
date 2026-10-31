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
kept apart from the machinery.

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
- Vision detection for cloud providers is a name-based heuristic: the
  APIs do not report modality, the name table is maintained by hand
  and updated when new models ship. Ollama detects live via /api/show.
- One file — one entity; mod.rs holds only imports and a doc header
  with a `# Submodules` section; every file starts with an SPDX header.
