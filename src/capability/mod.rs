// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Name-based capability detection: rules per provider, matched against
//! model names. Default rules ship embedded (`rules.json`) and are
//! maintained by hand; the user extends or replaces them at runtime
//! via `CapabilityRules::add`/`set` without a crate release.
//!
//! # Submodules
//!
//! - `rules` — `CapabilityRules`: parsing, matching, add/set.
//!
//! # Rule tables
//!
//! The `vision` tables are ported from the former `support::vision`
//! and verified against provider documentation. The remaining
//! capabilities (thinking, tools, audio/video in, image/audio/video
//! out, embedding) are best-effort as of November 2026 — verify
//! against the provider docs before trusting them. The embedded tables
//! are closed-world: every capability of every provider carries a rule
//! set or a `default`, a miss is a definitive `Some(false)`. A user
//! `set()` with a partial file lifts that guarantee: a capability the
//! file does not cover stays `None` ("no rule"), as does a provider
//! absent from the file.
//! A rule with `default` sets the miss value: `exclude` vetoes, an
//! include-side match wins, the rest resolves to the default — this is
//! how the implied-everywhere `text` is expressed (all models except
//! the known non-text generators).

mod rules;

pub use rules::CapabilityRules;

pub(crate) use rules::RawField;
pub(crate) use rules::RawFile;
pub(crate) use rules::SUPPORTED_VERSION;
pub(crate) use rules::parse_capability;
pub(crate) use rules::parse_provider;
