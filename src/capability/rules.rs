// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Name-based capability rules: a rule set per provider, matched against
//! the lowercase model name. Default rules are embedded (`rules.json`);
//! user JSON extends them via [`CapabilityRules::add`] or replaces them
//! via [`CapabilityRules::set`] — invalid input is rejected before it
//! touches the current set, so matching itself cannot fail.

use std::collections::HashMap;

use regex::Regex;
use serde::Deserialize;

use crate::data::Abilities;
use crate::data::Capabilities;
use crate::data::Input;
use crate::data::Output;
use crate::data::Provider;
use crate::error::ClusterError;
use crate::validate::ValidateTrait;

/// Rules file version this engine understands.
pub(crate) const SUPPORTED_VERSION: u32 = 1;

// ============================================================================
// PUBLIC API
// ============================================================================

/// Name-based capability rules matched against model names.
///
/// Matching never fails: user input is validated and compiled on
/// `add`/`set`, invalid input is rejected with `ClusterError::Config`.
#[derive(Debug, Clone, Default)]
pub struct CapabilityRules {
    providers: HashMap<Provider, HashMap<Capability, FieldRules>>,
}

impl CapabilityRules {
    /// Rules embedded in the crate (`rules.json`).
    pub fn new() -> Self {
        let mut rules = Self::default();
        // The embedded file ships with the crate and must parse.
        rules
            .add(include_str!("rules.json"))
            .expect("embedded rules.json is valid");
        rules
    }

    /// Appends rules from a user JSON to the current set: lists are
    /// concatenated per provider+capability, missing keys are created.
    /// On any validation error the current set stays untouched.
    pub fn add(&mut self, json: &str) -> Result<(), ClusterError> {
        let parsed = ParsedRules::from_str(json)?;
        for (provider, capabilities) in parsed.providers {
            let entry = self.providers.entry(provider).or_default();
            for (capability, field) in capabilities {
                entry.entry(capability).or_default().append(field);
            }
        }
        Ok(())
    }

    /// Replaces the whole rule set with the parsed user JSON.
    /// On any validation error the current set stays untouched.
    pub fn set(&mut self, json: &str) -> Result<(), ClusterError> {
        let parsed = ParsedRules::from_str(json)?;
        self.providers = parsed.providers;
        Ok(())
    }

    /// Resolves the capabilities of a model name by the rule set.
    /// A capability without a rule entry stays `None`; an entry that
    /// does not match yields a definitive `Some(false)` — that is what
    /// makes `exclude` meaningful.
    pub fn capabilities(&self, provider: &Provider, model: &str) -> Capabilities {
        let name = model.to_ascii_lowercase();
        let Some(rules) = self.providers.get(provider) else {
            return Capabilities::default();
        };
        let get = |capability| rules.get(&capability).map(|field| field.matches(&name));
        Capabilities {
            input: Input {
                text: get(Capability::InputText),
                vision: get(Capability::InputVision),
                audio: get(Capability::InputAudio),
                video: get(Capability::InputVideo),
            },
            output: Output {
                text: get(Capability::OutputText),
                image: get(Capability::OutputImage),
                audio: get(Capability::OutputAudio),
                video: get(Capability::OutputVideo),
            },
            abilities: Abilities {
                tools: get(Capability::Tools),
                thinking: get(Capability::Thinking),
                effort: get(Capability::Effort),
                embedding: get(Capability::Embedding),
                decision: get(Capability::Decision),
            },
        }
    }
}

// ============================================================================
// PARSING
// ============================================================================

/// A parsed and validated user file: provider → capability → rules,
/// regexes already compiled.
#[derive(Debug, Default)]
struct ParsedRules {
    providers: HashMap<Provider, HashMap<Capability, FieldRules>>,
}

impl ParsedRules {
    /// Parses, validates and compiles a rules file.
    fn from_str(json: &str) -> Result<Self, ClusterError> {
        let file: RawFile = serde_json::from_str(json).map_err(|error| {
            ClusterError::Config(format!("Capability rules: invalid JSON: {}", error))
        })?;
        file.validate()?;
        Ok(file.compile())
    }
}

/// Rules file as it comes from JSON.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawFile {
    /// Defaults to 0 so a missing field fails the version check with a
    /// clear message instead of a generic parse error.
    #[serde(default)]
    pub(crate) version: u32,
    #[serde(default)]
    pub(crate) providers: HashMap<String, RawProvider>,
}

/// One provider section: capability groups.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawProvider {
    #[serde(default)]
    pub(crate) input: HashMap<String, RawField>,
    #[serde(default)]
    pub(crate) output: HashMap<String, RawField>,
    #[serde(default)]
    pub(crate) abilities: HashMap<String, RawField>,
}

impl RawProvider {
    /// Iterates the capability groups as (name, rules) pairs.
    pub(crate) fn groups(&self) -> [(&'static str, &HashMap<String, RawField>); 3] {
        [
            ("input", &self.input),
            ("output", &self.output),
            ("abilities", &self.abilities),
        ]
    }
}

/// One capability rule set as it comes from JSON.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawField {
    /// The miss value: `exclude` vetoes, an include-side match wins,
    /// everything else resolves to this. The weakest rule — a user
    /// `default` replaces the embedded one on `add`.
    #[serde(default)]
    pub(crate) default: Option<bool>,
    #[serde(default)]
    pub(crate) names: Vec<String>,
    #[serde(default)]
    pub(crate) keywords: Vec<String>,
    #[serde(default)]
    pub(crate) include: Vec<String>,
    #[serde(default)]
    pub(crate) exclude: Vec<String>,
}

impl RawFile {
    /// Compiles the validated file into the matching structures.
    fn compile(self) -> ParsedRules {
        let mut providers: HashMap<Provider, HashMap<Capability, FieldRules>> = HashMap::new();
        for (provider, raw) in self.providers {
            let provider = parse_provider(&provider).expect("validated provider name");
            let entry = providers.entry(provider).or_default();
            for (group, capabilities) in raw.groups() {
                for (key, field) in capabilities {
                    let capability =
                        parse_capability(group, key).expect("validated capability key");
                    entry.insert(capability, field.clone().compile());
                }
            }
        }
        ParsedRules { providers }
    }
}

impl RawField {
    /// Compiles one field: plain strings are lowercased, regex patterns
    /// are kept as written — they must match lowercase names.
    fn compile(self) -> FieldRules {
        FieldRules {
            default: self.default,
            names: self
                .names
                .into_iter()
                .map(|entry| entry.to_ascii_lowercase())
                .collect(),
            keywords: self
                .keywords
                .into_iter()
                .map(|key| key.to_ascii_lowercase())
                .collect(),
            include: compile_regexes(self.include),
            exclude: compile_regexes(self.exclude),
        }
    }
}

/// Compiles pre-validated regex patterns.
fn compile_regexes(patterns: Vec<String>) -> Vec<Regex> {
    patterns
        .into_iter()
        .map(|pattern| Regex::new(&pattern).expect("validated regex"))
        .collect()
}

/// Maps a rules-file provider name to the `Provider` kind.
pub(crate) fn parse_provider(name: &str) -> Option<Provider> {
    match name {
        "ollama" => Some(Provider::Ollama),
        "openai" => Some(Provider::OpenAI),
        "deepseek" => Some(Provider::DeepSeek),
        "qwen" => Some(Provider::Qwen),
        "moonshot" => Some(Provider::Moonshot),
        "zhipu" => Some(Provider::Zhipu),
        "yandex" => Some(Provider::Yandex),
        "gigachat" => Some(Provider::GigaChat),
        _ => None,
    }
}

/// Maps a `group.key` pair to a capability field; each group accepts
/// only its own keys, typos fail validation.
pub(crate) fn parse_capability(group: &str, key: &str) -> Option<Capability> {
    match (group, key) {
        ("input", "text") => Some(Capability::InputText),
        ("input", "vision") => Some(Capability::InputVision),
        ("input", "audio") => Some(Capability::InputAudio),
        ("input", "video") => Some(Capability::InputVideo),
        ("output", "text") => Some(Capability::OutputText),
        ("output", "image") => Some(Capability::OutputImage),
        ("output", "audio") => Some(Capability::OutputAudio),
        ("output", "video") => Some(Capability::OutputVideo),
        ("abilities", "tools") => Some(Capability::Tools),
        ("abilities", "thinking") => Some(Capability::Thinking),
        ("abilities", "effort") => Some(Capability::Effort),
        ("abilities", "embedding") => Some(Capability::Embedding),
        ("abilities", "decision") => Some(Capability::Decision),
        _ => None,
    }
}

// ============================================================================
// MATCHING
// ============================================================================

/// A capability field of the `Capabilities` struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Capability {
    InputText,
    InputVision,
    InputAudio,
    InputVideo,
    OutputText,
    OutputImage,
    OutputAudio,
    OutputVideo,
    Tools,
    Thinking,
    Effort,
    Embedding,
    Decision,
}

/// Compiled rules of one capability field.
#[derive(Debug, Clone, Default)]
struct FieldRules {
    default: Option<bool>,
    names: Vec<String>,
    keywords: Vec<String>,
    include: Vec<Regex>,
    exclude: Vec<Regex>,
}

impl FieldRules {
    /// True when the lowercase model name passes the field's rules.
    /// Strength order: `exclude` vetoes, an include-side match wins,
    /// `default` resolves the miss.
    fn matches(&self, name: &str) -> bool {
        if self.exclude.iter().any(|regex| regex.is_match(name)) {
            return false;
        }
        if self.names.iter().any(|entry| entry == name)
            || self.keywords.iter().any(|key| has_key(name, key))
            || self.include.iter().any(|regex| regex.is_match(name))
        {
            return true;
        }
        self.default.unwrap_or(false)
    }

    /// Appends another compiled field (the `add` semantics); a user
    /// `default` overrides the embedded one.
    fn append(&mut self, other: FieldRules) {
        if other.default.is_some() {
            self.default = other.default;
        }
        self.names.extend(other.names);
        self.keywords.extend(other.keywords);
        self.include.extend(other.include);
        self.exclude.extend(other.exclude);
    }
}

/// Matches a key as a whole "word": the match counts only when the key is
/// bounded on both sides by the start/end of the string or a separator
/// (`-`, `_`, `.`, `:`, `/`).
///
/// So `vl` is found in `qwen3-vl-plus` but not inside `vllm`,
/// and `glm` is found in `glm-4v` but not in `glm4`.
fn has_key(name: &str, key: &str) -> bool {
    name.match_indices(key).any(|(start, _)| {
        let end = start + key.len();
        let left_boundary = start == 0 || is_separator(name.as_bytes()[start - 1]);
        let right_boundary = end == name.len() || is_separator(name.as_bytes()[end]);

        left_boundary && right_boundary
    })
}

/// Checks a model-name separator byte.
fn is_separator(byte: u8) -> bool {
    matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
}
