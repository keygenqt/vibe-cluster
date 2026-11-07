// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! CapabilityRules engine tests: parsing, validation, matching.
//! Pure tests — no network.

use vibe_cluster::CapabilityRules;
use vibe_cluster::ClusterError;
use vibe_cluster::Provider;

#[test]
fn default_rules_parse() {
    CapabilityRules::new();
}

#[test]
fn keyword_matches_word_only() {
    let rules = CapabilityRules::new();

    let caps = rules.capabilities(&Provider::Qwen, "qwen3-vl-plus");
    assert_eq!(caps.input.vision, Some(true));

    // `vl` inside `vllm` is not a whole word: the rule entry exists,
    // so the miss is a definitive `Some(false)`
    let caps = rules.capabilities(&Provider::Qwen, "vllm");
    assert_eq!(caps.input.vision, Some(false));
}

#[test]
fn names_match_exactly() {
    let rules = CapabilityRules::new();

    let caps = rules.capabilities(&Provider::OpenAI, "o1");
    assert_eq!(caps.input.vision, Some(true));

    // `o1-mini` is not in the names list and carries no vision keyword
    let caps = rules.capabilities(&Provider::OpenAI, "o1-mini");
    assert_eq!(caps.input.vision, Some(false));
}

#[test]
fn include_regex_matches() {
    let rules = CapabilityRules::new();

    let caps = rules.capabilities(&Provider::Zhipu, "glm-4v");
    assert_eq!(caps.input.vision, Some(true));

    let caps = rules.capabilities(&Provider::Zhipu, "glm-4.5v");
    assert_eq!(caps.input.vision, Some(true));

    // text-only GLM: no digit+v marker
    let caps = rules.capabilities(&Provider::Zhipu, "glm-4.5");
    assert_eq!(caps.input.vision, Some(false));
}

#[test]
fn exclude_regex_vetoes() {
    let rules = CapabilityRules::new();

    let caps = rules.capabilities(&Provider::OpenAI, "gpt-4o");
    assert_eq!(caps.abilities.tools, Some(true));

    // generators are excluded from tools by `^gpt-image` — the veto
    // turns the match into a definitive `Some(false)`
    let caps = rules.capabilities(&Provider::OpenAI, "gpt-image-1");
    assert_eq!(caps.abilities.tools, Some(false));
    assert_eq!(caps.output.image, Some(true));
}

#[test]
fn matching_is_case_insensitive() {
    let rules = CapabilityRules::new();

    let caps = rules.capabilities(&Provider::OpenAI, "GPT-4O");
    assert_eq!(caps.input.vision, Some(true));
}

#[test]
fn closed_world_matrix() {
    let rules = CapabilityRules::new();

    // every capability of every provider carries a rule set or a
    // default — a miss is a definitive `false`, `None` means the
    // detection itself failed, not "we do not know"
    let caps = rules.capabilities(&Provider::GigaChat, "whatever");
    assert_eq!(caps.input.text, Some(true));
    assert_eq!(caps.input.vision, Some(false));
    assert_eq!(caps.input.audio, Some(false));
    assert_eq!(caps.output.text, Some(true));
    assert_eq!(caps.abilities.tools, Some(false));
    assert_eq!(caps.abilities.decision, Some(false));
}

#[test]
fn has_modal_input() {
    let rules = CapabilityRules::new();

    let caps = rules.capabilities(&Provider::Qwen, "qwen-omni-turbo");
    assert!(caps.has_modal_input());

    let caps = rules.capabilities(&Provider::Qwen, "qwen-max");
    assert!(!caps.has_modal_input());
}

#[test]
fn add_appends_rules() {
    let mut rules = CapabilityRules::new();
    rules
        .add(r#"{"version":1,"providers":{"openai":{"input":{"vision":{"names":["my-model"]}}}}}"#)
        .unwrap();

    // the appended name matches (case-insensitive)
    let caps = rules.capabilities(&Provider::OpenAI, "My-Model");
    assert_eq!(caps.input.vision, Some(true));

    // the default rules still work
    let caps = rules.capabilities(&Provider::OpenAI, "gpt-4o");
    assert_eq!(caps.input.vision, Some(true));
}

#[test]
fn set_replaces_rules() {
    let mut rules = CapabilityRules::new();
    rules
        .set(r#"{"version":1,"providers":{"openai":{"input":{"vision":{"names":["my-model"]}}}}}"#)
        .unwrap();

    let caps = rules.capabilities(&Provider::OpenAI, "my-model");
    assert_eq!(caps.input.vision, Some(true));

    // the default tables are gone: the new table exists, so a miss is false
    let caps = rules.capabilities(&Provider::OpenAI, "gpt-4o");
    assert_eq!(caps.input.vision, Some(false));
}

#[test]
fn invalid_regex_fails_and_keeps_rules() {
    let mut rules = CapabilityRules::new();

    let error = rules
        .add(r#"{"version":1,"providers":{"openai":{"input":{"vision":{"include":["(["]}}}}}"#)
        .unwrap_err();
    assert!(matches!(error, ClusterError::Config(_)));

    // the current set is untouched
    let caps = rules.capabilities(&Provider::OpenAI, "gpt-4o");
    assert_eq!(caps.input.vision, Some(true));
}

#[test]
fn validation_rejects_unknown_provider() {
    let mut rules = CapabilityRules::new();

    let error = rules
        .add(r#"{"version":1,"providers":{"unknown":{"input":{"vision":{"names":["x"]}}}}}"#)
        .unwrap_err();
    assert!(matches!(error, ClusterError::Config(_)));
}

#[test]
fn validation_rejects_unknown_capability() {
    let mut rules = CapabilityRules::new();

    let error = rules
        .add(r#"{"version":1,"providers":{"openai":{"input":{"visio":{"names":["x"]}}}}}"#)
        .unwrap_err();
    assert!(matches!(error, ClusterError::Config(_)));
}

#[test]
fn validation_rejects_wrong_group_key() {
    // `vision` does not exist in the abilities group
    let mut rules = CapabilityRules::new();

    let error = rules
        .add(r#"{"version":1,"providers":{"openai":{"abilities":{"vision":{"names":["x"]}}}}}"#)
        .unwrap_err();
    assert!(matches!(error, ClusterError::Config(_)));
}

#[test]
fn validation_rejects_bad_version() {
    let mut rules = CapabilityRules::new();

    let error = rules.add(r#"{"version":2,"providers":{}}"#).unwrap_err();
    assert!(matches!(error, ClusterError::Config(_)));
}

#[test]
fn text_default_denylist() {
    let rules = CapabilityRules::new();

    // chat models take and produce text
    let caps = rules.capabilities(&Provider::OpenAI, "gpt-4o");
    assert_eq!(caps.input.text, Some(true));
    assert_eq!(caps.output.text, Some(true));

    // generators produce no text but still take a text prompt
    let caps = rules.capabilities(&Provider::OpenAI, "dall-e-3");
    assert_eq!(caps.output.text, Some(false));
    assert_eq!(caps.input.text, Some(true));

    // transcription accepts no text
    let caps = rules.capabilities(&Provider::OpenAI, "whisper-1");
    assert_eq!(caps.input.text, Some(false));
}

#[test]
fn user_lists_take_priority_over_default() {
    let mut rules = CapabilityRules::new();

    // builtin input.text: default true; the user flips the miss to
    // false and allowlists one model — lists beat the default
    rules
        .add(
            r#"{"version":1,"providers":{"openai":{"input":{"text":{
                "default":false,"names":["special"]}}}}}"#,
        )
        .unwrap();

    let caps = rules.capabilities(&Provider::OpenAI, "special");
    assert_eq!(caps.input.text, Some(true));

    let caps = rules.capabilities(&Provider::OpenAI, "gpt-4o");
    assert_eq!(caps.input.text, Some(false));

    // the builtin veto keeps working
    let caps = rules.capabilities(&Provider::OpenAI, "whisper-1");
    assert_eq!(caps.input.text, Some(false));
}
