// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Provider model catalog structure.
//! `ModelInfo` aggregates everything the provider APIs report about a
//! model; fields a provider does not report stay `None`.

/// Everything a provider reports about a single model.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelInfo {
    /// Model name as the provider addresses it (call/execute use it).
    pub name: String,

    // --- identity ---
    /// Content digest (Ollama /api/tags).
    pub digest: Option<String>,
    /// Name of the upstream model, if the model is remote (Ollama cloud).
    pub remote_model: Option<String>,
    /// URL of the upstream host, if the model is remote (Ollama cloud).
    pub remote_host: Option<String>,
    /// Owner the model belongs to (OpenAI-compatible /models `owned_by`).
    pub owned_by: Option<String>,
    /// Last modified timestamp, ISO 8601 (Ollama /api/tags `modified_at`).
    pub modified_at: Option<String>,
    /// Creation timestamp, unix seconds (OpenAI-compatible /models `created`).
    pub created: Option<i64>,

    // --- static facts (Ollama /api/tags `details`, /api/show) ---
    /// Model file format, e.g. "gguf".
    pub format: Option<String>,
    /// Primary model family, e.g. "llama".
    pub family: Option<String>,
    /// All families, incl. projector families like "clip".
    pub families: Option<Vec<String>>,
    /// Approximate parameter count label, e.g. "8.0B".
    pub params: Option<String>,
    /// Quantization level, e.g. "Q4_K_M".
    pub quantization: Option<String>,
    /// Size on disk, bytes.
    pub size: Option<u64>,
    /// Context window in tokens (/api/show → model_info.<arch>.context_length).
    pub context_length: Option<u64>,

    // --- capabilities (tri-state) ---
    pub vision: Option<bool>,
    pub tools: Option<bool>,
    pub thinking: Option<bool>,
    /// Model produces embeddings, not chat completions.
    pub embedding: Option<bool>,
}
