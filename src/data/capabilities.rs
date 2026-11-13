// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Model capabilities: input/output modalities and abilities.
//! Fields a source does not report stay `None`.

/// What a model can do: input modalities, output modalities and abilities.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Capabilities {
    /// Modalities the model accepts.
    pub input: Input,
    /// Modalities the model produces.
    pub output: Output,
    /// What the model can do and which tasks it serves.
    pub abilities: Abilities,
}

/// Modalities the model accepts. Text is a first-class field: catalogs
/// carry models without text input (transcription) even though the
/// cluster cannot call them through chat completions yet.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Input {
    /// Accepts text prompts.
    pub text: Option<bool>,
    /// Accepts images.
    pub vision: Option<bool>,
    /// Accepts audio.
    pub audio: Option<bool>,
    /// Accepts video.
    pub video: Option<bool>,
}

/// Modalities the model produces. Text is a first-class field: pure
/// generators (dall-e) produce no text at all.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Output {
    /// Produces text.
    pub text: Option<bool>,
    /// Generates images.
    pub image: Option<bool>,
    /// Generates audio (TTS, speech).
    pub audio: Option<bool>,
    /// Generates video.
    pub video: Option<bool>,
}

/// What the model can do and which tasks it serves.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Abilities {
    /// Supports tool (function) calling.
    pub tools: Option<bool>,
    /// Supports thinking (reasoning) output.
    pub thinking: Option<bool>,
    /// Supports reasoning depth control (`reasoning_effort` /
    /// `think` — the knob shape is the params family's business).
    pub effort: Option<bool>,
    /// Produces embeddings instead of chat completions.
    pub embedding: Option<bool>,
    /// System One classification (Ollama `decision`).
    pub decision: Option<bool>,
}

impl Capabilities {
    /// True if the model accepts any non-text input.
    pub fn has_modal_input(&self) -> bool {
        self.input.vision == Some(true)
            || self.input.audio == Some(true)
            || self.input.video == Some(true)
    }
}
