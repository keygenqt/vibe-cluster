//! Internal HTTP clients for each supported API type.
//! Ollama uses its own API; OpenAI-compatible covers DeepSeek, Qwen,
//! Moonshot, Zhipu and Yandex; GigaChat uses OAuth + OpenAI-compatible.

use crate::{
    clients::{
        client::Client, gigachat::GigaChatClient, ollama::OllamaClient, openai::OpenAiClient,
    },
    params::ConnectionParams,
    provider::Provider,
};

pub mod client;
mod gigachat;
mod ollama;
mod openai;
mod vision;

pub fn create_client(params: ConnectionParams) -> Box<dyn Client> {
    match params.provider {
        Provider::Ollama => Box::new(OllamaClient::new(params)),
        Provider::DeepSeek
        | Provider::Qwen
        | Provider::Moonshot
        | Provider::Zhipu
        | Provider::Yandex => Box::new(OpenAiClient::new(params)),
        Provider::GigaChat => Box::new(GigaChatClient::new(params)),
    }
}
