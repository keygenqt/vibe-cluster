// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Client behavior tests: mock-server flows and the local Ollama catalog.

mod common;

use common::mock_server;
use vibe_cluster::CancelToken;
use vibe_cluster::ConnectionParams;
use vibe_cluster::Prompt;
use vibe_cluster::Provider;
use vibe_cluster::count_tokens;
use vibe_cluster::create_client;

#[tokio::test]
async fn test_yandex_models_accepts_missing_catalog() {
    let (host, server) = mock_server(
        "404 Not Found",
        "application/json",
        vec![br#"{"error":"not found"}"#.to_vec()],
    )
    .await;
    let client = create_client(ConnectionParams {
        provider: Provider::Yandex,
        host,
        model: "yandexgpt-5-lite".into(),
        ..Default::default()
    });

    let models = client.list_models().await.unwrap();
    server.await.unwrap();

    let names: Vec<&str> = models.iter().map(|model| model.name.as_str()).collect();
    assert_eq!(names, ["yandexgpt-5-lite", "yandexgpt-5-pro"]);
}

#[tokio::test]
async fn test_ollama_rejects_json_error_with_success_status() {
    let (host, server) = mock_server(
        "200 OK",
        "application/json",
        vec![br#"{"error":"model failed"}"#.to_vec()],
    )
    .await;
    let client = create_client(ConnectionParams {
        provider: Provider::Ollama,
        host,
        model: "test".into(),
        ..Default::default()
    });

    let error = client
        .call(&Prompt::new("test", None, None))
        .await
        .unwrap_err();
    server.await.unwrap();

    assert!(matches!(error, vibe_cluster::ClusterError::Parse(_)));
}

#[tokio::test]
async fn test_stats_flag_counts_tokens() {
    let payload = r#"{"choices":[{"message":{"content":"Hello 👋"}}]}"#;
    let (host, server) = mock_server(
        "200 OK",
        "application/json",
        vec![payload.as_bytes().to_vec()],
    )
    .await;
    let client = create_client(ConnectionParams {
        provider: Provider::OpenAI,
        host,
        model: "gpt-4o".into(),
        stats: true,
        ..Default::default()
    });

    let result = client
        .execute(&Prompt::new("test", None, None), 1, &CancelToken::new())
        .await
        .unwrap();
    server.await.unwrap();

    assert!(result.prompt_tokens > 0);
    assert_eq!(result.completion_tokens, Some(count_tokens("Hello 👋")));
}

#[tokio::test]
async fn test_stats_default_counts_nothing() {
    let payload = r#"{"choices":[{"message":{"content":"Hello 👋"}}]}"#;
    let (host, server) = mock_server(
        "200 OK",
        "application/json",
        vec![payload.as_bytes().to_vec()],
    )
    .await;
    let client = create_client(ConnectionParams {
        provider: Provider::OpenAI,
        host,
        model: "gpt-4o".into(),
        ..Default::default()
    });

    let result = client
        .execute(&Prompt::new("test", None, None), 1, &CancelToken::new())
        .await
        .unwrap();
    server.await.unwrap();

    assert_eq!(result.prompt_tokens, 0);
    assert_eq!(result.completion_tokens, None);
}

#[tokio::test]
async fn test_ollama_list_models() {
    let client = create_client(ConnectionParams {
        provider: Provider::Ollama,
        host: "http://localhost:11434".into(),
        ..Default::default()
    });
    let models = client.list_models().await.unwrap();

    assert!(
        !models.is_empty(),
        "expected at least one model on the host"
    );

    // The model from test_config is pulled on the test host
    let qwen = models
        .iter()
        .find(|m| m.name.starts_with("qwen2.5-coder"))
        .expect("qwen2.5-coder must be pulled on the test host");

    // Facts from /api/tags must always be there for Ollama
    assert!(
        qwen.params.is_some(),
        "parameter_size expected from /api/tags"
    );
    assert!(qwen.size.is_some(), "size in bytes expected from /api/tags");

    println!("models: {}", models.len());
    for m in &models {
        println!(
            "{} — params {:?}, size {:?}, ctx {:?}, vision {:?}, tools {:?}, thinking {:?}, embedding {:?}, decision {:?}",
            m.name,
            m.params,
            m.size,
            m.context_length,
            m.capabilities.input.vision,
            m.capabilities.abilities.tools,
            m.capabilities.abilities.thinking,
            m.capabilities.abilities.embedding,
            m.capabilities.abilities.decision,
        );
    }
}
