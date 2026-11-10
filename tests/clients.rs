// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Client behavior tests: mock-server flows and the local Ollama catalog.

mod common;

use vibe_cluster::Bpe;
use vibe_cluster::CancelToken;
use vibe_cluster::Connection;
use vibe_cluster::OllamaParams;
use vibe_cluster::OpenAiParams;
use vibe_cluster::Params;
use vibe_cluster::Prompt;
use vibe_cluster::Provider;
use vibe_cluster::count_tokens;
use vibe_cluster::create_client;

use crate::common::mock_server::mock_server;
use crate::common::mock_server::mock_server_capture;

#[tokio::test]
async fn test_yandex_models_accepts_missing_catalog() {
    let (host, server) = mock_server(
        "404 Not Found",
        "application/json",
        vec![br#"{"error":"not found"}"#.to_vec()],
    )
    .await;
    let client = create_client(Connection {
        provider: Provider::Yandex,
        host,
        model: "yandexgpt-5-lite".into(),
        params: Params::OpenAi(OpenAiParams::default()),
        ..Default::default()
    })
    .unwrap();

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
    let client = create_client(Connection {
        provider: Provider::Ollama,
        host,
        model: "test".into(),
        ..Default::default()
    })
    .unwrap();

    let error = client
        .call(&Prompt::new("test", None, None))
        .await
        .unwrap_err();
    server.await.unwrap();

    assert!(matches!(error, vibe_cluster::ClusterError::Parse(_)));
}

#[tokio::test]
async fn test_usage_counts_provider_report() {
    let payload = r#"{"choices":[{"message":{"content":"Hello 👋"}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
    let (host, server) = mock_server(
        "200 OK",
        "application/json",
        vec![payload.as_bytes().to_vec()],
    )
    .await;
    let client = create_client(Connection {
        provider: Provider::OpenAI,
        host,
        model: "gpt-4o".into(),
        params: Params::OpenAi(OpenAiParams::default()),
        ..Default::default()
    })
    .unwrap();

    let result = client
        .execute(&Prompt::new("test", None, None), 1, &CancelToken::new())
        .await
        .unwrap();
    server.await.unwrap();

    let stats = result.stats.expect("the provider reported the usage");
    assert_eq!(stats.cached_tokens, None);
    assert_eq!(stats.input_tokens, Some(10));
    assert_eq!(stats.output_tokens, Some(5));
    assert_eq!(stats.thinking_tokens, None);
}

#[tokio::test]
async fn test_usage_absent_stays_none() {
    let payload = r#"{"choices":[{"message":{"content":"Hello 👋"}}]}"#;
    let (host, server) = mock_server(
        "200 OK",
        "application/json",
        vec![payload.as_bytes().to_vec()],
    )
    .await;
    let client = create_client(Connection {
        provider: Provider::OpenAI,
        host,
        model: "gpt-4o".into(),
        params: Params::OpenAi(OpenAiParams::default()),
        ..Default::default()
    })
    .unwrap();

    let result = client
        .execute(&Prompt::new("test", None, None), 1, &CancelToken::new())
        .await
        .unwrap();
    server.await.unwrap();

    let stats = result.stats.expect("a completed call carries the report");
    assert_eq!(stats.cached_tokens, None);
    assert_eq!(stats.input_tokens, None);
    assert_eq!(stats.output_tokens, None);
    assert_eq!(stats.thinking_tokens, None);
}

#[tokio::test]
async fn test_ollama_list_models() {
    let client = create_client(Connection {
        provider: Provider::Ollama,
        host: "http://localhost:11434".into(),
        ..Default::default()
    })
    .unwrap();
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

#[tokio::test]
async fn test_ollama_params_land_in_options() {
    let payload = r#"{"response":"ok","done":true}"#;
    let (host, server, request) = mock_server_capture(
        "200 OK",
        "application/x-ndjson",
        vec![payload.as_bytes().to_vec()],
    )
    .await;
    let client = create_client(Connection {
        provider: Provider::Ollama,
        host,
        model: "test".into(),
        params: Params::Ollama(OllamaParams {
            temperature: Some(0.7),
            top_k: Some(20),
            stop: Some(vec!["STOP".into()]),
            ..Default::default()
        }),
        ..Default::default()
    })
    .unwrap();

    client.call(&Prompt::new("test", None, None)).await.unwrap();
    server.await.unwrap();

    let request = request.lock().unwrap().clone();
    assert!(request.contains("\"top_k\":20"));
    assert!(request.contains("\"stop\":[\"STOP\"]"));
    // unset knobs stay out: the server default applies
    assert!(!request.contains("min_p"));
}

#[tokio::test]
async fn test_openai_params_land_in_body() {
    let payload = r#"{"choices":[{"message":{"content":"ok"}}]}"#;
    let (host, server, request) = mock_server_capture(
        "200 OK",
        "application/json",
        vec![payload.as_bytes().to_vec()],
    )
    .await;
    let client = create_client(Connection {
        provider: Provider::OpenAI,
        host,
        model: "gpt-4o".into(),
        params: Params::OpenAi(OpenAiParams {
            num_predict: Some(512),
            min_p: Some(0.05),
            stop: Some(Vec::new()),
            ..Default::default()
        }),
        ..Default::default()
    })
    .unwrap();

    client.call(&Prompt::new("test", None, None)).await.unwrap();
    server.await.unwrap();

    let request = request.lock().unwrap().clone();
    // num_predict goes out under the OpenAI wire name
    assert!(request.contains("\"max_tokens\":512"));
    assert!(request.contains("\"min_p\":0.05"));
    // an empty stop list stays out
    assert!(!request.contains("\"stop\""));
}

#[tokio::test]
async fn test_ollama_counters_from_final_chunk() {
    let payload = concat!(
        r#"{"response":"ok","done":false}"#,
        "\n",
        r#"{"response":"","done":true,"done_reason":"stop","prompt_eval_count":21,"prompt_eval_cached_count":20,"eval_count":35}"#,
        "\n",
    );
    let (host, server) = mock_server(
        "200 OK",
        "application/x-ndjson",
        vec![payload.as_bytes().to_vec()],
    )
    .await;
    let client = create_client(Connection {
        provider: Provider::Ollama,
        host,
        model: "test".into(),
        ..Default::default()
    })
    .unwrap();

    let result = client
        .execute(&Prompt::new("test", None, None), 1, &CancelToken::new())
        .await
        .unwrap();
    server.await.unwrap();

    // the cached prefix comes off the prompt, the thinking stays
    // unsplit — Ollama folds it into eval_count
    let stats = result.stats.expect("the final chunk carried the counters");
    assert_eq!(stats.cached_tokens, Some(20));
    assert_eq!(stats.input_tokens, Some(1));
    assert_eq!(stats.output_tokens, Some(35));
    assert_eq!(stats.thinking_tokens, None);
}

#[tokio::test]
async fn test_usage_deepseek_aliases() {
    // no prompt_tokens_details.cached_tokens — the DeepSeek alias fills
    // the cache; the completion includes the reasoning tokens
    let payload = r#"{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":35,"completion_tokens":9,"completion_tokens_details":{"reasoning_tokens":7},"prompt_cache_hit_tokens":5}}"#;
    let (host, server) = mock_server(
        "200 OK",
        "application/json",
        vec![payload.as_bytes().to_vec()],
    )
    .await;
    let client = create_client(Connection {
        provider: Provider::OpenAI,
        host,
        model: "deepseek-reasoner".into(),
        params: Params::OpenAi(OpenAiParams::default()),
        ..Default::default()
    })
    .unwrap();

    let result = client
        .execute(&Prompt::new("test", None, None), 1, &CancelToken::new())
        .await
        .unwrap();
    server.await.unwrap();

    let stats = result.stats.expect("the provider reported the usage");
    assert_eq!(stats.cached_tokens, Some(5));
    assert_eq!(stats.input_tokens, Some(30));
    assert_eq!(stats.output_tokens, Some(2));
    assert_eq!(stats.thinking_tokens, Some(7));
}

#[tokio::test]
async fn test_usage_bpe_fills_missing() {
    let payload = r#"{"choices":[{"message":{"content":"Hello 👋"}}]}"#;
    let (host, server) = mock_server(
        "200 OK",
        "application/json",
        vec![payload.as_bytes().to_vec()],
    )
    .await;
    let client = create_client(Connection {
        provider: Provider::OpenAI,
        host,
        model: "gpt-4o".into(),
        params: Params::OpenAi(OpenAiParams::default()),
        bpe: Some(Bpe::O200kBase),
        ..Default::default()
    })
    .unwrap();

    let result = client
        .execute(&Prompt::new("test", None, None), 1, &CancelToken::new())
        .await
        .unwrap();
    server.await.unwrap();

    // no usage in the response — the estimator fills what it can count;
    // the cache is never estimated
    let stats = result.stats.expect("a completed call carries the report");
    assert_eq!(stats.cached_tokens, None);
    assert_eq!(
        stats.input_tokens,
        Some(count_tokens("test", Bpe::O200kBase))
    );
    assert_eq!(
        stats.output_tokens,
        Some(count_tokens("Hello 👋", Bpe::O200kBase))
    );
    assert_eq!(stats.thinking_tokens, Some(0));
}
