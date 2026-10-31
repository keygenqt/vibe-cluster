//! Cluster integration tests
//!
//! cargo test --test cluster_test -- --test-threads=1 --nocapture

use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use vibe_cluster::{
    BatchOptions, CancelToken, Cluster, ClusterError, ConnectionParams, ModelFilter, Prompt,
    PromptEvent, Provider, create_client, normalize_text,
};

async fn mock_server(
    status: &'static str,
    content_type: &'static str,
    chunks: Vec<Vec<u8>>,
) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0; 1024];

        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = socket.read(&mut buffer).await.unwrap();
            assert!(read > 0, "connection closed before request headers");
            request.extend_from_slice(&buffer[..read]);
        }

        let body_len: usize = chunks.iter().map(Vec::len).sum();
        let headers = format!(
            "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            status, content_type, body_len
        );
        socket.write_all(headers.as_bytes()).await.unwrap();

        for chunk in chunks {
            socket.write_all(&chunk).await.unwrap();
            socket.flush().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        socket.shutdown().await.unwrap();
    });

    (format!("http://{}", address), handle)
}

#[test]
fn test_normalize_json_fence_and_backticks() {
    assert_eq!(normalize_text("```json\n{\"a\":1}\n```"), "{\"a\":1}");

    let raw = r#"{"text":"```"}"#;
    assert_eq!(normalize_text(raw), raw);
}

#[tokio::test]
async fn test_openai_stream_split_utf8_crlf_and_final_event() {
    let payload = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\r\n\r\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\" 👋\"}}]}"
    );
    let chunks = payload
        .as_bytes()
        .chunks(1)
        .map(|chunk| chunk.to_vec())
        .collect();
    let (host, server) = mock_server("200 OK", "text/event-stream", chunks).await;
    let client = create_client(ConnectionParams {
        provider: Provider::OpenAI,
        host,
        model: "gpt-4o".into(),
        ..Default::default()
    });

    let mut receiver = client
        .call_stream(&Prompt::new("test", None, None))
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(event) = receiver.recv().await {
        if let PromptEvent::Text(chunk) = event.unwrap() {
            text.push_str(&chunk);
        }
    }
    server.await.unwrap();

    assert_eq!(text, "Hello 👋");
}

#[tokio::test]
async fn test_ollama_stream_split_utf8_and_final_line() {
    let payload = concat!(
        "{\"response\":\"Hello\",\"done\":false}\n",
        "{\"response\":\" 👋\",\"done\":true}"
    );
    let chunks = payload
        .as_bytes()
        .chunks(1)
        .map(|chunk| chunk.to_vec())
        .collect();
    let (host, server) = mock_server("200 OK", "application/x-ndjson", chunks).await;
    let client = create_client(ConnectionParams {
        provider: Provider::Ollama,
        host,
        model: "test".into(),
        ..Default::default()
    });

    let mut receiver = client
        .call_stream(&Prompt::new("test", None, None))
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(event) = receiver.recv().await {
        if let PromptEvent::Text(chunk) = event.unwrap() {
            text.push_str(&chunk);
        }
    }
    server.await.unwrap();

    assert_eq!(text, "Hello 👋");
}

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

    assert!(matches!(error, ClusterError::Parse(_)));
}

fn test_config() -> Vec<ConnectionParams> {
    vec![ConnectionParams {
        provider: Provider::Ollama,
        host: "http://localhost:11434".into(),
        model: "qwen2.5-coder:3b-instruct".into(),
        timeout_secs: Some(60),
        temperature: Some(0.1),
        seed: Some(42),
        num_ctx: Some(4096),
        num_predict: Some(2048),
        ..Default::default()
    }]
}

fn test_config_two_models() -> Vec<ConnectionParams> {
    vec![
        ConnectionParams {
            provider: Provider::Ollama,
            host: "http://localhost:11434".into(),
            model: "qwen2.5-coder:3b-instruct".into(),
            timeout_secs: Some(60),
            temperature: Some(0.1),
            seed: Some(42),
            num_ctx: Some(4096),
            num_predict: Some(2048),
            ..Default::default()
        },
        ConnectionParams {
            provider: Provider::Ollama,
            host: "http://localhost:11434".into(),
            model: "qwen2.5-coder:7b-instruct".into(),
            timeout_secs: Some(60),
            temperature: Some(0.1),
            seed: Some(42),
            num_ctx: Some(4096),
            num_predict: Some(2048),
            ..Default::default()
        },
    ]
}

#[tokio::test]
async fn test_single_call() {
    let cluster = Cluster::new(test_config()).unwrap();
    let result = cluster
        .call(Prompt::new("Reply with just: ok", None, None))
        .await
        .unwrap();
    assert!(result.error.is_none());
    assert!(result.text.unwrap().to_lowercase().contains("ok"));
}

#[tokio::test]
async fn test_batch_call_returns_in_order() {
    let cluster = Cluster::new(test_config()).unwrap();
    let prompts: Vec<Prompt> = vec![
        Prompt::new("Reply with just: one", None, None),
        Prompt::new("Reply with just: two", None, None),
        Prompt::new("Reply with just: three", None, None),
    ];
    let results = cluster.batch_call(&prompts, |_, _, _| {}).await.unwrap();

    assert_eq!(results.len(), 3);
    for (i, r) in results.iter().enumerate() {
        assert!(r.error.is_none(), "prompt {} failed: {:?}", i, r.error);
    }
    assert!(results[0].text.as_ref().unwrap().contains("one"));
    assert!(results[1].text.as_ref().unwrap().contains("two"));
    assert!(results[2].text.as_ref().unwrap().contains("three"));
}

#[tokio::test]
async fn test_batch_call_parallel() {
    let cluster = Cluster::new(test_config_two_models()).unwrap();
    let prompts: Vec<Prompt> = (0..6)
        .map(|i| Prompt::new(format!("Reply with just: {}", i), None, None))
        .collect();
    let results = cluster.batch_call(&prompts, |_, _, _| {}).await.unwrap();

    assert_eq!(results.len(), 6);
    for (i, r) in results.iter().enumerate() {
        assert!(r.error.is_none(), "prompt {} failed: {:?}", i, r.error);
    }
}

#[tokio::test]
async fn test_local_only_filter() {
    let cluster = Cluster::new(test_config()).unwrap();
    let options = BatchOptions {
        models: ModelFilter::LocalOnly,
        ..Default::default()
    };
    let result = cluster
        .call_with_options(Prompt::new("Reply with just: local", None, None), &options)
        .await
        .unwrap();
    assert!(result.error.is_none());
    assert!(result.text.unwrap().to_lowercase().contains("local"));
}

#[tokio::test]
async fn test_cloud_only_filter_no_match() {
    let cluster = Cluster::new(test_config()).unwrap();
    let options = BatchOptions {
        models: ModelFilter::CloudOnly,
        ..Default::default()
    };
    let result = cluster
        .call_with_options(Prompt::new("Reply with just: test", None, None), &options)
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_exclusive_mode() {
    let cluster = Cluster::new(test_config()).unwrap();
    let options = BatchOptions {
        exclusive: true,
        ..Default::default()
    };
    let result = cluster
        .call_with_options(
            Prompt::new("Reply with just: exclusive", None, None),
            &options,
        )
        .await
        .unwrap();
    assert!(result.error.is_none());
    assert!(result.text.unwrap().to_lowercase().contains("exclusive"));
}

#[tokio::test]
async fn test_progress_callback() {
    let cluster = Cluster::new(test_config()).unwrap();
    let prompts: Vec<Prompt> = vec![
        Prompt::new("Reply with just: a", None, None),
        Prompt::new("Reply with just: b", None, None),
        Prompt::new("Reply with just: c", None, None),
    ];

    cluster
        .batch_call(&prompts, |result, done, total| {
            println!(
                "[{}/{}] {} in {}ms — {}",
                done,
                total,
                result.model,
                result.duration_ms,
                if result.error.is_none() { "OK" } else { "FAIL" }
            );
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn test_empty_prompts() {
    let cluster = Cluster::new(test_config()).unwrap();
    let results = cluster.batch_call(&[], |_, _, _| {}).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_empty_connections() {
    let result = Cluster::new(vec![]);
    assert!(result.is_err());
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

    // Capabilities are tri-state: None (old server / show failed) is valid too
    println!("models: {}", models.len());
    for m in &models {
        println!(
            "{} — params {:?}, size {:?}, vision {:?}, tools {:?}, thinking {:?}, embedding {:?}, ctx {:?}",
            m.name, m.params, m.size, m.vision, m.tools, m.thinking, m.embedding, m.context_length
        );
    }
}

#[tokio::test]
async fn test_unreachable_host() {
    let config = vec![ConnectionParams {
        provider: Provider::Ollama,
        host: "http://localhost:99999".into(),
        model: "qwen2.5-coder:3b-instruct".into(),
        timeout_secs: Some(1),
        ..Default::default()
    }];
    let cluster = Cluster::new(config).unwrap();
    let result = cluster.call(Prompt::new("test", None, None)).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_stream_call() {
    let cluster = Cluster::new(test_config()).unwrap();
    let cancel = CancelToken::new();

    let events: Arc<std::sync::Mutex<Vec<PromptEvent>>> = Arc::default();
    let events_seen = events.clone();

    let result = cluster
        .call_stream(
            Prompt::new("Reply with just: ok", None, None),
            &cancel,
            move |event| events_seen.lock().unwrap().push(event.clone()),
        )
        .await
        .unwrap();

    assert!(result.error.is_none());
    assert!(result.text.as_ref().unwrap().to_lowercase().contains("ok"));

    // the deltas the callback saw must assemble into the final text
    let mut assembled = String::new();
    for event in events.lock().unwrap().iter() {
        if let PromptEvent::Text(chunk) = event {
            assembled.push_str(chunk);
        }
    }
    assert_eq!(assembled, result.text.unwrap());
}

#[tokio::test]
async fn test_stream_cancel() {
    let cluster = Cluster::new(test_config()).unwrap();
    let cancel = CancelToken::new();

    // cancelling before the call makes the stream cut immediately
    cancel.cancel();

    let result = cluster
        .call_stream(
            Prompt::new("Reply with just: ok", None, None),
            &cancel,
            |_| {},
        )
        .await
        .unwrap();

    assert!(matches!(
        result.error,
        Some(vibe_cluster::ClusterError::Cancelled)
    ));
}

#[tokio::test]
async fn test_batch_cancel() {
    let cluster = Cluster::new(test_config()).unwrap();

    let cancel = CancelToken::new();
    cancel.cancel();

    let options = BatchOptions {
        cancel,
        ..Default::default()
    };

    let prompts: Vec<Prompt> = (0..3)
        .map(|i| Prompt::new(format!("Reply with just: {}", i), None, None))
        .collect();

    let results = cluster
        .batch_call_with_options(&prompts, &options, |_, _, _| {})
        .await
        .unwrap();

    assert_eq!(results.len(), 3);
    for (i, r) in results.iter().enumerate() {
        assert!(
            matches!(r.error, Some(vibe_cluster::ClusterError::Cancelled)),
            "prompt {} expected Cancelled: {:?}",
            i,
            r.error
        );
    }
}
