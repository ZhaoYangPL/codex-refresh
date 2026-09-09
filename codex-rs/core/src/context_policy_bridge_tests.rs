use std::path::Path;

use codex_config::types::ContextPolicyConfig;
use codex_utils_absolute_path::AbsolutePathBuf;
use serde_json::{Value, json};
use tempfile::TempDir;

use super::{ContextPolicyBridge, PROTOCOL_VERSION};

fn absolute(path: &Path) -> AbsolutePathBuf {
    AbsolutePathBuf::from_absolute_path(path).expect("test path is absolute")
}

fn python() -> AbsolutePathBuf {
    let candidates = if cfg!(windows) {
        ["python", "python3"]
    } else {
        ["python3", "python"]
    };
    candidates
        .into_iter()
        .find_map(|candidate| which::which(candidate).ok())
        .map(|path| absolute(&path))
        .expect("Phase 8B process-boundary tests require Python")
}

fn script(directory: &TempDir) -> AbsolutePathBuf {
    let path = directory.path().join("bridge_fixture.py");
    std::fs::write(
        &path,
        r#"import json
import sys
import time

behavior = sys.argv[1]
for line in sys.stdin:
    request = json.loads(line)
    if behavior == "exit":
        raise SystemExit(7)
    if behavior == "timeout":
        time.sleep(5)
    if behavior == "malformed":
        print("not-json", flush=True)
        continue
    if behavior == "nonfinite":
        print('{"q_keep":NaN}', flush=True)
        continue
    response = {key: request[key] for key in (
        "protocol_version", "request_id", "run_id", "task_id",
        "replicate_id", "thread_id", "epoch")}
    response["type"] = "initialized" if request["type"] == "initialize" else "decision"
    if response["type"] == "initialized":
        response["controller_mode"] = request.get("controller_mode")
    else:
        response.update(action="KEEP", decision_mode="monetary", q_keep=1.0,
                        q_compact=2.0, predicted_post_compact_L=None)
    if behavior == "identity":
        response["epoch"] += 1
    if behavior == "protocol":
        response["protocol_version"] = "wrong"
    print(json.dumps(response, separators=(",", ":")), flush=True)
"#,
    )
    .expect("write Python bridge fixture");
    absolute(&path)
}

fn config(directory: &TempDir, behavior: &str, timeout_ms: u64) -> ContextPolicyConfig {
    ContextPolicyConfig {
        bridge_command: Some(python()),
        bridge_args: vec![
            script(directory).to_string_lossy().into_owned(),
            behavior.to_string(),
        ],
        bridge_timeout_ms: Some(timeout_ms),
        ..Default::default()
    }
}

fn request(kind: &str, id: &str, epoch: u64) -> Value {
    json!({
        "type": kind, "protocol_version": PROTOCOL_VERSION, "request_id": id,
        "run_id": "run", "task_id": "task", "replicate_id": 0,
        "thread_id": "thread", "epoch": epoch, "controller_mode": "mpc_h1",
    })
}

#[test]
fn protocol_version_is_phase8b_v1() {
    assert_eq!(PROTOCOL_VERSION, "phase8b-v1");
}

#[tokio::test]
async fn rust_starts_child_and_exchanges_h1_and_full_decisions() {
    for mode in ["mpc_h1", "mpc"] {
        let directory = TempDir::new().expect("temp dir");
        let mut bridge = ContextPolicyBridge::spawn(&config(&directory, "ok", 2_000))
            .expect("spawn bridge");
        let mut initialize = request("initialize", "init", 0);
        initialize["controller_mode"] = json!(mode);
        let initialized = bridge.exchange(&initialize).await.expect("handshake");
        assert_eq!(initialized["controller_mode"], mode);
        let decision = bridge
            .exchange(&request("decide", "decision", 0))
            .await
            .expect("decision");
        assert_eq!(decision["action"], "KEEP");
    }
}

#[tokio::test]
async fn one_child_preserves_multiple_exchange_epochs() {
    let directory = TempDir::new().expect("temp dir");
    let mut bridge = ContextPolicyBridge::spawn(&config(&directory, "ok", 2_000))
        .expect("spawn bridge");
    bridge.exchange(&request("initialize", "init", 0)).await.expect("handshake");
    assert!(bridge.exchange(&request("decide", "d0", 0)).await.is_ok());
    assert!(bridge.exchange(&request("decide", "d1", 1)).await.is_ok());
}

#[tokio::test]
async fn malformed_identity_and_protocol_responses_fail_closed() {
    for (behavior, expected) in [
        ("malformed", "malformed response JSON"),
        ("nonfinite", "malformed response JSON"),
        ("identity", "identity mismatch"),
        ("protocol", "identity mismatch: protocol_version"),
    ] {
        let directory = TempDir::new().expect("temp dir");
        let mut bridge = ContextPolicyBridge::spawn(&config(&directory, behavior, 2_000))
            .expect("spawn bridge");
        let error = bridge.exchange(&request("initialize", "init", 0)).await
            .expect_err("bad response must fail");
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[tokio::test]
async fn child_exit_and_timeout_are_infrastructure_failures() {
    for (behavior, timeout, expected) in [
        ("exit", 2_000, "stdout closed"),
        ("timeout", 20, "response timeout"),
    ] {
        let directory = TempDir::new().expect("temp dir");
        let mut bridge = ContextPolicyBridge::spawn(&config(&directory, behavior, timeout))
            .expect("spawn bridge");
        let error = bridge.exchange(&request("initialize", "init", 0)).await
            .expect_err("process failure must fail closed");
        assert!(error.to_string().contains(expected), "{error}");
        assert!(error.to_string().contains("infrastructure failure"));
    }
}
