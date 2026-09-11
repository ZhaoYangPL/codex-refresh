use codex_config::types::ContextPolicyConfig;
use codex_config::types::ContextPolicyMode;
use codex_config::types::ExternalContextPolicyStub;
use codex_models_manager::bundled_models_response;
use codex_protocol::models::BaseInstructions;
use codex_protocol::openai_models::ModelInfo;
use codex_utils_absolute_path::AbsolutePathBuf;
use pretty_assertions::assert_eq;

use super::ContextPolicyAction;
use super::ContextPolicyObservation;
use super::ContextPolicySeam;
use super::native_context_limit_enabled;
use super::needs_bridge_terminal_shutdown;
use super::needs_compact_feedback;
use super::parse_bridge_decision;
use super::uses_external_bridge;
use super::uses_formal_prefix_state_k;
use super::uses_workload_prediction;
use super::validate_config;
use crate::Prompt;

fn absolute(path: &std::path::Path) -> AbsolutePathBuf {
    AbsolutePathBuf::from_absolute_path(path).expect("temporary path should be absolute")
}

fn controlled_config(
    mode: ContextPolicyMode,
    raw_log_path: AbsolutePathBuf,
) -> ContextPolicyConfig {
    ContextPolicyConfig {
        mode,
        fixed_threshold_tokens: (mode == ContextPolicyMode::ControlledFixed).then_some(1),
        external_stub: (mode == ContextPolicyMode::ExternalStub)
            .then_some(ExternalContextPolicyStub::CompactAtEpoch),
        external_stub_compact_at_epoch: (mode == ContextPolicyMode::ExternalStub).then_some(0),
        run_id: Some("run-1".to_string()),
        task_id: Some("task-1".to_string()),
        replicate_id: Some(2),
        raw_log_path: Some(raw_log_path),
        ..Default::default()
    }
}

/// A TCP run manifest that carries only what the TCP arm actually needs.
///
/// `seed` and `recovery_artifact_id` are deliberately absent: TCP owns no
/// scenario RNG and no recovery model, so neither may be required.
fn tcp_config(raw_log_path: AbsolutePathBuf) -> ContextPolicyConfig {
    ContextPolicyConfig {
        mode: ContextPolicyMode::TcpAccumulator,
        run_id: Some("run-1".to_string()),
        task_id: Some("task-1".to_string()),
        replicate_id: Some(2),
        raw_log_path: Some(raw_log_path),
        bridge_command: Some(absolute(
            std::env::current_exe().expect("current exe").as_path(),
        )),
        bridge_timeout_ms: Some(1_000),
        controller_config_id: Some("fixture-controller-v1".to_string()),
        z_schema_version: Some("phase4-observable-v1".to_string()),
        ..Default::default()
    }
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
        .expect("the diagnostics seam tests require Python")
}

/// A minimal TCP bridge that returns a diagnostics carrier when asked for one.
///
/// It echoes the protocol identity the bridge validates, answers the
/// handshake, and then returns one KEEP decision. `behavior` selects whether
/// that decision carries diagnostics at all, so the two cases -- policy
/// supplied evidence, and no evidence offered -- are produced by the same
/// fixture rather than by two divergent ones.
fn diagnostics_script(directory: &std::path::Path) -> AbsolutePathBuf {
    let path = directory.join("diagnostics_bridge_fixture.py");
    std::fs::write(
        &path,
        r#"import json
import sys

diagnostics = {
    "controller_mode": "tcp_accumulator",
    "rent_increment": 7,
    "accumulator_before_increment": 11,
    "accumulator_after_increment": 18,
    "buy_price": 42,
    "summary_cost": 3,
    "cache_loss_diagnostic": 5,
    "predicted_postcompact_length": 900,
    "keep_feasible": True,
    "compact_feasible": True,
    "decision_mode": "tcp_accumulator",
}
behavior = sys.argv[1]
for line in sys.stdin:
    request = json.loads(line)
    response = {key: request[key] for key in (
        "protocol_version", "request_id", "run_id", "task_id",
        "replicate_id", "thread_id", "epoch")}
    if request["type"] == "initialize":
        response["type"] = "initialized"
        response["controller_mode"] = request.get("controller_mode")
    else:
        response["type"] = "decision"
        response["action"] = "KEEP"
        response["decision_mode"] = "tcp_accumulator"
        response["predicted_post_compact_L"] = None
        if behavior == "with_diagnostics":
            response["diagnostics"] = diagnostics
    print(json.dumps(response, separators=(",", ":")), flush=True)
"#,
    )
    .expect("write the diagnostics bridge fixture");
    absolute(&path)
}

/// The same payload the fixture sends, as the host should record it.
fn expected_tcp_diagnostics() -> serde_json::Value {
    serde_json::json!({
        "controller_mode": "tcp_accumulator",
        "rent_increment": 7,
        "accumulator_before_increment": 11,
        "accumulator_after_increment": 18,
        "buy_price": 42,
        "summary_cost": 3,
        "cache_loss_diagnostic": 5,
        "predicted_postcompact_length": 900,
        "keep_feasible": true,
        "compact_feasible": true,
        "decision_mode": "tcp_accumulator",
    })
}

fn diagnostics_bridge_config(
    raw_log_path: AbsolutePathBuf,
    script: AbsolutePathBuf,
    behavior: &str,
) -> ContextPolicyConfig {
    ContextPolicyConfig {
        mode: ContextPolicyMode::TcpAccumulator,
        run_id: Some("run-1".to_string()),
        task_id: Some("task-1".to_string()),
        replicate_id: Some(2),
        raw_log_path: Some(raw_log_path),
        bridge_command: Some(python()),
        bridge_args: vec![script.to_string_lossy().into_owned(), behavior.to_string()],
        bridge_timeout_ms: Some(10_000),
        controller_config_id: Some("fixture-controller-v1".to_string()),
        z_schema_version: Some("phase4-observable-v1".to_string()),
        ..Default::default()
    }
}

fn read_records(path: &std::path::Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .expect("read records")
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("valid JSONL"))
        .collect()
}

fn prompt() -> Prompt {
    Prompt {
        base_instructions: BaseInstructions {
            text: "short instructions".to_string(),
            provenance: None,
        },
        ..Default::default()
    }
}

fn model_info() -> ModelInfo {
    let mut model = bundled_models_response()
        .expect("bundled models should parse")
        .models
        .into_iter()
        .next()
        .expect("bundled model catalog should not be empty");
    model.context_window = Some(100);
    model.effective_context_window_percent = 100;
    model
}

fn observation() -> ContextPolicyObservation {
    ContextPolicyObservation::new(
        &prompt(),
        serde_json::json!({"input": [], "instructions": "short instructions"}),
        &model_info(),
        "thread-1",
        "turn-1",
        "provider-1",
    )
}

#[tokio::test]
async fn fixed_and_external_stub_share_decision_and_recording_seam() {
    let directory = tempfile::tempdir().expect("create temp dir");
    for (mode, filename) in [
        (ContextPolicyMode::ControlledFixed, "fixed.jsonl"),
        (ContextPolicyMode::ExternalStub, "stub.jsonl"),
    ] {
        let path = directory.path().join(filename);
        let mut seam = ContextPolicySeam::new(controlled_config(mode, absolute(&path)));
        let decision = seam
            .decide(0, observation())
            .await
            .expect("record decision")
            .expect("controlled decision");
        assert_eq!(decision.action, Some(ContextPolicyAction::Compact));
        seam.record_ready_to_invoke(decision, observation())
            .await
            .expect("record invocation");

        let records = std::fs::read_to_string(path).expect("read records");
        let records = records
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("valid JSONL"))
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["event"], "decision");
        assert_eq!(records[0]["action"], "COMPACT");
        assert_eq!(records[0]["compaction_reason"], "context_limit");
        assert_eq!(records[1]["event"], "ready_to_invoke");
        assert_eq!(records[1]["epoch"], 0);

        // A mode with no external policy has no policy diagnostics, and the
        // record says so rather than inventing a set. The carrier appears on
        // the decision and nowhere else, so a reader cannot mistake its
        // absence on another event for a policy that supplied nothing.
        assert_eq!(records[0]["decision_diagnostics"], serde_json::Value::Null);
        assert!(records[1].get("decision_diagnostics").is_none());
    }
}

#[tokio::test]
async fn tcp_bridge_diagnostics_reach_the_decision_record() {
    let directory = tempfile::tempdir().expect("create temp dir");
    let path = directory.path().join("tcp-diagnostics.jsonl");
    let script = diagnostics_script(directory.path());
    let mut seam = ContextPolicySeam::new(diagnostics_bridge_config(
        absolute(&path),
        script,
        "with_diagnostics",
    ));

    let decision = seam
        .decide(0, observation())
        .await
        .expect("record decision")
        .expect("tcp decision");
    assert_eq!(decision.action, Some(ContextPolicyAction::Keep));
    seam.record_ready_to_invoke(decision, observation())
        .await
        .expect("record invocation");

    let records = read_records(&path);
    assert_eq!(records.len(), 2);
    // The whole point: the paid run's plan evidence survives into the record
    // the reconstruction reads, unchanged and uninterpreted.
    assert_eq!(
        records[0]["decision_diagnostics"],
        expected_tcp_diagnostics()
    );
}

#[tokio::test]
async fn a_bridge_that_supplies_no_diagnostics_records_null_not_defaults() {
    let directory = tempfile::tempdir().expect("create temp dir");
    let path = directory.path().join("tcp-bare.jsonl");
    let script = diagnostics_script(directory.path());
    let mut seam =
        ContextPolicySeam::new(diagnostics_bridge_config(absolute(&path), script, "bare"));

    let decision = seam
        .decide(0, observation())
        .await
        .expect("record decision")
        .expect("tcp decision");
    seam.record_ready_to_invoke(decision, observation())
        .await
        .expect("record invocation");

    let records = read_records(&path);
    assert_eq!(records[0]["decision_diagnostics"], serde_json::Value::Null);
    // Not a partial object, and not the fields the host happens to know about.
    assert_eq!(records[0]["decision_mode"], "tcp_accumulator");
}

#[tokio::test]
async fn controlled_fixed_uses_formal_l_not_the_coarse_diagnostic_estimate() {
    let directory = tempfile::tempdir().expect("create temp dir");
    let path = absolute(&directory.path().join("fixed-formal-l.jsonl"));
    let mut config = controlled_config(ContextPolicyMode::ControlledFixed, path);
    config.fixed_threshold_tokens = Some(50);
    let mut seam = ContextPolicySeam::new(config);

    let mut below = observation();
    below.estimated_input_tokens = 500;
    below.formal_input_tokens = 49;
    assert_eq!(
        seam.decide(0, below)
            .await
            .expect("fixed decision")
            .expect("controlled decision")
            .action,
        Some(ContextPolicyAction::Keep)
    );

    let mut at_threshold = observation();
    at_threshold.estimated_input_tokens = 1;
    at_threshold.formal_input_tokens = 50;
    assert_eq!(
        seam.decide(1, at_threshold)
            .await
            .expect("fixed decision")
            .expect("controlled decision")
            .action,
        Some(ContextPolicyAction::Compact)
    );
}

#[test]
fn planner_emergency_is_a_typed_actionless_decision() {
    let response = serde_json::json!({
        "decision_mode": "emergency",
        "action": null,
        "predicted_post_compact_L": null,
    });
    let parsed = parse_bridge_decision(&response).expect("valid planner emergency");
    assert_eq!(parsed.action, None);
    assert_eq!(parsed.predicted, None);
    assert_eq!(parsed.mode, "emergency");
}

// --------------------------------------------------------------------------
// The policy's diagnostics are evidence, and the host is not their author
// --------------------------------------------------------------------------
//
// `decisions.jsonl` is the only record of what a paid run did. The host parses
// exactly the three fields it must act on and forwards the rest untouched, so
// that a plan can be reconstructed from the record rather than by re-running a
// policy whose state has moved on. Nothing here is TCP-specific on purpose: the
// same carrier has to hold an MPC plan's evidence without a second host change.

/// The full diagnostic payload a TCP bridge returns for a real decision.
fn tcp_bridge_diagnostics() -> serde_json::Value {
    serde_json::json!({
        "controller_mode": "tcp_accumulator",
        "rent_increment": 7,
        "accumulator_before_increment": 11,
        "accumulator_after_increment": 18,
        "buy_price": 42,
        "summary_cost": 3,
        "cache_loss_diagnostic": 5,
        "predicted_postcompact_length": 900,
        "keep_feasible": true,
        "compact_feasible": true,
        "decision_mode": "tcp_accumulator",
    })
}

#[test]
fn tcp_bridge_diagnostics_are_preserved_verbatim() {
    let diagnostics = tcp_bridge_diagnostics();
    let response = serde_json::json!({
        "type": "decision",
        "decision_mode": "tcp_accumulator",
        "action": "KEEP",
        "predicted_post_compact_L": null,
        "diagnostics": diagnostics,
    });

    let parsed = parse_bridge_decision(&response).expect("valid tcp decision");

    assert_eq!(parsed.action, Some(ContextPolicyAction::Keep));
    assert_eq!(parsed.mode, "tcp_accumulator");
    assert_eq!(parsed.diagnostics, Some(diagnostics));
}

#[test]
fn mpc_style_diagnostics_survive_without_tcp_specific_host_logic() {
    // Deliberately unlike the TCP payload: nested objects, arrays, and none of
    // the accumulator keys. If the host had learned any of TCP's vocabulary,
    // this is where it would show.
    let diagnostics = serde_json::json!({
        "q_keep": 1.5,
        "q_compact": 2.25,
        "planner": {"H": 2, "delta": 0.5, "M": 64},
        "scenarios": [{"length": 10, "weight": 0.5}, {"length": 20, "weight": 0.5}],
        "workload_estimator_version": "phase8d-v1",
        "reset_estimator_version": "phase8d-v1",
    });
    let response = serde_json::json!({
        "type": "decision",
        "decision_mode": "monetary",
        "action": "COMPACT",
        "predicted_post_compact_L": 1234,
        "diagnostics": diagnostics,
    });

    let parsed = parse_bridge_decision(&response).expect("valid mpc decision");

    assert_eq!(parsed.action, Some(ContextPolicyAction::Compact));
    assert_eq!(parsed.predicted, Some(1234));
    assert_eq!(parsed.mode, "monetary");
    assert_eq!(parsed.diagnostics, Some(diagnostics));
}

#[test]
fn a_bridge_response_without_diagnostics_carries_none() {
    let response = serde_json::json!({
        "type": "decision",
        "decision_mode": "monetary",
        "action": "KEEP",
        "predicted_post_compact_L": null,
    });

    let parsed = parse_bridge_decision(&response).expect("valid decision");

    assert_eq!(
        parsed.diagnostics, None,
        "an absent carrier must not be fabricated"
    );
}

#[test]
fn mandatory_fields_stay_fail_closed_when_diagnostics_are_present() {
    // A rich diagnostic payload must not buy a malformed decision a pass.
    for response in [
        serde_json::json!({
            "action": "KEEP", "diagnostics": tcp_bridge_diagnostics(),
        }),
        serde_json::json!({
            "decision_mode": "monetary", "action": "SIDEWAYS",
            "diagnostics": tcp_bridge_diagnostics(),
        }),
        serde_json::json!({
            "decision_mode": "monetary", "action": "COMPACT",
            "predicted_post_compact_L": null,
            "diagnostics": tcp_bridge_diagnostics(),
        }),
    ] {
        assert!(
            parse_bridge_decision(&response).is_err(),
            "a decision missing a mandatory field was accepted: {response}"
        );
    }
}

#[tokio::test]
async fn native_fixed_bypasses_the_controlled_seam_and_logging() {
    let directory = tempfile::tempdir().expect("create temp dir");
    let path = directory.path().join("native.jsonl");
    let config = ContextPolicyConfig {
        raw_log_path: Some(absolute(&path)),
        ..Default::default()
    };
    let mut seam = ContextPolicySeam::new(config.clone());

    assert_eq!(
        seam.decide(0, observation())
            .await
            .expect("native decision"),
        None
    );
    assert!(native_context_limit_enabled(&config));
    assert!(!path.exists());
}

#[test]
fn controlled_configuration_is_fail_closed() {
    let directory = tempfile::tempdir().expect("create temp dir");
    let path = absolute(&directory.path().join("policy.jsonl"));
    let controlled = controlled_config(ContextPolicyMode::ControlledFixed, path.clone());
    assert!(validate_config(&controlled, /*token_budget_enabled*/ false).is_ok());
    assert!(validate_config(&controlled, /*token_budget_enabled*/ true).is_err());

    let missing_identity = ContextPolicyConfig {
        mode: ContextPolicyMode::ControlledFixed,
        fixed_threshold_tokens: Some(10),
        raw_log_path: Some(path),
        ..Default::default()
    };
    assert!(validate_config(&missing_identity, /*token_budget_enabled*/ false).is_err());

    let request_log_without_identity = ContextPolicyConfig {
        request_raw_log_path: controlled.raw_log_path,
        ..Default::default()
    };
    assert!(
        validate_config(
            &request_log_without_identity,
            /*token_budget_enabled*/ false
        )
        .is_err()
    );

    let schedule_without_request_log = ContextPolicyConfig {
        pricing_schedule_id: Some("fixture-price-v1".to_string()),
        ..Default::default()
    };
    assert!(
        validate_config(
            &schedule_without_request_log,
            /*token_budget_enabled*/ false
        )
        .is_err()
    );
}

#[test]
fn tcp_shares_bridge_capabilities_but_is_not_an_mpc_mode() {
    for mode in [
        ContextPolicyMode::TcpAccumulator,
        ContextPolicyMode::MpcH1,
        ContextPolicyMode::Mpc,
    ] {
        assert!(uses_external_bridge(mode), "{mode:?} uses the bridge");
        assert!(needs_compact_feedback(mode), "{mode:?} needs feedback");
        assert!(
            needs_bridge_terminal_shutdown(mode),
            "{mode:?} needs teardown"
        );
        assert!(
            uses_formal_prefix_state_k(mode),
            "{mode:?} observes the formal reusable prefix"
        );
    }

    // Only the MPC arms predict future workload.
    assert!(!uses_workload_prediction(ContextPolicyMode::TcpAccumulator));
    assert!(uses_workload_prediction(ContextPolicyMode::MpcH1));
    assert!(uses_workload_prediction(ContextPolicyMode::Mpc));

    for mode in [
        ContextPolicyMode::NativeFixed,
        ContextPolicyMode::ControlledFixed,
        ContextPolicyMode::ExternalStub,
    ] {
        assert!(!uses_external_bridge(mode), "{mode:?} is in-process");
        assert!(!needs_compact_feedback(mode));
        assert!(!needs_bridge_terminal_shutdown(mode));
        assert!(!uses_formal_prefix_state_k(mode));
        assert!(!uses_workload_prediction(mode));
    }
}

#[test]
fn tcp_accumulator_bridge_configuration_is_fail_closed() {
    let directory = tempfile::tempdir().expect("create temp dir");
    let path = absolute(&directory.path().join("tcp.jsonl"));
    let valid = tcp_config(path);
    assert!(validate_config(&valid, /*token_budget_enabled*/ false).is_ok());
    assert!(validate_config(&valid, /*token_budget_enabled*/ true).is_err());

    let cases: Vec<(&str, ContextPolicyConfig)> = vec![
        (
            "bridge_command",
            ContextPolicyConfig {
                bridge_command: None,
                ..valid.clone()
            },
        ),
        (
            "controller_config_id",
            ContextPolicyConfig {
                controller_config_id: None,
                ..valid.clone()
            },
        ),
        (
            "z_schema_version",
            ContextPolicyConfig {
                z_schema_version: Some("invented-v1".to_string()),
                ..valid.clone()
            },
        ),
        (
            "timeout",
            ContextPolicyConfig {
                bridge_timeout_ms: Some(0),
                ..valid.clone()
            },
        ),
        (
            "fixed_threshold_tokens",
            ContextPolicyConfig {
                fixed_threshold_tokens: Some(10),
                ..valid.clone()
            },
        ),
        (
            "external_stub",
            ContextPolicyConfig {
                external_stub: Some(ExternalContextPolicyStub::AlwaysKeep),
                ..valid
            },
        ),
    ];
    for (name, config) in cases {
        assert!(
            validate_config(&config, /*token_budget_enabled*/ false).is_err(),
            "tcp_accumulator must reject an incomplete/inconsistent config ({name})"
        );
    }
}

#[test]
fn tcp_accumulator_does_not_require_mpc_only_provenance() {
    let directory = tempfile::tempdir().expect("create temp dir");
    let path = absolute(&directory.path().join("tcp-provenance.jsonl"));
    let tcp = tcp_config(path);
    assert!(tcp.seed.is_none() && tcp.recovery_artifact_id.is_none());

    // A shared run manifest may still carry both; they stay provenance only.
    let carrying = ContextPolicyConfig {
        seed: Some(7),
        recovery_artifact_id: Some("fixture-recovery-v1".to_string()),
        ..tcp.clone()
    };
    assert!(validate_config(&carrying, /*token_budget_enabled*/ false).is_ok());

    // The relaxation is TCP-specific: an MPC arm still requires both.
    let mpc_base = ContextPolicyConfig {
        mode: ContextPolicyMode::MpcH1,
        seed: Some(7),
        recovery_artifact_id: Some("fixture-recovery-v1".to_string()),
        ..tcp
    };
    assert!(validate_config(&mpc_base, /*token_budget_enabled*/ false).is_ok());
    for (name, config) in [
        (
            "seed",
            ContextPolicyConfig {
                seed: None,
                ..mpc_base.clone()
            },
        ),
        (
            "recovery_artifact_id",
            ContextPolicyConfig {
                recovery_artifact_id: None,
                ..mpc_base
            },
        ),
    ] {
        assert!(
            validate_config(&config, /*token_budget_enabled*/ false).is_err(),
            "MPC bridge configuration must still require {name}"
        );
    }
}
