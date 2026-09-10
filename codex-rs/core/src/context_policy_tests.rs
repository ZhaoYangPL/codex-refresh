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
use super::parse_bridge_decision;
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
    }
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
    let (action, predicted, mode) =
        parse_bridge_decision(&response).expect("valid planner emergency");
    assert_eq!(action, None);
    assert_eq!(predicted, None);
    assert_eq!(mode, "emergency");
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
