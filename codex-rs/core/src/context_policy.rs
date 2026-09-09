//! Canonical context-policy seam and the Phase 8B local MPC adapter.

use std::io;

use codex_config::types::{ContextPolicyConfig, ContextPolicyMode, ExternalContextPolicyStub};
use codex_protocol::openai_models::ModelInfo;
use codex_utils_cache::sha1_digest;
use codex_utils_string::approx_token_count;
use serde_json::Value;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;

use crate::Prompt;
use crate::context_manager::estimate_item_token_count;
use crate::context_policy_bridge::{ContextPolicyBridge, PROTOCOL_VERSION};

const PHASE8A_PROTOCOL_VERSION: &str = "phase8a-v1";
const L_KIND: &str = "codex_model_visible_components_v1";
const K_KIND_PRIOR: &str = "codex_pre_request_common_prefix_v1";
const K_KIND_NO_PRIOR: &str = "codex_pre_request_no_prior_v1";
const Z_SCHEMA_VERSION: &str = "phase4-observable-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(crate) enum ContextPolicyAction {
    Keep,
    Compact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContextPolicyDecision {
    pub(crate) epoch: u64,
    pub(crate) action: ContextPolicyAction,
    pre_action_l: i64,
    predicted_post_compact_l: Option<i64>,
}

#[derive(Clone)]
pub(crate) struct ContextPolicyObservation {
    estimated_input_tokens: i64,
    formal_input_tokens: i64,
    reusable_prefix_tokens: i64,
    reusable_prefix_kind: &'static str,
    prior_request_hash: Option<String>,
    nominal_context_window: Option<i64>,
    usable_context_window: Option<i64>,
    input_item_count: usize,
    model_visible_tool_count: usize,
    model_id: String,
    provider_id: String,
    thread_id: String,
    turn_id: String,
    request_hash: String,
    model_visible_request: Value,
}

impl ContextPolicyObservation {
    pub(crate) fn new(
        prompt: &Prompt,
        model_visible_request: Value,
        model_info: &ModelInfo,
        thread_id: &str,
        turn_id: &str,
        provider_id: &str,
    ) -> Self {
        let request_bytes = serde_json::to_vec(&model_visible_request).unwrap_or_default();
        Self {
            estimated_input_tokens: i64::try_from(approx_token_count(
                &model_visible_request.to_string(),
            ))
            .unwrap_or(i64::MAX),
            formal_input_tokens: formal_request_tokens(prompt),
            reusable_prefix_tokens: 0,
            reusable_prefix_kind: K_KIND_NO_PRIOR,
            prior_request_hash: None,
            nominal_context_window: model_info.context_window,
            usable_context_window: model_info.usable_context_window(),
            input_item_count: prompt.input.len(),
            model_visible_tool_count: prompt.tools.len(),
            model_id: model_info.slug.clone(),
            provider_id: provider_id.to_string(),
            thread_id: thread_id.to_string(),
            turn_id: turn_id.to_string(),
            request_hash: hex_sha1(&request_bytes),
            model_visible_request,
        }
    }

    fn with_prefix(mut self, previous: Option<&Self>) -> Self {
        if let Some(previous) = previous
            && previous.model_id == self.model_id
            && previous.provider_id == self.provider_id
        {
            let old = serde_json::to_vec(&previous.model_visible_request).unwrap_or_default();
            let new = serde_json::to_vec(&self.model_visible_request).unwrap_or_default();
            let matched = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
            let boundary = floor_utf8_boundary(&new, matched);
            let common = String::from_utf8_lossy(&new[..boundary]);
            self.reusable_prefix_tokens = i64::try_from(approx_token_count(&common))
                .unwrap_or(i64::MAX)
                .min(self.formal_input_tokens);
            self.reusable_prefix_kind = K_KIND_PRIOR;
            self.prior_request_hash = Some(previous.request_hash.clone());
        }
        self
    }
}

pub(crate) fn validate_config(
    config: &ContextPolicyConfig,
    token_budget_enabled: bool,
) -> io::Result<()> {
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
    if config.mode == ContextPolicyMode::NativeFixed {
        return Ok(());
    }
    if token_budget_enabled {
        return Err(invalid(
            "controlled context policy modes require TokenBudget to be disabled",
        ));
    }
    if config.run_id.as_deref().is_none_or(str::is_empty)
        || config.task_id.as_deref().is_none_or(str::is_empty)
        || config.replicate_id.is_none()
        || config.raw_log_path.is_none()
    {
        return Err(invalid(
            "controlled context policy modes require run_id, task_id, replicate_id, and raw_log_path",
        ));
    }
    match config.mode {
        ContextPolicyMode::NativeFixed => unreachable!(),
        ContextPolicyMode::ControlledFixed => {
            if config.fixed_threshold_tokens.is_none_or(|value| value <= 0) {
                return Err(invalid("controlled_fixed requires fixed_threshold_tokens"));
            }
            if config.external_stub.is_some() {
                return Err(invalid("controlled_fixed cannot include external stub settings"));
            }
        }
        ContextPolicyMode::ExternalStub => match config.external_stub {
            Some(ExternalContextPolicyStub::AlwaysKeep)
                if config.external_stub_compact_at_epoch.is_none() => {}
            Some(ExternalContextPolicyStub::CompactAtEpoch)
                if config.external_stub_compact_at_epoch.is_some() => {}
            _ => return Err(invalid("external_stub settings are incomplete or inconsistent")),
        },
        ContextPolicyMode::MpcH1 | ContextPolicyMode::Mpc => {
            let identities = [
                config.controller_config_id.as_deref(),
                config.recovery_artifact_id.as_deref(),
                config.z_schema_version.as_deref(),
            ];
            if config.bridge_command.is_none()
                || config.seed.is_none()
                || config.bridge_timeout_ms.is_none_or(|value| value == 0)
                || identities.into_iter().any(|value| value.is_none_or(str::is_empty))
                || config.z_schema_version.as_deref() != Some(Z_SCHEMA_VERSION)
            {
                return Err(invalid("MPC bridge configuration is incomplete"));
            }
            if config.fixed_threshold_tokens.is_some() || config.external_stub.is_some() {
                return Err(invalid("MPC cannot include fixed/stub policy settings"));
            }
        }
    }
    Ok(())
}

pub(crate) fn native_context_limit_enabled(config: &ContextPolicyConfig) -> bool {
    config.mode == ContextPolicyMode::NativeFixed
}

pub(crate) struct ContextPolicySeam {
    config: ContextPolicyConfig,
    log: Option<File>,
    bridge: Option<ContextPolicyBridge>,
    bridge_initialized: bool,
    next_request_id: u64,
    previous_ready: Option<ContextPolicyObservation>,
    previous_action: Option<ContextPolicyAction>,
}

impl ContextPolicySeam {
    pub(crate) fn new(config: ContextPolicyConfig) -> Self {
        Self {
            config,
            log: None,
            bridge: None,
            bridge_initialized: false,
            next_request_id: 0,
            previous_ready: None,
            previous_action: None,
        }
    }

    pub(crate) fn is_controlled(&self) -> bool {
        self.config.mode != ContextPolicyMode::NativeFixed
    }

    fn is_mpc(&self) -> bool {
        matches!(self.config.mode, ContextPolicyMode::MpcH1 | ContextPolicyMode::Mpc)
    }

    pub(crate) async fn decide(
        &mut self,
        epoch: u64,
        observation: ContextPolicyObservation,
    ) -> io::Result<Option<ContextPolicyDecision>> {
        if !self.is_controlled() {
            return Ok(None);
        }
        let observation = observation.with_prefix(self.previous_ready.as_ref());
        let (action, predicted) = match self.config.mode {
            ContextPolicyMode::NativeFixed => unreachable!(),
            ContextPolicyMode::ControlledFixed => (
                if observation.estimated_input_tokens
                    >= self.config.fixed_threshold_tokens.unwrap_or(i64::MAX)
                {
                    ContextPolicyAction::Compact
                } else {
                    ContextPolicyAction::Keep
                },
                None,
            ),
            ContextPolicyMode::ExternalStub => (
                match self.config.external_stub {
                    Some(ExternalContextPolicyStub::CompactAtEpoch)
                        if self.config.external_stub_compact_at_epoch == Some(epoch) =>
                    {
                        ContextPolicyAction::Compact
                    }
                    _ => ContextPolicyAction::Keep,
                },
                None,
            ),
            ContextPolicyMode::MpcH1 | ContextPolicyMode::Mpc => {
                self.initialize_bridge(epoch, &observation).await?;
                self.send_transition(epoch, &observation).await?;
                let window = observation
                    .usable_context_window
                    .ok_or_else(|| infrastructure("usable W missing"))?;
                let request = self.message(
                    "decide",
                    epoch,
                    &observation,
                    serde_json::json!({
                        "state": {"L": observation.formal_input_tokens, "K": observation.reusable_prefix_tokens,
                            "W": window, "z_schema_version": Z_SCHEMA_VERSION, "z": {}},
                        "measurement": {"L_kind": L_KIND, "K_kind": observation.reusable_prefix_kind,
                            "context_request_hash": observation.request_hash,
                            "prior_request_hash": observation.prior_request_hash},
                        "model_id": observation.model_id, "provider_id": observation.provider_id,
                    }),
                );
                let response = self.exchange(&request).await?;
                if response.get("type").and_then(Value::as_str) != Some("decision") {
                    return Err(infrastructure("expected decision response"));
                }
                let action = match response.get("action").and_then(Value::as_str) {
                    Some("KEEP") => ContextPolicyAction::Keep,
                    Some("COMPACT") => ContextPolicyAction::Compact,
                    _ => return Err(infrastructure("invalid or emergency action")),
                };
                let predicted = response.get("predicted_post_compact_L").and_then(Value::as_i64);
                if action == ContextPolicyAction::Compact && predicted.is_none() {
                    return Err(infrastructure("COMPACT response lacks reset prediction"));
                }
                (action, predicted)
            }
        };
        let decision = ContextPolicyDecision {
            epoch,
            action,
            pre_action_l: observation.formal_input_tokens,
            predicted_post_compact_l: predicted,
        };
        let record = self.record("decision", decision, &observation);
        self.append(&record).await?;
        Ok(Some(decision))
    }

    pub(crate) async fn record_ready_to_invoke(
        &mut self,
        decision: ContextPolicyDecision,
        observation: ContextPolicyObservation,
    ) -> io::Result<()> {
        let observation = observation.with_prefix(self.previous_ready.as_ref());
        if self.is_mpc() && decision.action == ContextPolicyAction::Compact {
            let request = self.message(
                "compact_feedback",
                decision.epoch,
                &observation,
                serde_json::json!({"feedback": {
                    "pre_compact_L": decision.pre_action_l,
                    "predicted_post_compact_L": decision.predicted_post_compact_l,
                    "observed_post_compact_L": observation.formal_input_tokens,
                    "post_request_hash": observation.request_hash,
                    "measurement_kind": L_KIND,
                }}),
            );
            let response = self.exchange(&request).await?;
            if response.get("type").and_then(Value::as_str)
                != Some("compact_feedback_accepted")
            {
                return Err(infrastructure("compact feedback was not accepted"));
            }
        }
        let record = self.record("ready_to_invoke", decision, &observation);
        self.append(&record).await?;
        self.previous_ready = Some(observation);
        self.previous_action = Some(decision.action);
        Ok(())
    }

    pub(crate) async fn terminal(&mut self, epoch: u64, reason: &str) -> io::Result<()> {
        if !self.is_mpc() || !self.bridge_initialized {
            return Ok(());
        }
        let observation = self
            .previous_ready
            .clone()
            .ok_or_else(|| infrastructure("terminal without prior invocation"))?;
        let request = self.message("terminal", epoch, &observation, serde_json::json!({"reason": reason}));
        let response = self.exchange(&request).await?;
        if response.get("type").and_then(Value::as_str) != Some("terminated") {
            return Err(infrastructure("terminal was not accepted"));
        }
        let request = self.message("shutdown", epoch, &observation, serde_json::json!({}));
        let response = self.exchange(&request).await?;
        if response.get("type").and_then(Value::as_str) != Some("shutdown_complete") {
            return Err(infrastructure("shutdown was not accepted"));
        }
        Ok(())
    }

    async fn initialize_bridge(
        &mut self,
        epoch: u64,
        observation: &ContextPolicyObservation,
    ) -> io::Result<()> {
        if self.bridge_initialized {
            return Ok(());
        }
        self.bridge = Some(ContextPolicyBridge::spawn(&self.config)?);
        let mode = match self.config.mode {
            ContextPolicyMode::MpcH1 => "mpc_h1",
            ContextPolicyMode::Mpc => "mpc",
            _ => unreachable!(),
        };
        let request = self.message(
            "initialize",
            epoch,
            observation,
            serde_json::json!({"controller_mode": mode, "model_id": observation.model_id,
                "provider_id": observation.provider_id, "seed": self.config.seed,
                "controller_config_id": self.config.controller_config_id,
                "recovery_artifact_id": self.config.recovery_artifact_id,
                "z_schema_version": self.config.z_schema_version}),
        );
        let response = self.exchange(&request).await?;
        if response.get("type").and_then(Value::as_str) != Some("initialized")
            || response.get("controller_mode").and_then(Value::as_str) != Some(mode)
        {
            return Err(infrastructure("bridge handshake mismatch"));
        }
        self.bridge_initialized = true;
        Ok(())
    }

    async fn send_transition(
        &mut self,
        epoch: u64,
        current: &ContextPolicyObservation,
    ) -> io::Result<()> {
        let Some(previous) = self.previous_ready.clone() else {
            return Ok(());
        };
        let from_epoch = epoch
            .checked_sub(1)
            .ok_or_else(|| infrastructure("prior request exists at epoch zero"))?;
        let additional = current
            .formal_input_tokens
            .checked_sub(previous.formal_input_tokens)
            .ok_or_else(|| infrastructure("negative realized context growth"))?;
        let action = match self.previous_action {
            Some(ContextPolicyAction::Keep) => "KEEP",
            Some(ContextPolicyAction::Compact) => "COMPACT",
            None => return Err(infrastructure("previous action missing")),
        };
        let request = self.message(
            "observe_transition",
            epoch,
            current,
            serde_json::json!({"transition": {"from_epoch": from_epoch, "to_epoch": epoch,
                "previous_action": action, "previous_post_action_length": previous.formal_input_tokens,
                "current_pre_action_length": current.formal_input_tokens, "output_tokens": 0,
                "additional_tokens": additional, "continued": true, "measurement_kind": L_KIND}}),
        );
        let response = self.exchange(&request).await?;
        if response.get("type").and_then(Value::as_str) != Some("transition_observed") {
            return Err(infrastructure("transition was not accepted"));
        }
        Ok(())
    }

    fn message(
        &mut self,
        kind: &str,
        epoch: u64,
        observation: &ContextPolicyObservation,
        body: Value,
    ) -> Value {
        let request_id = format!("{}:{}", self.run_id(), self.next_request_id);
        self.next_request_id = self.next_request_id.saturating_add(1);
        let mut message = serde_json::json!({"type": kind, "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id, "run_id": self.run_id(), "task_id": self.task_id(),
            "replicate_id": self.replicate_id(), "thread_id": observation.thread_id, "epoch": epoch});
        if let (Some(target), Some(source)) = (message.as_object_mut(), body.as_object()) {
            target.extend(source.clone());
        }
        message
    }

    async fn exchange(&mut self, request: &Value) -> io::Result<Value> {
        self.bridge
            .as_mut()
            .ok_or_else(|| infrastructure("bridge is not running"))?
            .exchange(request)
            .await
    }

    fn record(&self, event: &str, decision: ContextPolicyDecision, observation: &ContextPolicyObservation) -> Value {
        serde_json::json!({"event": event,
            "protocol_version": if self.is_mpc() { PROTOCOL_VERSION } else { PHASE8A_PROTOCOL_VERSION },
            "run_id": self.run_id(), "task_id": self.task_id(), "replicate_id": self.replicate_id(),
            "thread_id": observation.thread_id, "turn_id": observation.turn_id, "epoch": decision.epoch,
            "policy_mode": self.config.mode, "action": decision.action,
            "compaction_reason": (decision.action == ContextPolicyAction::Compact).then_some("context_limit"),
            "estimated_input_tokens": observation.estimated_input_tokens,
            "formal_input_tokens": observation.formal_input_tokens,
            "reusable_prefix_tokens": self.is_mpc().then_some(observation.reusable_prefix_tokens),
            "nominal_context_window": observation.nominal_context_window,
            "usable_context_window": observation.usable_context_window,
            "input_item_count": observation.input_item_count,
            "model_visible_tool_count": observation.model_visible_tool_count,
            "model_id": observation.model_id, "provider_id": observation.provider_id,
            "measurement": "codex_coarse_prompt_estimate_v1", "L_measurement_kind": L_KIND,
            "K_measurement_kind": observation.reusable_prefix_kind,
            "model_visible_request_hash": observation.request_hash,
            "prior_request_hash": observation.prior_request_hash,
            "z_schema_version": Z_SCHEMA_VERSION, "z": {},
            "model_visible_request": observation.model_visible_request})
    }

    fn run_id(&self) -> &str { self.config.run_id.as_deref().unwrap_or_default() }
    fn task_id(&self) -> &str { self.config.task_id.as_deref().unwrap_or_default() }
    fn replicate_id(&self) -> u64 { self.config.replicate_id.unwrap_or_default() }

    async fn append(&mut self, record: &Value) -> io::Result<()> {
        if self.log.is_none() {
            let path = self.config.raw_log_path.as_ref()
                .ok_or_else(|| io::Error::other("controlled policy log path is missing"))?;
            self.log = Some(OpenOptions::new().create(true).append(true).open(path.as_path()).await?);
        }
        let mut line = serde_json::to_vec(record).map_err(io::Error::other)?;
        line.push(b'\n');
        if let Some(log) = self.log.as_mut() {
            log.write_all(&line).await?;
            log.flush().await?;
        }
        Ok(())
    }
}

fn formal_request_tokens(prompt: &Prompt) -> i64 {
    let base = i64::try_from(approx_token_count(&prompt.base_instructions.text)).unwrap_or(i64::MAX);
    let input = prompt.input.iter().map(estimate_item_token_count).fold(0_i64, i64::saturating_add);
    let tools = serde_json::to_string(&prompt.tools)
        .map(|value| i64::try_from(approx_token_count(&value)).unwrap_or(i64::MAX))
        .unwrap_or(i64::MAX);
    let output = prompt.output_schema.as_ref().and_then(|schema| serde_json::to_string(schema).ok())
        .map_or(0, |value| i64::try_from(approx_token_count(&value)).unwrap_or(i64::MAX));
    base.saturating_add(input).saturating_add(tools).saturating_add(output)
}

fn floor_utf8_boundary(bytes: &[u8], mut index: usize) -> usize {
    while index > 0 && std::str::from_utf8(&bytes[..index]).is_err() { index -= 1; }
    index
}

fn hex_sha1(bytes: &[u8]) -> String {
    sha1_digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn infrastructure(message: impl Into<String>) -> io::Error {
    io::Error::other(format!("context policy infrastructure failure: {}", message.into()))
}

#[cfg(test)]
#[path = "context_policy_tests.rs"]
mod tests;
