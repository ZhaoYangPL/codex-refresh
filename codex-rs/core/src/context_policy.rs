//! Thin Phase 8A context-limit policy seam and reconstruction log.

use std::io;

use codex_config::types::ContextPolicyConfig;
use codex_config::types::ContextPolicyMode;
use codex_config::types::ExternalContextPolicyStub;
use codex_protocol::openai_models::ModelInfo;
use codex_utils_string::approx_token_count;
use serde::Serialize;
use tokio::fs::File;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;

use crate::Prompt;

const PROTOCOL_VERSION: &str = "phase8a-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(crate) enum ContextPolicyAction {
    Keep,
    Compact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContextPolicyDecision {
    pub(crate) epoch: u64,
    pub(crate) action: ContextPolicyAction,
}

pub(crate) struct ContextPolicyObservation {
    estimated_input_tokens: i64,
    nominal_context_window: Option<i64>,
    usable_context_window: Option<i64>,
    input_item_count: usize,
    model_visible_tool_count: usize,
    model_id: String,
    provider_id: String,
    thread_id: String,
    turn_id: String,
    model_visible_request: serde_json::Value,
}

impl ContextPolicyObservation {
    pub(crate) fn new(
        prompt: &Prompt,
        model_visible_request: serde_json::Value,
        model_info: &ModelInfo,
        thread_id: &str,
        turn_id: &str,
        provider_id: &str,
    ) -> Self {
        Self {
            estimated_input_tokens: estimated_request_tokens(&model_visible_request),
            nominal_context_window: model_info.context_window,
            usable_context_window: model_info.usable_context_window(),
            input_item_count: prompt.input.len(),
            model_visible_tool_count: prompt.tools.len(),
            model_id: model_info.slug.clone(),
            provider_id: provider_id.to_string(),
            thread_id: thread_id.to_string(),
            turn_id: turn_id.to_string(),
            model_visible_request,
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum ContextPolicyRecord {
    Decision {
        protocol_version: &'static str,
        run_id: String,
        task_id: String,
        replicate_id: u64,
        thread_id: String,
        turn_id: String,
        epoch: u64,
        policy_mode: ContextPolicyMode,
        action: ContextPolicyAction,
        compaction_reason: Option<&'static str>,
        estimated_input_tokens: i64,
        reusable_prefix_tokens: Option<i64>,
        nominal_context_window: Option<i64>,
        usable_context_window: Option<i64>,
        input_item_count: usize,
        model_visible_tool_count: usize,
        model_id: String,
        provider_id: String,
        measurement: &'static str,
        model_visible_request: serde_json::Value,
    },
    ReadyToInvoke {
        protocol_version: &'static str,
        run_id: String,
        task_id: String,
        replicate_id: u64,
        thread_id: String,
        turn_id: String,
        epoch: u64,
        policy_mode: ContextPolicyMode,
        action: ContextPolicyAction,
        estimated_input_tokens: i64,
        reusable_prefix_tokens: Option<i64>,
        nominal_context_window: Option<i64>,
        usable_context_window: Option<i64>,
        input_item_count: usize,
        model_visible_tool_count: usize,
        model_id: String,
        provider_id: String,
        measurement: &'static str,
        model_visible_request: serde_json::Value,
    },
}

pub(crate) fn validate_config(
    config: &ContextPolicyConfig,
    token_budget_enabled: bool,
) -> io::Result<()> {
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
    match config.mode {
        ContextPolicyMode::NativeFixed => Ok(()),
        ContextPolicyMode::ControlledFixed | ContextPolicyMode::ExternalStub => {
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
                ContextPolicyMode::ControlledFixed => {
                    if config.fixed_threshold_tokens.is_none_or(|value| value <= 0) {
                        return Err(invalid(
                            "controlled_fixed requires a positive fixed_threshold_tokens",
                        ));
                    }
                    if config.external_stub.is_some()
                        || config.external_stub_compact_at_epoch.is_some()
                    {
                        return Err(invalid(
                            "controlled_fixed cannot include external stub settings",
                        ));
                    }
                }
                ContextPolicyMode::ExternalStub => {
                    if config.fixed_threshold_tokens.is_some() {
                        return Err(invalid(
                            "external_stub cannot include fixed_threshold_tokens",
                        ));
                    }
                    match config.external_stub {
                        Some(ExternalContextPolicyStub::AlwaysKeep) => {
                            if config.external_stub_compact_at_epoch.is_some() {
                                return Err(invalid(
                                    "always_keep cannot include external_stub_compact_at_epoch",
                                ));
                            }
                        }
                        Some(ExternalContextPolicyStub::CompactAtEpoch) => {
                            if config.external_stub_compact_at_epoch.is_none() {
                                return Err(invalid(
                                    "compact_at_epoch requires external_stub_compact_at_epoch",
                                ));
                            }
                        }
                        None => {
                            return Err(invalid(
                                "external_stub requires an explicit external_stub policy",
                            ));
                        }
                    }
                }
                ContextPolicyMode::NativeFixed => unreachable!(),
            }
            Ok(())
        }
    }
}

pub(crate) fn native_context_limit_enabled(config: &ContextPolicyConfig) -> bool {
    config.mode == ContextPolicyMode::NativeFixed
}

pub(crate) struct ContextPolicySeam {
    config: ContextPolicyConfig,
    log: Option<File>,
}

impl ContextPolicySeam {
    pub(crate) fn new(config: ContextPolicyConfig) -> Self {
        Self { config, log: None }
    }

    pub(crate) fn is_controlled(&self) -> bool {
        self.config.mode != ContextPolicyMode::NativeFixed
    }

    pub(crate) async fn decide(
        &mut self,
        epoch: u64,
        observation: ContextPolicyObservation,
    ) -> io::Result<Option<ContextPolicyDecision>> {
        if self.config.mode == ContextPolicyMode::NativeFixed {
            return Ok(None);
        }

        let estimated_input_tokens = observation.estimated_input_tokens;
        let action = match self.config.mode {
            ContextPolicyMode::NativeFixed => unreachable!(),
            ContextPolicyMode::ControlledFixed => {
                if estimated_input_tokens >= self.config.fixed_threshold_tokens.unwrap_or(i64::MAX)
                {
                    ContextPolicyAction::Compact
                } else {
                    ContextPolicyAction::Keep
                }
            }
            ContextPolicyMode::ExternalStub => match self.config.external_stub {
                Some(ExternalContextPolicyStub::AlwaysKeep) => ContextPolicyAction::Keep,
                Some(ExternalContextPolicyStub::CompactAtEpoch)
                    if self.config.external_stub_compact_at_epoch == Some(epoch) =>
                {
                    ContextPolicyAction::Compact
                }
                Some(ExternalContextPolicyStub::CompactAtEpoch) | None => ContextPolicyAction::Keep,
            },
        };
        let decision = ContextPolicyDecision { epoch, action };
        let record = ContextPolicyRecord::Decision {
            protocol_version: PROTOCOL_VERSION,
            run_id: self.run_id().to_string(),
            task_id: self.task_id().to_string(),
            replicate_id: self.replicate_id(),
            thread_id: observation.thread_id,
            turn_id: observation.turn_id,
            epoch,
            policy_mode: self.config.mode,
            action,
            compaction_reason: (action == ContextPolicyAction::Compact).then_some("context_limit"),
            estimated_input_tokens,
            reusable_prefix_tokens: None,
            nominal_context_window: observation.nominal_context_window,
            usable_context_window: observation.usable_context_window,
            input_item_count: observation.input_item_count,
            model_visible_tool_count: observation.model_visible_tool_count,
            model_id: observation.model_id,
            provider_id: observation.provider_id,
            measurement: "codex_coarse_prompt_estimate_v1",
            model_visible_request: observation.model_visible_request,
        };
        self.append(&record).await?;
        Ok(Some(decision))
    }

    pub(crate) async fn record_ready_to_invoke(
        &mut self,
        decision: ContextPolicyDecision,
        observation: ContextPolicyObservation,
    ) -> io::Result<()> {
        if self.config.mode == ContextPolicyMode::NativeFixed {
            return Ok(());
        }
        let record = ContextPolicyRecord::ReadyToInvoke {
            protocol_version: PROTOCOL_VERSION,
            run_id: self.run_id().to_string(),
            task_id: self.task_id().to_string(),
            replicate_id: self.replicate_id(),
            thread_id: observation.thread_id,
            turn_id: observation.turn_id,
            epoch: decision.epoch,
            policy_mode: self.config.mode,
            action: decision.action,
            estimated_input_tokens: observation.estimated_input_tokens,
            reusable_prefix_tokens: None,
            nominal_context_window: observation.nominal_context_window,
            usable_context_window: observation.usable_context_window,
            input_item_count: observation.input_item_count,
            model_visible_tool_count: observation.model_visible_tool_count,
            model_id: observation.model_id,
            provider_id: observation.provider_id,
            measurement: "codex_coarse_prompt_estimate_v1",
            model_visible_request: observation.model_visible_request,
        };
        self.append(&record).await
    }

    fn run_id(&self) -> &str {
        self.config.run_id.as_deref().unwrap_or_default()
    }

    fn task_id(&self) -> &str {
        self.config.task_id.as_deref().unwrap_or_default()
    }

    fn replicate_id(&self) -> u64 {
        self.config.replicate_id.unwrap_or_default()
    }

    async fn append(&mut self, record: &ContextPolicyRecord) -> io::Result<()> {
        if self.log.is_none() {
            let path = self
                .config
                .raw_log_path
                .as_ref()
                .ok_or_else(|| io::Error::other("controlled policy log path is missing"))?;
            self.log = Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path.as_path())
                    .await?,
            );
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

fn estimated_request_tokens(model_visible_request: &serde_json::Value) -> i64 {
    i64::try_from(approx_token_count(&model_visible_request.to_string())).unwrap_or(i64::MAX)
}

#[cfg(test)]
#[path = "context_policy_tests.rs"]
mod tests;
