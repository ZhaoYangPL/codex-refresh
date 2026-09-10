//! Append-only raw request evidence for deterministic Phase 8C accounting.
//!
//! This module deliberately records provider facts without pricing them.  The
//! research repository owns canonical normalization and monetary accounting.

use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use codex_config::types::ContextPolicyConfig;
use codex_config::types::ContextPolicyMode;
use codex_protocol::protocol::TokenUsage;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

pub(crate) const RAW_REQUEST_EVENT_SCHEMA_VERSION: &str = "codex-request-events-v1";

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
// Phase 8C executes serve/summary only; the versioned schema reserves the
// remaining real request classes without fabricating any such lifecycle.
#[allow(dead_code)]
pub(crate) enum RequestPurpose {
    Serve,
    CompactSummary,
    RefreshNative,
    EvaluationOnly,
    Prewarm,
    Memory,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RequestLinkage {
    pub(crate) decision_epoch: Option<u64>,
    pub(crate) decision_action: Option<&'static str>,
    pub(crate) compaction_id: Option<String>,
    pub(crate) reset_id: Option<String>,
    pub(crate) compaction_kind: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub(crate) struct RequestDescriptor<'a> {
    pub(crate) purpose: RequestPurpose,
    pub(crate) thread_id: &'a str,
    pub(crate) session_id: Option<&'a str>,
    pub(crate) turn_id: Option<&'a str>,
    pub(crate) provider_id: &'a str,
    pub(crate) model_id: &'a str,
    pub(crate) linkage: RequestLinkage,
}

#[derive(Clone, Debug)]
pub(crate) struct RawRequestHandle {
    pub(crate) request_index: u64,
    pub(crate) logical_request_id: String,
    descriptor: OwnedRequestDescriptor,
    started_at: Instant,
}

#[derive(Clone, Debug)]
struct OwnedRequestDescriptor {
    purpose: RequestPurpose,
    thread_id: String,
    session_id: Option<String>,
    turn_id: Option<String>,
    provider_id: String,
    model_id: String,
    linkage: RequestLinkage,
}

#[derive(Debug, Default)]
struct Counters {
    next_event_index: u64,
    next_request_index: u64,
}

#[derive(Clone, Debug)]
struct Identity {
    run_id: String,
    task_id: String,
    replicate_id: u64,
    arm: &'static str,
    pricing_schedule_id: Option<String>,
}

#[derive(Debug)]
pub(crate) struct RawRequestLedger {
    path: PathBuf,
    identity: Identity,
    counters: Mutex<Counters>,
    write_order: Semaphore,
}

impl RawRequestLedger {
    pub(crate) fn run_id(&self) -> &str {
        &self.identity.run_id
    }

    pub(crate) fn from_config(config: &ContextPolicyConfig) -> io::Result<Option<Arc<Self>>> {
        let Some(path) = config.request_raw_log_path.as_ref() else {
            return Ok(None);
        };
        let run_id = config.run_id.clone().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "request ledger requires run_id",
            )
        })?;
        let task_id = config.task_id.clone().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "request ledger requires task_id",
            )
        })?;
        let replicate_id = config.replicate_id.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "request ledger requires replicate_id",
            )
        })?;
        let counters = scan_existing(path.as_path(), &run_id)?;
        Ok(Some(Arc::new(Self {
            path: path.as_path().to_path_buf(),
            identity: Identity {
                run_id,
                task_id,
                replicate_id,
                arm: arm_name(config.mode),
                pricing_schedule_id: config.pricing_schedule_id.clone(),
            },
            counters: Mutex::new(counters),
            write_order: Semaphore::new(1),
        })))
    }

    pub(crate) async fn begin_request(
        &self,
        descriptor: RequestDescriptor<'_>,
    ) -> io::Result<RawRequestHandle> {
        let permit = self
            .write_order
            .acquire()
            .await
            .map_err(|_| io::Error::other("request ledger writer closed"))?;
        let request_index = {
            let mut counters = self
                .counters
                .lock()
                .map_err(|_| io::Error::other("request ledger counters poisoned"))?;
            let request_index = counters.next_request_index;
            counters.next_request_index += 1;
            request_index
        };
        let logical_request_id = format!("{}:request:{request_index}", self.identity.run_id);
        let owned = OwnedRequestDescriptor {
            purpose: descriptor.purpose,
            thread_id: descriptor.thread_id.to_string(),
            session_id: descriptor.session_id.map(str::to_string),
            turn_id: descriptor.turn_id.map(str::to_string),
            provider_id: descriptor.provider_id.to_string(),
            model_id: descriptor.model_id.to_string(),
            linkage: descriptor.linkage,
        };
        let handle = RawRequestHandle {
            request_index,
            logical_request_id,
            descriptor: owned,
            started_at: Instant::now(),
        };
        self.append_with_permit(&handle, "request_started", "start", Value::Null)
            .await?;
        drop(permit);
        Ok(handle)
    }

    pub(crate) async fn attempt_started(
        &self,
        handle: &RawRequestHandle,
        attempt_index: u64,
    ) -> io::Result<()> {
        self.append(
            handle,
            "attempt_started",
            &format!("attempt:{attempt_index}:start"),
            json!({
                "attempt_index": attempt_index,
            }),
        )
        .await
    }

    pub(crate) async fn attempt_failed(
        &self,
        handle: &RawRequestHandle,
        attempt_index: u64,
        failure_kind: &str,
        failure_reason: &str,
    ) -> io::Result<()> {
        self.append(
            handle,
            "attempt_failed",
            &format!("attempt:{attempt_index}:failed"),
            json!({
                "attempt_index": attempt_index,
                "failure_kind": failure_kind,
                "failure_reason": failure_reason,
                "billable": null,
            }),
        )
        .await
    }

    pub(crate) async fn retry_scheduled(
        &self,
        handle: &RawRequestHandle,
        attempt_index: u64,
        wait_seconds: f64,
    ) -> io::Result<()> {
        self.append(
            handle,
            "retry_scheduled",
            &format!("attempt:{attempt_index}:retry"),
            json!({
                "attempt_index": attempt_index,
                "wait_seconds": wait_seconds,
            }),
        )
        .await
    }

    pub(crate) async fn completed(
        &self,
        handle: &RawRequestHandle,
        response_id: Option<&str>,
        usage: Option<&TokenUsage>,
        visible_output_tokens: Option<i64>,
        stop_reason: &str,
    ) -> io::Result<()> {
        self.append(
            handle,
            "request_completed",
            "terminal",
            json!({
                "response_id": response_id,
                "provider_usage": usage.map(provider_usage),
                "usage_provenance": usage.map(|_| json!({
                    "input_tokens_total": "provider_reported_via_codex_token_usage",
                    "cache_read_tokens": "codex_normalized_provider_detail_or_default_zero",
                    "cache_write_tokens": "codex_normalized_provider_detail_or_default_zero",
                    "uncached_input_tokens": "host_derived_from_codex_normalized_buckets",
                    "billed_output_tokens": "provider_reported_via_codex_token_usage",
                    "reasoning_tokens": "codex_normalized_provider_detail_or_default_zero",
                    "visible_output_tokens": "host_estimated_model_visible_items_v1",
                })),
                "input_token_semantics": usage.map(|_| "total_includes_cache"),
                "visible_output_tokens": visible_output_tokens,
                "provider_reported_cost": null,
                "zero_cost_fixture": false,
                "latency_seconds": handle.started_at.elapsed().as_secs_f64(),
                "stop_reason": stop_reason,
            }),
        )
        .await
    }

    pub(crate) async fn failed(
        &self,
        handle: &RawRequestHandle,
        failure_kind: &str,
        failure_reason: &str,
    ) -> io::Result<()> {
        self.append(
            handle,
            "request_failed",
            "terminal",
            json!({
                "provider_usage": null,
                "input_token_semantics": null,
                "visible_output_tokens": null,
                "provider_reported_cost": null,
                "zero_cost_fixture": false,
                "latency_seconds": handle.started_at.elapsed().as_secs_f64(),
                "failure_kind": failure_kind,
                "failure_reason": failure_reason,
            }),
        )
        .await
    }

    async fn append(
        &self,
        handle: &RawRequestHandle,
        event: &str,
        event_suffix: &str,
        fields: Value,
    ) -> io::Result<()> {
        let permit = self
            .write_order
            .acquire()
            .await
            .map_err(|_| io::Error::other("request ledger writer closed"))?;
        let result = self
            .append_with_permit(handle, event, event_suffix, fields)
            .await;
        drop(permit);
        result
    }

    async fn append_with_permit(
        &self,
        handle: &RawRequestHandle,
        event: &str,
        event_suffix: &str,
        fields: Value,
    ) -> io::Result<()> {
        let event_index = {
            let mut counters = self
                .counters
                .lock()
                .map_err(|_| io::Error::other("request ledger counters poisoned"))?;
            let event_index = counters.next_event_index;
            counters.next_event_index += 1;
            event_index
        };
        let descriptor = &handle.descriptor;
        let mut record = json!({
            "schema_version": RAW_REQUEST_EVENT_SCHEMA_VERSION,
            "event_id": format!("{}:{event_suffix}", handle.logical_request_id),
            "event_index": event_index,
            "event": event,
            "run_id": self.identity.run_id,
            "task_id": self.identity.task_id,
            "replicate_id": self.identity.replicate_id,
            "arm": self.identity.arm,
            "thread_id": descriptor.thread_id,
            "request_index": handle.request_index,
            "logical_request_id": handle.logical_request_id,
            "request_purpose": descriptor.purpose,
            "decision_epoch": descriptor.linkage.decision_epoch,
            "decision_action": descriptor.linkage.decision_action,
            "compaction_id": descriptor.linkage.compaction_id,
            "reset_id": descriptor.linkage.reset_id,
            "compaction_kind": descriptor.linkage.compaction_kind,
            "session_id": descriptor.session_id,
            "turn_id": descriptor.turn_id,
            "provider_id": descriptor.provider_id,
            "model_id": descriptor.model_id,
            "pricing_schedule_id": self.identity.pricing_schedule_id,
        });
        if let (Some(target), Some(source)) = (record.as_object_mut(), fields.as_object()) {
            target.extend(source.clone());
        }
        let mut line = serde_json::to_vec(&record).map_err(io::Error::other)?;
        line.push(b'\n');
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await?;
        file.write_all(&line).await?;
        file.flush().await
    }
}

fn provider_usage(usage: &TokenUsage) -> Value {
    json!({
        "input_tokens_total": usage.input_tokens,
        "cache_read_tokens": usage.cached_input_tokens,
        "cache_write_tokens": usage.cache_write_input_tokens,
        "uncached_input_tokens": usage.input_tokens
            .saturating_sub(usage.cached_input_tokens)
            .saturating_sub(usage.cache_write_input_tokens),
        // Provider output already includes the separately reported reasoning bucket.
        "billed_output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_output_tokens,
    })
}

fn arm_name(mode: ContextPolicyMode) -> &'static str {
    match mode {
        ContextPolicyMode::NativeFixed => "native_fixed",
        ContextPolicyMode::ControlledFixed => "controlled_fixed",
        ContextPolicyMode::ExternalStub => "external_stub",
        ContextPolicyMode::MpcH1 => "mpc_h1",
        ContextPolicyMode::Mpc => "mpc",
    }
}

fn scan_existing(path: &std::path::Path, run_id: &str) -> io::Result<Counters> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Counters::default()),
        Err(error) => return Err(error),
    };
    let mut event_ids: HashMap<String, String> = HashMap::new();
    let mut counters = Counters::default();
    for (line_index, line) in content.lines().enumerate() {
        let value: Value = serde_json::from_str(line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid request ledger line {}: {error}", line_index + 1),
            )
        })?;
        if value.get("schema_version").and_then(Value::as_str)
            != Some(RAW_REQUEST_EVENT_SCHEMA_VERSION)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request ledger contains an unsupported schema",
            ));
        }
        if value.get("run_id").and_then(Value::as_str) != Some(run_id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request ledger path contains a different run_id",
            ));
        }
        let event_id = value
            .get("event_id")
            .and_then(Value::as_str)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "event_id is missing"))?;
        let signature = serde_json::to_string(&value).map_err(io::Error::other)?;
        if let Some(prior) = event_ids.insert(event_id.to_string(), signature.clone())
            && prior != signature
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request ledger contains a conflicting duplicate event_id",
            ));
        }
        let event_index = value
            .get("event_index")
            .and_then(Value::as_u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "event_index is missing"))?;
        let request_index = value
            .get("request_index")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "request_index is missing")
            })?;
        counters.next_event_index = counters.next_event_index.max(event_index + 1);
        counters.next_request_index = counters.next_request_index.max(request_index + 1);
    }
    Ok(counters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_utils_absolute_path::AbsolutePathBuf;
    use tempfile::TempDir;

    fn config(path: &std::path::Path) -> ContextPolicyConfig {
        ContextPolicyConfig {
            mode: ContextPolicyMode::Mpc,
            run_id: Some("run".into()),
            task_id: Some("task".into()),
            replicate_id: Some(4),
            request_raw_log_path: Some(
                AbsolutePathBuf::from_absolute_path(path).expect("absolute ledger path"),
            ),
            pricing_schedule_id: Some("fixture-price-v1".into()),
            ..Default::default()
        }
    }

    fn descriptor<'a>() -> RequestDescriptor<'a> {
        RequestDescriptor {
            purpose: RequestPurpose::Serve,
            thread_id: "thread",
            session_id: Some("session"),
            turn_id: Some("turn"),
            provider_id: "fixture-provider",
            model_id: "fixture-model",
            linkage: RequestLinkage {
                decision_epoch: Some(2),
                decision_action: Some("KEEP"),
                ..Default::default()
            },
        }
    }

    #[tokio::test]
    async fn emits_one_logical_lifecycle_with_raw_usage() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("requests.raw.jsonl");
        let ledger = RawRequestLedger::from_config(&config(&path))
            .expect("ledger")
            .expect("enabled");
        let request = ledger.begin_request(descriptor()).await.expect("start");
        ledger.attempt_started(&request, 0).await.expect("attempt");
        ledger
            .completed(
                &request,
                Some("response"),
                Some(&TokenUsage {
                    input_tokens: 10,
                    cached_input_tokens: 3,
                    cache_write_input_tokens: 1,
                    output_tokens: 6,
                    reasoning_output_tokens: 2,
                    total_tokens: 16,
                    codex_rollout_budget_units: None,
                }),
                Some(4),
                "completed",
            )
            .await
            .expect("complete");
        let rows = std::fs::read_to_string(&path)
            .expect("read")
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("json"))
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 3);
        assert!(
            rows.iter()
                .all(|row| row["logical_request_id"] == "run:request:0")
        );
        assert_eq!(rows[2]["provider_usage"]["uncached_input_tokens"], 6);
        assert_eq!(rows[2]["provider_usage"]["billed_output_tokens"], 6);
        assert_eq!(rows[2]["provider_usage"]["reasoning_tokens"], 2);
        assert_eq!(rows[2]["visible_output_tokens"], 4);
    }

    #[tokio::test]
    async fn retry_stays_one_request_and_resume_continues_indices() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("requests.raw.jsonl");
        let cfg = config(&path);
        let ledger = RawRequestLedger::from_config(&cfg)
            .expect("ledger")
            .expect("enabled");
        let request = ledger.begin_request(descriptor()).await.expect("start");
        ledger.attempt_started(&request, 0).await.expect("attempt");
        ledger
            .attempt_failed(&request, 0, "rate_limit", "429")
            .await
            .expect("failure");
        ledger
            .retry_scheduled(&request, 0, 1.0)
            .await
            .expect("retry");
        ledger.attempt_started(&request, 1).await.expect("attempt");
        ledger
            .failed(&request, "provider", "unavailable")
            .await
            .expect("terminal");
        drop(ledger);

        let resumed = RawRequestLedger::from_config(&cfg)
            .expect("resume")
            .expect("enabled");
        let next = resumed
            .begin_request(descriptor())
            .await
            .expect("next start");
        assert_eq!(next.request_index, 1);
        assert_eq!(next.logical_request_id, "run:request:1");
        let rows = std::fs::read_to_string(path).expect("read");
        let indices = rows
            .lines()
            .map(|line| {
                serde_json::from_str::<Value>(line).expect("json")["event_index"]
                    .as_u64()
                    .expect("index")
            })
            .collect::<Vec<_>>();
        assert_eq!(indices, (0..indices.len() as u64).collect::<Vec<_>>());
    }

    #[test]
    fn schema_supports_all_current_and_future_purpose_classes() {
        let purposes = [
            RequestPurpose::Serve,
            RequestPurpose::CompactSummary,
            RequestPurpose::RefreshNative,
            RequestPurpose::EvaluationOnly,
            RequestPurpose::Prewarm,
            RequestPurpose::Memory,
        ];
        assert_eq!(
            purposes
                .iter()
                .map(|purpose| serde_json::to_value(purpose).expect("serialize"))
                .collect::<Vec<_>>(),
            vec![
                json!("serve"),
                json!("compact_summary"),
                json!("refresh_native"),
                json!("evaluation_only"),
                json!("prewarm"),
                json!("memory")
            ]
        );
    }
}
