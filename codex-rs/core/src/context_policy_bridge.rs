//! Persistent fail-closed JSONL client for the local Phase 8B policy process.

use std::io;
use std::process::Stdio;
use std::time::Duration;

use codex_config::types::ContextPolicyConfig;
use serde_json::Value;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::io::Lines;
use tokio::process::Child;
use tokio::process::ChildStdin;
use tokio::process::ChildStdout;
use tokio::process::Command;

pub(crate) const PROTOCOL_VERSION: &str = "phase8b-v1";

pub(crate) struct ContextPolicyBridge {
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    timeout: Duration,
}

impl ContextPolicyBridge {
    pub(crate) fn spawn(config: &ContextPolicyConfig) -> io::Result<Self> {
        let executable = config
            .bridge_command
            .as_ref()
            .ok_or_else(|| invalid("MPC bridge_command is missing"))?;
        let mut command = Command::new(executable.as_path());
        command
            .args(&config.bridge_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        if let Some(directory) = &config.bridge_working_directory {
            command.current_dir(directory.as_path());
        }
        let mut child = command.spawn().map_err(|error| {
            io::Error::other(format!(
                "context policy infrastructure failure: spawn: {error}"
            ))
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| infrastructure("child stdin unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| infrastructure("child stdout unavailable"))?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout).lines(),
            timeout: Duration::from_millis(config.bridge_timeout_ms.unwrap_or(5_000)),
        })
    }

    pub(crate) async fn exchange(&mut self, request: &Value) -> io::Result<Value> {
        if let Some(status) = self.child.try_wait()? {
            return Err(infrastructure(format!("child exited with {status}")));
        }
        let mut bytes = serde_json::to_vec(request).map_err(io::Error::other)?;
        bytes.push(b'\n');
        tokio::time::timeout(self.timeout, async {
            self.stdin.write_all(&bytes).await?;
            self.stdin.flush().await?;
            Ok::<_, io::Error>(())
        })
        .await
        .map_err(|_| infrastructure("write timeout"))??;

        let line = tokio::time::timeout(self.timeout, self.stdout.next_line())
            .await
            .map_err(|_| infrastructure("response timeout"))??
            .ok_or_else(|| infrastructure("child stdout closed"))?;
        let response: Value = serde_json::from_str(&line)
            .map_err(|error| infrastructure(format!("malformed response JSON: {error}")))?;
        validate_finite(&response)?;
        if response.get("type").and_then(Value::as_str) == Some("infrastructure_error") {
            let code = response
                .get("error_code")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let message = response
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("bridge rejected request");
            return Err(infrastructure(format!("{code}: {message}")));
        }
        for field in [
            "protocol_version",
            "request_id",
            "run_id",
            "task_id",
            "replicate_id",
            "thread_id",
            "epoch",
        ] {
            if response.get(field) != request.get(field) {
                return Err(infrastructure(format!(
                    "response identity mismatch: {field}"
                )));
            }
        }
        Ok(response)
    }
}

fn validate_finite(value: &Value) -> io::Result<()> {
    match value {
        Value::Number(number) if number.as_f64().is_some_and(|number| !number.is_finite()) => {
            Err(infrastructure("non-finite numeric diagnostic"))
        }
        Value::Array(items) => items.iter().try_for_each(validate_finite),
        Value::Object(items) => items.values().try_for_each(validate_finite),
        _ => Ok(()),
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn infrastructure(message: impl Into<String>) -> io::Error {
    io::Error::other(format!(
        "context policy infrastructure failure: {}",
        message.into()
    ))
}

#[cfg(test)]
#[path = "context_policy_bridge_tests.rs"]
mod tests;
