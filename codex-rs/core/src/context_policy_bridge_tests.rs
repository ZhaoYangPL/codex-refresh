use serde_json::json;

use super::PROTOCOL_VERSION;
use super::validate_finite;

#[test]
fn protocol_version_is_phase8b_v1() {
    assert_eq!(PROTOCOL_VERSION, "phase8b-v1");
}

#[test]
fn nonfinite_diagnostics_fail_closed() {
    assert!(validate_finite(&json!({"q_keep": f64::INFINITY})).is_err());
    assert!(validate_finite(&json!({"q_keep": 1.0})).is_ok());
}
