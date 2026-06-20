//! `_eval_literal`: the property-definition-time type coercion.
//!
//! Faithful port of canonical xacro's `Table._eval_literal`
//! (`xacro/__init__.py:327`):
//! ```python
//! @staticmethod
//! def _eval_literal(value):
//!     if isinstance(value, str):
//!         # remove single quotes from escaped string
//!         if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
//!             return value[1:-1]
//!         # PEP515: python drops underscores in number literals; keep such
//!         # literals as plain strings instead.
//!         if '_' in value:
//!             return value
//!         for f in [int, float, lambda x: get_boolean_value(x, None)]:  # ORDER MATTERS
//!             try:
//!                 return f(value)
//!             except Exception:
//!                 pass
//!     return value
//! ```
//!
//! The exact ordering and guards are load-bearing:
//!   1. **single-quote strip happens FIRST**: so `'42'` becomes the *string*
//!      `42`, never the int 42 (the quote-strip returns immediately).
//!   2. **underscore guard (PEP515)**: any value containing `_` stays a string
//!      (so `1_000` and `wheel_1` are both strings), guarding against Python's
//!      silent underscore-dropping in numeric literals.
//!   3. **int -> float -> bool, in that order**: `int("42")` wins over
//!      `float`; only non-numeric `true/false/True/False/0/1`-style values reach
//!      the boolean coercion (which is `get_boolean_value(x, None)`).

use std::str::FromStr;

use num_bigint::BigInt;

use crate::value::XacroValue;

/// Coerce a literal `text` (a raw property RHS) to a typed [`XacroValue`],
/// reproducing canonical xacro's `_eval_literal` exactly.
pub fn eval_literal(text: &str) -> XacroValue {
    // (1) strip a surrounding single-quote pair (escaped string), and RETURN.
    // Mirrors `value[1:-1]`: just the inner bytes, no further coercion.
    let chars: Vec<char> = text.chars().collect();
    if chars.len() >= 2 && chars[0] == '\'' && chars[chars.len() - 1] == '\'' {
        let inner: String = chars[1..chars.len() - 1].iter().collect();
        return XacroValue::Str(inner);
    }

    // (2) PEP515 underscore guard: keep as string.
    if text.contains('_') {
        return XacroValue::Str(text.to_owned());
    }

    // (3) int -> float -> bool, in that order. The FIRST that parses wins.
    if let Some(i) = parse_py_int(text) {
        return XacroValue::Int(i);
    }
    if let Some(f) = parse_py_float(text) {
        return XacroValue::Float(f);
    }
    if let Some(b) = parse_py_bool(text) {
        return XacroValue::Bool(b);
    }

    // Nothing matched: a plain string.
    XacroValue::Str(text.to_owned())
}

/// Parse like Python's `int(str)`: an optional sign then ASCII digits, with
/// surrounding whitespace tolerated (Python's `int()` strips it). Rejects
/// underscores (already filtered above) and any non-integer form. Returns an
/// arbitrary-precision [`BigInt`] because Python's `int()` is unbounded; a
/// literal wider than `i64` (e.g. `99999999999999999999999`) must coerce to an
/// int, not fall through to `float`.
fn parse_py_int(text: &str) -> Option<BigInt> {
    let t = text.trim();
    if t.is_empty() {
        return None;
    }
    // `num_bigint::BigInt::from_str` accepts an optional leading +/- then ASCII
    // digits, rejecting a bare sign and any embedded whitespace/dot/exponent,
    // matching Python `int()` over the literal forms xacro sees.
    BigInt::from_str(t).ok()
}

/// Parse like Python's `float(str)`: int/float/exponent forms, whitespace
/// tolerated. Python also accepts `inf`/`nan`, which Rust's parser handles too.
fn parse_py_float(text: &str) -> Option<f64> {
    let t = text.trim();
    if t.is_empty() {
        return None;
    }
    // Guard against Rust accepting forms Python's float() also accepts but which
    // we want to keep numeric (e.g. "1e-3", "3.14"). Rust's f64 parser is a
    // close superset match for the literals xacro sees. Underscores are already
    // excluded by the PEP515 guard upstream.
    t.parse::<f64>().ok()
}

/// Reproduce `get_boolean_value(x, None)` for a string input: only the literal
/// tokens canonical xacro accepts. `"true"`/`"True"` -> true, `"false"`/
/// `"False"` -> false, else `bool(int(value))` (so `"0"`->false, `"1"`->true).
/// In `_eval_literal`'s pipeline the int/float branches have already consumed
/// `"0"`/`"1"`/etc., so in practice only `true`/`false`/`True`/`False` reach
/// here, but we reproduce the full rule for fidelity.
fn parse_py_bool(text: &str) -> Option<bool> {
    match text {
        "true" | "True" => Some(true),
        "false" | "False" => Some(false),
        _ => {
            // bool(int(value)): parse as int, then non-zero -> true.
            text.trim().parse::<i64>().ok().map(|i| i != 0)
        }
    }
}
