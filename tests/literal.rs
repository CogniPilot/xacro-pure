//! `_eval_literal` coercion fidelity table.
//!
//! Each expected value was cross-checked against canonical xacro's
//! `Table._eval_literal` semantics via `python3` (see the port report). The
//! load-bearing subtleties: single-quote strip happens FIRST (so `'42'` is the
//! *string* `42`, not the int), the PEP515 underscore guard keeps any value with
//! `_` as a string, and numeric coercion is int -> float -> bool in that order.

use xacro_pure::{eval_literal, XacroValue};

#[test]
fn ints() {
    // python3 _eval_literal: "42"->42, "-7"->-7, "0"->0, "1"->1 (all int)
    assert_eq!(eval_literal("42"), XacroValue::int(42));
    assert_eq!(eval_literal("-7"), XacroValue::int(-7));
    assert_eq!(eval_literal("0"), XacroValue::int(0));
    assert_eq!(eval_literal("1"), XacroValue::int(1));
}

#[test]
fn floats() {
    // python3: "2.71"->2.71, "1e-3"->0.001, "-2.5"->-2.5, "1.0"->1.0 (all float)
    assert_eq!(eval_literal("2.71"), XacroValue::Float(2.71));
    assert_eq!(eval_literal("1e-3"), XacroValue::Float(0.001));
    assert_eq!(eval_literal("-2.5"), XacroValue::Float(-2.5));
    assert_eq!(eval_literal("1.0"), XacroValue::Float(1.0));
}

#[test]
fn bools() {
    // python3: "true"/"True"->True, "false"/"False"->False (the boolean coercion
    // is reached only after int/float fail, which they do for these tokens).
    assert_eq!(eval_literal("true"), XacroValue::Bool(true));
    assert_eq!(eval_literal("True"), XacroValue::Bool(true));
    assert_eq!(eval_literal("false"), XacroValue::Bool(false));
    assert_eq!(eval_literal("False"), XacroValue::Bool(false));
}

#[test]
fn single_quoted_strings_strip_quotes() {
    // python3: "'hello'" -> 'hello' (the inner string), quotes stripped FIRST.
    assert_eq!(eval_literal("'hello'"), XacroValue::Str("hello".to_owned()));
    // And critically: "'42'" -> the STRING '42', NOT the int 42, because the
    // quote-strip returns immediately before any numeric coercion.
    assert_eq!(eval_literal("'42'"), XacroValue::Str("42".to_owned()));
    // A quoted value containing '_' also strips to the inner string (the strip
    // happens before the underscore guard).
    assert_eq!(
        eval_literal("'with_underscore'"),
        XacroValue::Str("with_underscore".to_owned())
    );
    // An empty single-quoted pair "''" strips to the empty string.
    assert_eq!(eval_literal("''"), XacroValue::Str(String::new()));
}

#[test]
fn underscore_values_stay_string_pep515() {
    // python3: "1_000" -> '1_000' (a STRING, PEP515 guard, NOT the int 1000).
    assert_eq!(eval_literal("1_000"), XacroValue::Str("1_000".to_owned()));
    // python3: "wheel_1" -> 'wheel_1' (string; it would not parse numeric anyway,
    // but the underscore guard short-circuits it explicitly).
    assert_eq!(
        eval_literal("wheel_1"),
        XacroValue::Str("wheel_1".to_owned())
    );
    // A float-looking value with an underscore also stays a string.
    assert_eq!(eval_literal("1_0.5"), XacroValue::Str("1_0.5".to_owned()));
}

#[test]
fn plain_strings_stay_string() {
    // python3: "plain" -> 'plain', "robot_base" handled by underscore guard but
    // "base" reaches the numeric coercions, all fail -> string.
    assert_eq!(eval_literal("plain"), XacroValue::Str("plain".to_owned()));
    assert_eq!(eval_literal("base"), XacroValue::Str("base".to_owned()));
    // Mixed alphanumerics that are not valid numbers stay strings.
    assert_eq!(eval_literal("3.4.5"), XacroValue::Str("3.4.5".to_owned()));
    assert_eq!(eval_literal("12px"), XacroValue::Str("12px".to_owned()));
}

#[test]
fn coercion_order_int_before_float() {
    // "42" must be an Int, not a Float; int is tried first and wins.
    assert!(matches!(eval_literal("42"), XacroValue::Int(_)));
    // "42.0" cannot parse as int, falls through to float.
    assert!(matches!(eval_literal("42.0"), XacroValue::Float(_)));
}

#[test]
fn whitespace_only_and_empty() {
    // Empty string: not quoted, no underscore, parses as no number -> stays "".
    assert_eq!(eval_literal(""), XacroValue::Str(String::new()));
    // A lone single quote (len 1) is not a quoted pair -> string.
    assert_eq!(eval_literal("'"), XacroValue::Str("'".to_owned()));
}
