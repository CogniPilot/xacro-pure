//! Gate: the 9 corpus expression shapes each evaluate to the
//! CPython-correct value, plus the security posture (dunder rejection,
//! `__builtins__` unreachable) and determinism.
//!
//! Every expected value below was cross-checked against `python3 -c` (see the
//! port report) so these are CPython-correct, not assumed.

use std::str::FromStr;

use indexmap::IndexMap;
use num_bigint::BigInt;
use xacro_pure::{safe_eval, EvalError, Namespace, XacroValue};

fn eval(expr: &str) -> XacroValue {
    safe_eval(expr, &Namespace::new()).unwrap_or_else(|e| panic!("eval `{expr}` failed: {e}"))
}

/// An `XacroValue::Int` from a decimal string (for values too big for an int
/// literal). `decimal` is assumed valid.
fn big(decimal: &str) -> XacroValue {
    XacroValue::Int(BigInt::from_str(decimal).expect("valid decimal"))
}

fn eval_ns(expr: &str, ns: &Namespace) -> XacroValue {
    safe_eval(expr, ns).unwrap_or_else(|e| panic!("eval `{expr}` failed: {e}"))
}

// ---- The 9 corpus expression shapes (the gate) ----

#[test]
fn case1_dict_index() {
    // python3: dict(x=1.0)['x'] == 1.0  (a float)
    assert_eq!(eval("dict(x=1.0)['x']"), XacroValue::Float(1.0));
}

#[test]
fn case2_string_concat() {
    // python3: 'a' + 'b' == 'ab'
    assert_eq!(eval("'a' + 'b'"), XacroValue::Str("ab".to_owned()));
}

#[test]
fn case3_ternary() {
    // python3: (10 if True else 20) == 10
    let mut ns = Namespace::new();
    ns.set("A", XacroValue::int(10));
    ns.set("B", XacroValue::int(20));
    ns.set("C", XacroValue::Bool(true));
    assert_eq!(eval_ns("A if C else B", &ns), XacroValue::int(10));
}

#[test]
fn case4_membership() {
    // python3: 'config' in dict(config=1) == True
    assert_eq!(eval("'config' in dict(config=1)"), XacroValue::Bool(true));
    // and a namespace-driven membership: k in d
    let mut ns = Namespace::new();
    let mut d = IndexMap::new();
    d.insert("wheel".to_owned(), XacroValue::int(1));
    ns.set("d", XacroValue::Dict(d));
    ns.set("k", XacroValue::Str("wheel".to_owned()));
    assert_eq!(eval_ns("k in d", &ns), XacroValue::Bool(true));
}

#[test]
fn case5_percent_format() {
    // python3: '%.3f' % 1.23456 == '1.235'
    let mut ns = Namespace::new();
    ns.set("v", XacroValue::Float(1.23456));
    assert_eq!(
        eval_ns("'%.3f' % v", &ns),
        XacroValue::Str("1.235".to_owned())
    );
}

#[test]
fn case6_list_comprehension() {
    // python3: [n for n in [1,2,3] if n > 1] == [2, 3]
    assert_eq!(
        eval("[n for n in [1,2,3] if n > 1]"),
        XacroValue::List(vec![XacroValue::int(2), XacroValue::int(3)])
    );
}

#[test]
fn case7_negative_index() {
    // python3: [10,20,30][-1] == 30
    let mut ns = Namespace::new();
    ns.set(
        "xs",
        XacroValue::List(vec![
            XacroValue::int(10),
            XacroValue::int(20),
            XacroValue::int(30),
        ]),
    );
    assert_eq!(eval_ns("xs[-1]", &ns), XacroValue::int(30));
}

#[test]
fn case8_slice() {
    // python3: [10,20,30][1:3] == [20, 30]
    let mut ns = Namespace::new();
    ns.set(
        "xs",
        XacroValue::List(vec![
            XacroValue::int(10),
            XacroValue::int(20),
            XacroValue::int(30),
        ]),
    );
    assert_eq!(
        eval_ns("xs[1:3]", &ns),
        XacroValue::List(vec![XacroValue::int(20), XacroValue::int(30)])
    );
}

#[test]
fn case9_injected_rust_function() {
    // The seam for xacro.load_yaml etc.: a Rust fn callable from the Python
    // expression. Stub: myfunc(n) = n * 10. python3 (myfunc(7)+1) == 71.
    let mut ns = Namespace::new();
    ns.register_fn_i64("myfunc", |a| a * 10);
    assert_eq!(eval_ns("myfunc(7) + 1", &ns), XacroValue::int(71));
}

// ---- Security posture (matches canonical safe_eval) ----

#[test]
fn rejects_dunder_name_access() {
    // A dunder-name reference must be rejected pre-eval, like canonical xacro's
    // `code.co_names` filter. ().__class__ references the dunder name __class__.
    let err = safe_eval("().__class__", &Namespace::new()).unwrap_err();
    match err {
        EvalError::InvalidName(names) => assert!(
            names.contains("__class__"),
            "expected __class__ in invalid names, got: {names}"
        ),
        other => panic!("expected InvalidName, got {other:?}"),
    }
}

#[test]
fn rejects_dunder_builtins_name() {
    // Referencing __builtins__ by name is a dunder access -> rejected.
    let err = safe_eval("__builtins__", &Namespace::new()).unwrap_err();
    assert!(matches!(err, EvalError::InvalidName(_)));
}

#[test]
fn default_builtins_are_unreachable() {
    // __builtins__ is blanked, so a builtin NOT in our injected set (e.g. `open`,
    // `eval`, `__import__`) is undefined -> a NameError at runtime, not a value.
    // (`open` is not a dunder, so it passes the name filter and fails at eval.)
    let err = safe_eval("open('x')", &Namespace::new()).unwrap_err();
    assert!(
        matches!(err, EvalError::Runtime(ref m) if m.contains("NameError") || m.contains("not defined")),
        "expected a NameError for `open`, got: {err:?}"
    );
}

#[test]
fn dangerous_builtins_are_neutralized() {
    // The classic escape vectors must all fail. `eval`/`getattr`/`compile` are
    // shadowed (calling them raises a NameError-equivalent); `__import__` and
    // `__subclasses__` are rejected by the dunder-name filter pre-eval. None of
    // these may return a value.
    for expr in ["eval('1')", "getattr(1, 'real')", "compile('1','x','eval')"] {
        let err = safe_eval(expr, &Namespace::new()).unwrap_err();
        assert!(
            matches!(err, EvalError::Runtime(ref m) if m.contains("not defined")),
            "expr `{expr}` should be neutralized, got: {err:?}"
        );
    }
    for expr in ["__import__('os')", "(1).__class__"] {
        let err = safe_eval(expr, &Namespace::new()).unwrap_err();
        assert!(
            matches!(err, EvalError::InvalidName(_)),
            "expr `{expr}` should be rejected as invalid name, got: {err:?}"
        );
    }
}

#[test]
fn injected_safe_builtins_still_work() {
    // The builtins canonical xacro DOES expose (dict/list/...) are reachable
    // because rustpython's scope provides them and they are not dunder names.
    assert_eq!(eval("len([1,2,3])"), XacroValue::int(3));
    assert_eq!(eval("str(42)"), XacroValue::Str("42".to_owned()));
}

// ---- Determinism ----

#[test]
fn deterministic_same_expr_same_namespace() {
    let mut ns = Namespace::new();
    ns.set("a", XacroValue::Float(2.5));
    ns.set("b", XacroValue::int(4));
    let expr = "a * b + 1";
    let first = eval_ns(expr, &ns);
    for _ in 0..10 {
        assert_eq!(eval_ns(expr, &ns), first);
    }
    // python3: 2.5*4 + 1 == 11.0
    assert_eq!(first, XacroValue::Float(11.0));
}

#[test]
fn dict_result_preserves_insertion_order() {
    // A ${dict(...)} survives as a real typed dict (the passthrough); key order
    // is preserved so the conversion is deterministic.
    let v = eval("dict(a=1, b=2, c=3)");
    match v {
        XacroValue::Dict(m) => {
            let keys: Vec<&String> = m.keys().collect();
            assert_eq!(keys, vec!["a", "b", "c"]);
        }
        other => panic!("expected dict, got {other:?}"),
    }
}

// ---- Error paths return EvalError, never panic ----

#[test]
fn syntax_error_is_compile_error() {
    let err = safe_eval("1 +", &Namespace::new()).unwrap_err();
    assert!(matches!(err, EvalError::Compile(_)), "got {err:?}");
}

#[test]
fn missing_name_is_runtime_error() {
    let err = safe_eval("undefined_property + 1", &Namespace::new()).unwrap_err();
    assert!(matches!(err, EvalError::Runtime(_)), "got {err:?}");
}

// ---- Arbitrary-precision int parity with CPython/canonical xacro ----
//
// CPython's `int` is unbounded; canonical `xacro.safe_eval` was re-checked live:
//   '2**63'   -> 9223372036854775808
//   '10**30'  -> 1000000000000000000000000000000
//   '2**100'  -> 1267650600228229401496703205376
//   '-2**63'  -> -9223372036854775808   (== i64::MIN, the old boundary)
// The port must produce these EXACT values, not overflow / Bridge-error.

#[test]
fn bigint_two_pow_63_is_exact() {
    // The old i64 boundary: 2**63 is i64::MAX + 1 and used to hard-error.
    assert_eq!(eval("2**63"), big("9223372036854775808"));
}

#[test]
fn bigint_ten_pow_30_is_exact() {
    assert_eq!(eval("10**30"), big("1000000000000000000000000000000"));
}

#[test]
fn bigint_two_pow_100_is_exact() {
    assert_eq!(eval("2**100"), big("1267650600228229401496703205376"));
}

#[test]
fn bigint_i64_min_boundary_still_exact() {
    // -2**63 is exactly i64::MIN; it worked before and must still be exact.
    assert_eq!(eval("-2**63"), XacroValue::int(i64::MIN));
}

#[test]
fn bigint_str_renders_full_magnitude() {
    // str(2**100) in Python is the full decimal; to_python_str must match (this
    // is how a typed big-int result is finally serialized into XML text).
    match eval("2**100") {
        XacroValue::Int(_) => {
            assert_eq!(
                eval("2**100").to_python_str(),
                "1267650600228229401496703205376"
            );
        }
        other => panic!("expected Int, got {other:?}"),
    }
}

#[test]
fn bigint_injected_property_round_trips() {
    // A big-int property injected into the scope must reach the expression with
    // full precision: (2**100) + 1 evaluated where the base is passed in.
    let mut ns = Namespace::new();
    ns.set("base", big("1267650600228229401496703205376"));
    assert_eq!(
        eval_ns("base + 1", &ns),
        big("1267650600228229401496703205377")
    );
}

#[test]
fn bigint_arithmetic_stays_exact() {
    // No silent float widening: a product that exceeds i64 stays an exact int.
    // 10**12 * 10**12 == 10**24 (python-verified).
    assert_eq!(
        eval("1000000000000 * 1000000000000"),
        big("1000000000000000000000000")
    );
}

#[test]
fn bigint_literal_coerces_in_eval_literal() {
    // _eval_literal: a decimal literal wider than i64 coerces to Int, not float.
    use xacro_pure::eval_literal;
    assert_eq!(
        eval_literal("9223372036854775808"),
        big("9223372036854775808")
    );
}
