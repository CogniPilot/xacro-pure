//! The typed value that crosses the Rust <-> Python boundary.
//!
//! Canonical xacro evaluates a `${...}` to a *Python value* whose Python type
//! is load-bearing downstream: an `int` vs a `float` vs a `str` changes how a
//! later `_eval_literal` / `eval_text` / `%`-format behaves, and a `dict`/`list`
//! can survive as a typed single-result passthrough (a `${dict(...)}` stays a
//! real dict). So this enum keeps the **int/float/bool/str/list/dict/null**
//! distinction rather than collapsing numbers to one `f64`; Python's `bool` is
//! a subclass of `int`, so we must surface it as its own variant to round-trip
//! type identity faithfully.

use indexmap::IndexMap;
use num_bigint::BigInt;
use num_traits::ToPrimitive;

/// A Python value as seen from Rust. The variants mirror exactly the Python
/// types that canonical xacro's `safe_eval` / `_eval_literal` can produce for a
/// `${...}` expression or a property literal.
#[derive(Debug, Clone, PartialEq)]
pub enum XacroValue {
    /// Python `int`. Held as an arbitrary-precision [`BigInt`] (not `i64`)
    /// because CPython's `int` is unbounded: canonical xacro evaluates
    /// `2**63`, `10**30`, `2**100`, etc. to exact integers, so the port must
    /// represent them losslessly rather than overflow an `i64`. Kept distinct
    /// from [`Float`](XacroValue::Float) for type fidelity.
    Int(BigInt),
    /// Python `float`.
    Float(f64),
    /// Python `bool` (distinct from `Int`: `bool` is an `int` subclass in
    /// Python, but its identity matters for round-tripping and for `str()`).
    Bool(bool),
    /// Python `str`.
    Str(String),
    /// Python `list`.
    List(Vec<XacroValue>),
    /// Python `dict`. Insertion order is preserved (Python dicts are ordered
    /// since 3.7) so conversion is deterministic.
    Dict(IndexMap<String, XacroValue>),
    /// Python `tuple`. Kept distinct from [`List`](XacroValue::List) because its
    /// `repr`/`str` differs: parenthesized, and a 1-tuple carries a trailing
    /// comma (`(1,)`). A `${(1, 2.0)}` result must round-trip as a tuple, not a
    /// list, to match canonical xacro's `str(tuple)`.
    Tuple(Vec<XacroValue>),
    /// A xacro `NameSpace` object (canonical `NameSpace`): a property bag exposing
    /// its entries via DOTTED ATTRIBUTE access (`ns.prop`), created by a
    /// `<xacro:include ns="..">`. Distinct from [`Dict`](XacroValue::Dict) because
    /// canonical's `NameSpace.__getattr__` forwards `ns.prop` to `__getitem__`,
    /// whereas a plain dict only supports `ns['prop']`. Bridged to a Python module-
    /// like object so `${ns.prop}` resolves in `safe_eval` exactly as canonical's
    /// attribute access does. Entries are the namespace's own + inherited keys
    /// (materialized at access time), preserving insertion order for determinism.
    Namespace(IndexMap<String, XacroValue>),
    /// Python `None`.
    Null,
}

impl XacroValue {
    /// Construct an [`Int`](XacroValue::Int) from anything that converts into a
    /// [`BigInt`] (`i32`/`i64`/`u64`/`BigInt`/...). Convenience so call sites can
    /// write `XacroValue::int(42)` instead of `XacroValue::Int(BigInt::from(42))`.
    pub fn int(value: impl Into<BigInt>) -> Self {
        XacroValue::Int(value.into())
    }

    /// Best-effort numeric view (an `Int` or `Bool` widened to `f64`). Returns
    /// `None` for non-numeric variants. Convenience for callers that just want a
    /// number regardless of int/float; the variant itself preserves fidelity.
    ///
    /// A `BigInt` too large to land in an `f64` saturates to `±inf` (matching
    /// CPython's `float(huge_int)` which raises `OverflowError`; here we keep the
    /// total `Option<f64>` contract and surface infinity rather than panic).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            XacroValue::Int(i) => Some(i.to_f64().unwrap_or_else(|| {
                if i.sign() == num_bigint::Sign::Minus {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }
            })),
            XacroValue::Float(f) => Some(*f),
            XacroValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// The `str()` rendering of this value, matching Python's `str()` for the
    /// shapes xacro emits into XML text (which is how a typed result is finally
    /// serialized). Notably: Python `True`/`False` (capitalized), `None` ->
    /// `"None"`, floats via Python's float repr-ish form.
    pub fn to_python_str(&self) -> String {
        match self {
            XacroValue::Int(i) => i.to_string(),
            XacroValue::Float(f) => format_py_float(*f),
            XacroValue::Bool(b) => {
                if *b {
                    "True".to_owned()
                } else {
                    "False".to_owned()
                }
            }
            XacroValue::Str(s) => s.clone(),
            XacroValue::Null => "None".to_owned(),
            XacroValue::List(items) => {
                let inner: Vec<String> = items.iter().map(XacroValue::to_python_repr).collect();
                format!("[{}]", inner.join(", "))
            }
            XacroValue::Dict(map) | XacroValue::Namespace(map) => {
                // `NameSpace` is a `dict` subclass in canonical xacro, so `str()`
                // renders it exactly like a dict.
                let inner: Vec<String> = map
                    .iter()
                    .map(|(k, v)| format!("{}: {}", py_str_repr(k), v.to_python_repr()))
                    .collect();
                format!("{{{}}}", inner.join(", "))
            }
            XacroValue::Tuple(items) => {
                let inner: Vec<String> = items.iter().map(XacroValue::to_python_repr).collect();
                // Python tuple repr: `()` empty, `(x,)` single (trailing comma),
                // `(a, b)` otherwise.
                match inner.len() {
                    0 => "()".to_owned(),
                    1 => format!("({},)", inner[0]),
                    _ => format!("({})", inner.join(", ")),
                }
            }
        }
    }

    /// The `repr()` rendering, used for elements *inside* a list/dict (where
    /// Python uses `repr`, not `str`, so strings get quotes).
    fn to_python_repr(&self) -> String {
        match self {
            XacroValue::Str(s) => py_str_repr(s),
            _ => self.to_python_str(),
        }
    }
}

/// Render an `f64` exactly the way CPython's `str(float)` / `repr(float)` does.
///
/// CPython uses `PyOS_double_to_string(x, 'r', 0, ...)` (repr mode): it computes
/// the SHORTEST decimal digit string that round-trips to `x` (David Gay / Ryū),
/// then chooses fixed vs scientific notation from the decimal point position
/// `decpt` (the count of significant digits BEFORE the decimal point):
///   * scientific iff `decpt <= -4` or `decpt > 16`;
///   * otherwise fixed, with a trailing `.0` for an integral value.
///
/// Scientific form is `<m>e[+-]NN`: a sign is ALWAYS present and the exponent is
/// zero-padded to at least two digits (`1e+16`, `1e-07`, `1.5e+300`).
///
/// We obtain the shortest digits + base-10 exponent from Rust's `{:e}` formatter
/// (which, like `{}`, emits the shortest round-trip mantissa (verified) but in
/// scientific layout so the exponent is explicit). From `<digits>e<exp>` we have
/// `decpt = exp + 1`, then reconstruct the chosen notation byte-for-byte.
///
/// This matters because Rust's bare `{}` NEVER switches to scientific notation:
/// `${1e16}` would otherwise stringify to `10000000000000000`, `${1e-7}` to
/// `0.0000001`, and `${1.5e300}` to a 301-digit number, all numerically equal
/// but textually divergent from canonical xacro (and mis-parsed by some
/// downstream consumers). Joint inertias/limits/masses routinely land in the
/// scientific band, so the fix is load-bearing for real converted URDFs.
fn format_py_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_owned();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".to_owned() } else { "-inf".to_owned() };
    }
    if f == 0.0 {
        // Python distinguishes 0.0 / -0.0 in str().
        return if f.is_sign_negative() {
            "-0.0".to_owned()
        } else {
            "0.0".to_owned()
        };
    }

    let neg = f.is_sign_negative();
    let mag = f.abs();

    // `{:e}` => "<m>e<exp>" with the shortest round-trip mantissa, e.g.
    // "1.2345678901234568e17", "5e-1", "9.999999999999998e15".
    let sci = format!("{mag:e}");
    let (mantissa, exp_str) = sci.split_once('e').expect("{:e} output always has 'e'");
    let exp: i32 = exp_str.parse().expect("{:e} exponent is a valid integer");
    // The shortest significant digit string (mantissa without the '.').
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    // decimal point position relative to the first significant digit.
    let decpt = exp + 1;

    let body = if decpt <= -4 || decpt > 16 {
        // Scientific: d[.ddd]e[+-]NN.
        let lead = &digits[..1];
        let rest = &digits[1..];
        let mant = if rest.is_empty() {
            lead.to_owned()
        } else {
            format!("{lead}.{rest}")
        };
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mant}e{sign}{:02}", exp.abs())
    } else if decpt <= 0 {
        // 0.00ddd (the value is < 1, leading zeros after the point).
        let zeros = (-decpt) as usize;
        format!("0.{}{}", "0".repeat(zeros), digits)
    } else {
        let dp = decpt as usize;
        if dp >= digits.len() {
            // Integral value: pad to the decimal point, then a trailing ".0".
            let pad = dp - digits.len();
            format!("{}{}.0", digits, "0".repeat(pad))
        } else {
            format!("{}.{}", &digits[..dp], &digits[dp..])
        }
    };

    if neg {
        format!("-{body}")
    } else {
        body
    }
}

/// Python `repr()` of a string: single-quoted, with single quotes escaped.
fn py_str_repr(s: &str) -> String {
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
}
