//! The `xacro.load_yaml` builtin: parse a YAML document into a typed
//! [`XacroValue`] (a dict/list/scalar tree), applying the `!radians`/`!degrees`/
//! `!meters`/`!millimeters`/`!foot`/`!inches` UNIT CONSTRUCTORS exactly as
//! canonical xacro's `ConstructUnits` does.
//!
//! Faithful to canonical `xacro/__init__.py::load_yaml` + `ConstructUnits`
//! (`__init__.py:113-149`):
//!   * each unit tag's scalar is `safe_eval`'d as a Python expression and
//!     multiplied by the unit's conversion constant
//!     (`!radians`=1.0, `!degrees`=pi/180, `!meters`=1.0, `!millimeters`=0.001,
//!     `!foot`=0.3048, `!inches`=0.0254);
//!   * the result is wrapped so that downstream `cfg['k']` subscripting yields a
//!     nested dict/list/scalar.
//!
//! We drive `yaml-rust2`'s lower-level EVENT parser (not its high-level `Yaml`
//! enum, which DROPS custom tags) so the `!radians`/... tags survive and can be
//! intercepted. Plain scalars get YAML's standard core-schema type resolution
//! (int/float/bool/null/str), matching what PyYAML's `safe_load` produces.
//!
//! ## Dotted attribute access (`cfg.angle`)
//! Canonical wraps the dict in `YamlDictWrapper`, whose `__getattr__` enables
//! `cfg.angle` in ADDITION to `cfg['angle']`. The bridge reproduces that:
//! [`crate::bridge::to_py`] wraps every dict/list in the `_W`/`_L` Python
//! wrapper classes (faithful `YamlDictWrapper`/`YamlListWrapper` ports defined
//! once per VM), so `cfg.geometry.scale.x` chains resolve exactly as canonical;
//! the OpenArm corpus test drives this end-to-end.

use indexmap::IndexMap;
use yaml_rust2::parser::{Event, Parser};
use yaml_rust2::scanner::TScalarStyle;

use crate::error::EvalError;
use crate::eval::safe_eval;
use crate::namespace::Namespace;
use crate::value::XacroValue;

/// The unit constructors, mirroring canonical `ConstructUnits`: `(tag-suffix,
/// conversion-constant)`. The handle is always `!` (a local tag), so we match on
/// the suffix.
const UNIT_CONSTRUCTORS: &[(&str, f64)] = &[
    ("radians", 1.0),
    ("degrees", std::f64::consts::PI / 180.0),
    ("meters", 1.0),
    ("millimeters", 0.001),
    ("foot", 0.3048),
    ("inches", 0.0254),
];

/// Parse `yaml_src` into a typed [`XacroValue`], applying the unit constructors.
/// This is the pure, filesystem-free core (the document pipeline reads the file
/// and calls this), which also keeps it wasm-testable.
pub fn load_yaml_str(yaml_src: &str) -> Result<XacroValue, EvalError> {
    let mut parser = Parser::new_from_str(yaml_src);
    // Walk the event stream into a value. We hold a small explicit stack so that
    // nested mappings/sequences build without recursion over the borrow.
    Builder::new().run(&mut parser)
}

/// Register `load_yaml(src)` into a [`Namespace`] as a Rust callable, for the
/// `${xacro.load_yaml(...)}` seam. The string-keyed dict it returns supports
/// `['k']` subscripting inside the expression. (We expose it under the bare name
/// `load_yaml` AND would expose `xacro.load_yaml` once a namespace object exists;
/// the document pipeline registers the bare name, matching canonical's
/// backwards-compatible direct exposure.)
///
/// NOTE: this takes the YAML *source string* (not a filename) so it stays pure /
/// wasm-clean; the document pipeline is responsible for reading the file (or an
/// injected virtual file map) and registering a closure that calls
/// [`load_yaml_str`].
pub fn register_load_yaml_from_source<F>(ns: &mut Namespace, name: &'static str, read: F)
where
    F: Fn(&str) -> Result<String, String> + Clone + 'static,
{
    // Share one reader between the bare `load_yaml` callable AND the namespace's
    // YAML-reader slot (which the eval layer uses to build `xacro.load_yaml`), so
    // both the deprecated bare form and the canonical `xacro.load_yaml` resolve
    // through the identical file access.
    let reader: crate::namespace::YamlReader = std::rc::Rc::new(read);
    let bare = reader.clone();
    ns.register_raw(name, move |vm| {
        let read = bare.clone();
        vm.new_function(
            name,
            move |filename: String, vm: &rustpython_vm::VirtualMachine| -> rustpython_vm::PyResult {
                let src = read(&filename)
                    .map_err(|e| vm.new_runtime_error(format!("load_yaml('{filename}'): {e}")))?;
                let value = load_yaml_str(&src)
                    .map_err(|e| vm.new_runtime_error(format!("load_yaml('{filename}'): {e}")))?;
                Ok(crate::bridge::to_py(&value, vm))
            },
        )
        .into()
    });
    ns.set_yaml_reader(reader);
}

/// Event-driven YAML -> [`XacroValue`] builder.
struct Builder;

/// A partially-built container on the build stack.
enum Frame {
    /// A sequence accumulating its items.
    Seq(Vec<XacroValue>),
    /// A mapping; `key` holds a pending key awaiting its value.
    Map {
        map: IndexMap<String, XacroValue>,
        key: Option<String>,
    },
}

impl Builder {
    fn new() -> Self {
        Builder
    }

    /// Drive the parser to completion, returning the first document's value.
    fn run(self, parser: &mut Parser<std::str::Chars>) -> Result<XacroValue, EvalError> {
        let mut stack: Vec<Frame> = Vec::new();
        // The completed top-level value (set when the outermost container/scalar
        // finishes). An empty document yields Null (matching `safe_load('')`).
        let mut result: XacroValue = XacroValue::Null;

        loop {
            let (event, _marker) = parser
                .next_token()
                .map_err(|e| EvalError::Runtime(format!("YAML parse error: {e}")))?;
            match event {
                Event::StreamEnd => break,
                Event::StreamStart | Event::DocumentStart | Event::DocumentEnd | Event::Nothing => {}
                Event::Alias(_) => {
                    return Err(EvalError::Runtime(
                        "YAML aliases are not supported in load_yaml".to_owned(),
                    ))
                }
                Event::Scalar(value, style, _anchor, tag) => {
                    let v = scalar_to_value(&value, style, tag.as_ref())?;
                    Self::emit(&mut stack, &mut result, v);
                }
                Event::SequenceStart(_, _) => stack.push(Frame::Seq(Vec::new())),
                Event::MappingStart(_, _) => stack.push(Frame::Map {
                    map: IndexMap::new(),
                    key: None,
                }),
                Event::SequenceEnd => {
                    let frame = stack.pop().ok_or_else(unbalanced)?;
                    let v = match frame {
                        Frame::Seq(items) => XacroValue::List(items),
                        Frame::Map { .. } => return Err(unbalanced()),
                    };
                    Self::emit(&mut stack, &mut result, v);
                }
                Event::MappingEnd => {
                    let frame = stack.pop().ok_or_else(unbalanced)?;
                    let v = match frame {
                        Frame::Map { map, key: None } => XacroValue::Dict(map),
                        Frame::Map { key: Some(_), .. } => return Err(unbalanced()),
                        Frame::Seq(_) => return Err(unbalanced()),
                    };
                    Self::emit(&mut stack, &mut result, v);
                }
            }
        }
        Ok(result)
    }

    /// Place a completed value `v` into the current container (or as the result
    /// when the stack is empty / a map is awaiting its key).
    fn emit(stack: &mut [Frame], result: &mut XacroValue, v: XacroValue) {
        match stack.last_mut() {
            None => *result = v,
            Some(Frame::Seq(items)) => items.push(v),
            Some(Frame::Map { map, key }) => match key.take() {
                // No pending key -> this scalar IS the key.
                None => *key = Some(v.to_python_str()),
                // Pending key -> this value completes the pair.
                Some(k) => {
                    map.insert(k, v);
                }
            },
        }
    }
}

/// Build the "unbalanced YAML events" error (an internal-consistency failure
/// from the parser the canonical loader would never hit on valid YAML).
fn unbalanced() -> EvalError {
    EvalError::Runtime("unbalanced YAML container events".to_owned())
}

/// Convert one YAML scalar (its text, style, and optional tag) to an
/// [`XacroValue`]. A unit-constructor tag (`!radians`, ...) `safe_eval`s the
/// scalar and scales it. A plain (unquoted, untagged) scalar gets YAML core-
/// schema type resolution; a quoted scalar is always a string.
fn scalar_to_value(
    text: &str,
    style: TScalarStyle,
    tag: Option<&yaml_rust2::parser::Tag>,
) -> Result<XacroValue, EvalError> {
    if let Some(tag) = tag {
        // Local `!suffix` unit tags.
        if tag.handle == "!" {
            if let Some(&(_, factor)) = UNIT_CONSTRUCTORS.iter().find(|(s, _)| *s == tag.suffix) {
                // safe_eval the scalar (canonical: `float(safe_eval(value, ...))`).
                let v = safe_eval(text, &Namespace::new())
                    .map_err(|e| EvalError::Runtime(format!("invalid expression: {text} ({e})")))?;
                let n = v.as_f64().ok_or_else(|| {
                    EvalError::Runtime(format!("unit value is not numeric: {text}"))
                })?;
                return Ok(XacroValue::Float(n * factor));
            }
        }
        // An unrecognized tag -> treat the scalar as a plain string (PyYAML's
        // SafeLoader would error on an unknown tag, but for the property-model
        // subset a permissive string is the safe, non-crashing choice).
        return Ok(XacroValue::Str(text.to_owned()));
    }

    // A quoted scalar is always a string (no type resolution).
    if matches!(
        style,
        TScalarStyle::SingleQuoted | TScalarStyle::DoubleQuoted
    ) {
        return Ok(XacroValue::Str(text.to_owned()));
    }

    Ok(resolve_plain_scalar(text))
}

/// YAML 1.1/1.2 core-schema resolution of a PLAIN scalar (what PyYAML's
/// `safe_load` does for an unquoted token): null / bool / int / float / str.
fn resolve_plain_scalar(text: &str) -> XacroValue {
    match text {
        "" | "~" | "null" | "Null" | "NULL" => return XacroValue::Null,
        "true" | "True" | "TRUE" => return XacroValue::Bool(true),
        "false" | "False" | "FALSE" => return XacroValue::Bool(false),
        ".inf" | ".Inf" | ".INF" | "+.inf" => return XacroValue::Float(f64::INFINITY),
        "-.inf" | "-.Inf" | "-.INF" => return XacroValue::Float(f64::NEG_INFINITY),
        ".nan" | ".NaN" | ".NAN" => return XacroValue::Float(f64::NAN),
        _ => {}
    }
    // int (decimal; YAML also has 0x/0o, handled below).
    if let Some(stripped) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        if let Some(i) = num_bigint::BigInt::parse_bytes(stripped.as_bytes(), 16) {
            return XacroValue::Int(i);
        }
    }
    if let Ok(i) = text.parse::<num_bigint::BigInt>() {
        return XacroValue::Int(i);
    }
    // FLOAT: must match PyYAML's implicit float resolver, NOT Rust's permissive
    // `f64::from_str`. PyYAML's regex (yaml/resolver.py) is:
    //   ^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
    //     |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
    //     |[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$
    // i.e. a float REQUIRES a decimal point, and an exponent (if present) MUST be
    // SIGNED. So `1.0e-7` is a float but `2.5e16` (unsigned exponent) and `1e16`
    // (no point) are STRINGS, exactly what canonical xacro's `load_yaml` (PyYAML
    // `safe_load`) produces. Using Rust's parser would wrongly turn `2.5e16` into
    // a float, then stringify it as `2.5e+16` (a textual divergence from the
    // oracle, which keeps the literal string `2.5e16`).
    if is_pyyaml_float(text) {
        // The matched text is a valid Rust float literal (point present,
        // exponent signed), so this parse cannot fail.
        if let Ok(f) = text.replace('_', "").parse::<f64>() {
            return XacroValue::Float(f);
        }
    }
    XacroValue::Str(text.to_owned())
}

/// Does `text` match PyYAML's implicit float resolver? (`.inf`/`.nan` are handled
/// earlier in [`resolve_plain_scalar`]; this covers the numeric forms.) A float
/// needs a decimal point, and any exponent must carry an explicit sign.
fn is_pyyaml_float(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut i = 0;
    // optional sign
    if bytes[i] == b'+' || bytes[i] == b'-' {
        i += 1;
    }
    // Two accepted shapes: `D[D_]*.[D_]*` or `.D[D_]*` (the leading-dot form has
    // NO sign in PyYAML's second alternative; `+.5` is NOT a float there).
    let leading_dot = i < bytes.len() && bytes[i] == b'.';
    if leading_dot {
        // The leading-dot alternative in PyYAML does not allow a sign.
        if i != 0 {
            return false;
        }
        i += 1; // consume '.'
        // requires at least one digit after the point: `\.[0-9][0-9_]*`
        if i >= bytes.len() || !bytes[i].is_ascii_digit() {
            return false;
        }
        while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'_') {
            i += 1;
        }
    } else {
        // `[0-9][0-9_]*\.[0-9_]*`
        if i >= bytes.len() || !bytes[i].is_ascii_digit() {
            return false;
        }
        while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'_') {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'.' {
            return false;
        }
        i += 1; // consume '.'
        // fractional part is `[0-9_]*` (may be empty: `1.` is a float)
        while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'_') {
            i += 1;
        }
    }
    // optional exponent `[eE][-+][0-9]+`: the SIGN IS MANDATORY.
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        if i >= bytes.len() || (bytes[i] != b'+' && bytes[i] != b'-') {
            return false;
        }
        i += 1;
        if i >= bytes.len() || !bytes[i].is_ascii_digit() {
            return false;
        }
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }
    i == bytes.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_and_nesting() {
        let src = "limits:\n  lower: -1.5\n  upper: 2\nname: arm\nflags:\n  - true\n  - false\n";
        let v = load_yaml_str(src).unwrap();
        let d = match v {
            XacroValue::Dict(d) => d,
            other => panic!("expected dict, got {other:?}"),
        };
        let limits = match &d["limits"] {
            XacroValue::Dict(m) => m,
            other => panic!("limits not a dict: {other:?}"),
        };
        assert_eq!(limits["lower"], XacroValue::Float(-1.5));
        assert_eq!(limits["upper"], XacroValue::int(2));
        assert_eq!(d["name"], XacroValue::Str("arm".to_owned()));
        assert_eq!(
            d["flags"],
            XacroValue::List(vec![XacroValue::Bool(true), XacroValue::Bool(false)])
        );
    }

    #[test]
    fn unit_constructors() {
        let src = "ang: !radians 0.5\ndeg: !degrees 90\nlen: !inches 12\n";
        let v = load_yaml_str(src).unwrap();
        let d = match v {
            XacroValue::Dict(d) => d,
            other => panic!("expected dict, got {other:?}"),
        };
        assert_eq!(d["ang"], XacroValue::Float(0.5));
        // !degrees 90 = 90 * pi/180 = pi/2.
        match d["deg"] {
            XacroValue::Float(f) => assert!((f - std::f64::consts::FRAC_PI_2).abs() < 1e-12),
            ref other => panic!("deg not float: {other:?}"),
        }
        match d["len"] {
            XacroValue::Float(f) => assert!((f - 12.0 * 0.0254).abs() < 1e-12),
            ref other => panic!("len not float: {other:?}"),
        }
    }

    #[test]
    fn unit_constructor_evaluates_expression() {
        // The scalar of a unit tag is safe_eval'd (canonical: float(safe_eval(...))).
        let src = "x: !degrees 45*2\n";
        let v = load_yaml_str(src).unwrap();
        let d = match v {
            XacroValue::Dict(d) => d,
            other => panic!("expected dict, got {other:?}"),
        };
        match d["x"] {
            XacroValue::Float(f) => assert!((f - std::f64::consts::FRAC_PI_2).abs() < 1e-12),
            ref other => panic!("x not float: {other:?}"),
        }
    }
}
