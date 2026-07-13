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
                Event::SequenceStart(_, tag) => {
                    validate_collection_tag(tag.as_ref(), Collection::Seq)?;
                    stack.push(Frame::Seq(Vec::new()));
                }
                Event::MappingStart(_, tag) => {
                    validate_collection_tag(tag.as_ref(), Collection::Map)?;
                    stack.push(Frame::Map {
                        map: IndexMap::new(),
                        key: None,
                    });
                }
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

/// Which kind of collection a start event opened, so tag validation can require
/// the matching core-schema tag (`!!seq` for a sequence, `!!map` for a mapping).
#[derive(Clone, Copy)]
enum Collection {
    Seq,
    Map,
}

/// Validate the tag on a `SequenceStart`/`MappingStart`. An untagged collection
/// is fine. PyYAML's SafeLoader accepts the standard core-schema `!!seq` on a
/// sequence and `!!map` on a mapping (they name the very structure the untagged
/// node already builds), so we honor exactly those. Every other tag on a
/// collection has no constructor here, so we raise like SafeLoader instead of
/// silently dropping the tag: a local `!custom [1, 2]`, a mismatched
/// `!!seq {a: 1}`, a named/verbatim tag. This mirrors the scalar path, where an
/// unrecognized tag is an error rather than a stringified value.
fn validate_collection_tag(
    tag: Option<&yaml_rust2::parser::Tag>,
    kind: Collection,
) -> Result<(), EvalError> {
    let Some(tag) = tag else {
        return Ok(());
    };
    // Core-schema `!!` tags: yaml-rust2 expands the shorthand to the standard
    // `tag:yaml.org,2002:` prefix. Accept only the tag naming THIS collection.
    if tag.handle == "tag:yaml.org,2002:" {
        let matches = match kind {
            Collection::Seq => tag.suffix == "seq",
            Collection::Map => tag.suffix == "map",
        };
        if matches {
            return Ok(());
        }
        return Err(unknown_tag_error(&format!(
            "tag:yaml.org,2002:{}",
            tag.suffix
        )));
    }
    // A local `!suffix` tag (handle `!`) or any other handle: unrecognized.
    Err(unknown_tag_error(&format!("{}{}", tag.handle, tag.suffix)))
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
        // Local `!suffix` tags: the unit constructors scale their scalar; any
        // other local tag is unrecognized. PyYAML's SafeLoader raises on an
        // unknown tag rather than silently producing a string, and so do we: a
        // typo like `!radian` is a mistake, not a string, and hiding it lets it
        // resurface later as a confusing type error.
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
            return Err(unknown_tag_error(&format!("!{}", tag.suffix)));
        }
        // Core-schema `!!` tags: yaml-rust2 expands the `!!` shorthand to the
        // standard `tag:yaml.org,2002:` prefix. An explicit tag FORCES the type
        // (`!!str 5` is the string "5", `!!int 5` is the integer 5) instead of the
        // implicit resolution a plain scalar would get; matching PyYAML.
        if tag.handle == "tag:yaml.org,2002:" {
            return core_schema_scalar(&tag.suffix, text);
        }
        // Any other tag (a named `!handle!suffix`, a verbatim URI, ...) is
        // unrecognized.
        return Err(unknown_tag_error(&format!("{}{}", tag.handle, tag.suffix)));
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

/// Build the "unknown tag" error, matching PyYAML's `ConstructorError` phrasing
/// ("could not determine a constructor for the tag '...'").
fn unknown_tag_error(tag: &str) -> EvalError {
    EvalError::Runtime(format!("could not determine a constructor for the tag '{tag}'"))
}

/// Construct a scalar explicitly typed by a core-schema `!!` tag, matching
/// PyYAML's SafeLoader constructors. The tag FORCES the type: `!!str 5` is the
/// string "5", `!!int 5` is the integer 5. A value that cannot be constructed for
/// the tag (`!!int abc`), or a core-schema tag outside the scalar set this subset
/// supports (`!!seq`, `!!binary`, ...), errors rather than silently stringifying.
fn core_schema_scalar(suffix: &str, text: &str) -> Result<XacroValue, EvalError> {
    match suffix {
        "str" => Ok(XacroValue::Str(text.to_owned())),
        // PyYAML's `construct_yaml_null` yields None for any `!!null` scalar.
        "null" => Ok(XacroValue::Null),
        "bool" => parse_core_bool(text)
            .map(XacroValue::Bool)
            .ok_or_else(|| EvalError::Runtime(format!("!!bool value is not a boolean: {text}"))),
        "int" => parse_core_int(text)
            .map(XacroValue::Int)
            .ok_or_else(|| EvalError::Runtime(format!("!!int value is not an integer: {text}"))),
        "float" => parse_core_float(text)
            .map(XacroValue::Float)
            .ok_or_else(|| EvalError::Runtime(format!("!!float value is not a float: {text}"))),
        other => Err(unknown_tag_error(&format!("tag:yaml.org,2002:{other}"))),
    }
}

/// Parse a core-schema boolean (`!!bool`) the way PyYAML's SafeConstructor does.
/// `construct_yaml_bool` lowercases the scalar and looks it up in a fixed table
/// (`yes`/`true`/`on` -> true, `no`/`false`/`off` -> false), so an EXPLICIT
/// `!!bool` accepts ANY casing of those six spellings (`yEs`, `On`, `OFF`, ...);
/// anything else is not a boolean. (This is broader than the implicit plain-
/// scalar resolver, which is left untouched.)
fn parse_core_bool(text: &str) -> Option<bool> {
    match text.to_ascii_lowercase().as_str() {
        "yes" | "true" | "on" => Some(true),
        "no" | "false" | "off" => Some(false),
        _ => None,
    }
}

/// Parse a core-schema integer (`!!int`), a direct port of PyYAML's
/// `SafeConstructor.construct_yaml_int`. After stripping `_` separators and an
/// optional sign it dispatches by prefix, arbitrary precision throughout:
///   * `0` -> zero;
///   * `0b...` -> binary (lowercase `b` only; `0B...` falls through to octal and
///     fails, matching PyYAML's case-sensitive `startswith('0b')`);
///   * `0x...` -> hexadecimal (lowercase `x` only, same reason);
///   * a leading `0` (including `0o`/`0O`) -> octal, so `017`, `0o17`, `0O17` are
///     15 but `08` errors;
///   * a `:`-separated value -> sexagesimal (base 60), so `1:30` is 90 and
///     `190:20:30` is 685230. PyYAML applies this base-60 form and we match it;
///     note the octal branch is checked FIRST, so `0:30` errors just as it does
///     in PyYAML;
///   * otherwise decimal.
fn parse_core_int(text: &str) -> Option<num_bigint::BigInt> {
    use num_bigint::BigInt;
    // PyYAML strips underscores from the whole scalar before anything else.
    let stripped = text.replace('_', "");
    let bytes = stripped.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let negative = bytes[0] == b'-';
    let body: &str = if bytes[0] == b'+' || bytes[0] == b'-' {
        &stripped[1..]
    } else {
        &stripped
    };
    if body.is_empty() {
        return None;
    }
    let magnitude: BigInt = if body == "0" {
        BigInt::from(0)
    } else if let Some(bin) = body.strip_prefix("0b") {
        BigInt::parse_bytes(bin.as_bytes(), 2)?
    } else if let Some(hex) = body.strip_prefix("0x") {
        BigInt::parse_bytes(hex.as_bytes(), 16)?
    } else if body.starts_with('0') {
        // Leading-zero octal. Python's `int(value, 8)` also accepts a `0o`/`0O`
        // prefix, so strip that when present; otherwise the leading zeros parse
        // fine in base 8. A digit outside 0-7 (`08`) yields None, i.e. an error.
        let digits = body
            .strip_prefix("0o")
            .or_else(|| body.strip_prefix("0O"))
            .unwrap_or(body);
        BigInt::parse_bytes(digits.as_bytes(), 8)?
    } else if body.contains(':') {
        // Sexagesimal: parse each `:`-separated part in base 10, most significant
        // first, accumulating with base 60 (PyYAML reverses the parts and walks
        // powers of 60). Any empty or non-decimal part yields None (an error).
        let mut acc = BigInt::from(0);
        let mut place = BigInt::from(1);
        for part in body.rsplit(':') {
            let digit = part.parse::<BigInt>().ok()?;
            acc += &digit * &place;
            place *= 60;
        }
        acc
    } else {
        BigInt::parse_bytes(body.as_bytes(), 10)?
    };
    Some(if negative { -magnitude } else { magnitude })
}

/// Parse a core-schema float (`!!float`). An EXPLICIT `!!float` tag is more
/// permissive than the implicit plain-scalar resolver ([`is_pyyaml_float`], which
/// requires a decimal point and a signed exponent): PyYAML's float constructor
/// accepts an unsigned exponent and a point-less mantissa like `1e16`, so a plain
/// Rust float parse (with the infinity/nan spellings handled first) matches it.
fn parse_core_float(text: &str) -> Option<f64> {
    match text {
        ".inf" | ".Inf" | ".INF" | "+.inf" | "+.Inf" | "+.INF" => return Some(f64::INFINITY),
        "-.inf" | "-.Inf" | "-.INF" => return Some(f64::NEG_INFINITY),
        ".nan" | ".NaN" | ".NAN" => return Some(f64::NAN),
        _ => {}
    }
    text.replace('_', "").parse::<f64>().ok()
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
    fn unknown_local_tag_errors() {
        // A typo'd unit tag (unknown local `!` tag) errors instead of silently
        // becoming a string, matching PyYAML's SafeLoader.
        let err = load_yaml_str("x: !radian 5\n").unwrap_err();
        assert!(
            format!("{err}").contains("constructor for the tag"),
            "expected an unknown-tag error, got: {err}"
        );
    }

    #[test]
    fn core_schema_int_tag_is_int() {
        // `!!int 5` forces an integer.
        let v = load_yaml_str("x: !!int 5\n").unwrap();
        let d = match v {
            XacroValue::Dict(d) => d,
            other => panic!("expected dict, got {other:?}"),
        };
        assert_eq!(d["x"], XacroValue::int(5));
    }

    #[test]
    fn core_schema_str_tag_is_string() {
        // `!!str 5` forces the string "5" (no numeric resolution).
        let v = load_yaml_str("x: !!str 5\n").unwrap();
        let d = match v {
            XacroValue::Dict(d) => d,
            other => panic!("expected dict, got {other:?}"),
        };
        assert_eq!(d["x"], XacroValue::Str("5".to_owned()));
    }

    #[test]
    fn core_schema_float_bool_null_tags() {
        // The remaining core-schema `!!` scalar tags construct correctly.
        let v = load_yaml_str("f: !!float 5\nb: !!bool true\nn: !!null ~\n").unwrap();
        let d = match v {
            XacroValue::Dict(d) => d,
            other => panic!("expected dict, got {other:?}"),
        };
        assert_eq!(d["f"], XacroValue::Float(5.0));
        assert_eq!(d["b"], XacroValue::Bool(true));
        assert_eq!(d["n"], XacroValue::Null);
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

    #[test]
    fn unknown_collection_tag_errors() {
        // An unknown local tag on a collection errors like SafeLoader instead of
        // silently passing the un-tagged structure (`!custom [1, 2]`).
        let err = load_yaml_str("x: !custom [1, 2]\n").unwrap_err();
        assert!(
            format!("{err}").contains("constructor for the tag '!custom'"),
            "expected an unknown-tag error, got: {err}"
        );
    }

    #[test]
    fn core_schema_seq_and_map_tags_accepted() {
        // SafeLoader accepts `!!seq` on a sequence and `!!map` on a mapping; they
        // build the same structure the untagged node would.
        let v = load_yaml_str("s: !!seq [1, 2]\nm: !!map {a: 1}\n").unwrap();
        let d = match v {
            XacroValue::Dict(d) => d,
            other => panic!("expected dict, got {other:?}"),
        };
        assert_eq!(
            d["s"],
            XacroValue::List(vec![XacroValue::int(1), XacroValue::int(2)])
        );
        let m = match &d["m"] {
            XacroValue::Dict(m) => m,
            other => panic!("m not a dict: {other:?}"),
        };
        assert_eq!(m["a"], XacroValue::int(1));
    }

    #[test]
    fn mismatched_collection_tag_errors() {
        // `!!seq` on a mapping node is a type mismatch; SafeLoader raises and so
        // do we (rather than silently building a dict).
        let err = load_yaml_str("x: !!seq {a: 1}\n").unwrap_err();
        assert!(
            format!("{err}").contains("constructor for the tag 'tag:yaml.org,2002:seq'"),
            "expected a mismatched-tag error, got: {err}"
        );
    }

    #[test]
    fn core_bool_tag_extra_spellings() {
        // An explicit `!!bool` accepts PyYAML's full yes/no/on/off set in any
        // casing (the constructor lowercases and table-looks-up).
        for (text, want) in [
            ("yes", true),
            ("Yes", true),
            ("YES", true),
            ("yEs", true),
            ("on", true),
            ("On", true),
            ("no", false),
            ("off", false),
            ("OFF", false),
        ] {
            let src = format!("x: !!bool {text}\n");
            let v = load_yaml_str(&src).unwrap();
            let d = match v {
                XacroValue::Dict(d) => d,
                other => panic!("expected dict, got {other:?}"),
            };
            assert_eq!(d["x"], XacroValue::Bool(want), "!!bool {text}");
        }
    }

    #[test]
    fn core_bool_tag_rejects_non_bool() {
        // Spellings PyYAML's bool constructor does NOT accept error under an
        // explicit tag (`y`/`n` are resolver-only, never constructor keys).
        for text in ["y", "n", "tru", "1"] {
            let src = format!("x: !!bool {text}\n");
            assert!(
                load_yaml_str(&src).is_err(),
                "!!bool {text} should error"
            );
        }
    }

    #[test]
    fn core_int_tag_extra_radixes() {
        // Forms PyYAML's int constructor accepts beyond plain decimal/hex.
        for (text, want) in [
            ("0b101", 5_i64),
            ("0o17", 15),
            ("0O17", 15),
            ("017", 15),
            ("0777", 511),
            ("1_000", 1000),
            ("0x1f", 31),
            ("1:30", 90),
            ("190:20:30", 685230),
            ("-0b101", -5),
            ("+017", 15),
        ] {
            let src = format!("x: !!int {text}\n");
            let v = load_yaml_str(&src).unwrap();
            let d = match v {
                XacroValue::Dict(d) => d,
                other => panic!("expected dict, got {other:?}"),
            };
            assert_eq!(d["x"], XacroValue::int(want), "!!int {text}");
        }
    }

    #[test]
    fn core_int_tag_rejects_bad_forms() {
        // Forms PyYAML's int constructor rejects (uppercase radix prefixes route
        // through the octal branch and fail; `08` is not octal; `0:30` hits the
        // octal branch before sexagesimal; a float is not an int).
        for text in ["0B101", "0X1F", "08", "0:30", "0xg", "1.5", "abc"] {
            let src = format!("x: !!int {text}\n");
            assert!(load_yaml_str(&src).is_err(), "!!int {text} should error");
        }
    }
}
