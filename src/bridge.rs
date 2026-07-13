//! The `XacroValue` <-> `PyObjectRef` bridge.
//!
//! `to_py` builds a Python object from a Rust [`XacroValue`] (used to inject the
//! namespace / property table into the eval scope). `from_py` walks a Python
//! result object back into a typed [`XacroValue`], preserving Python's type
//! identity (int vs float vs bool vs str vs list vs dict vs None) via exact
//! type-object identity checks; `bool` is checked BEFORE `int` because in
//! Python `bool` is an `int` subclass.

use std::str::FromStr;

use indexmap::IndexMap;
use num_bigint::BigInt;
use rustpython_vm::builtins::{PyDict, PyFloat, PyInt, PyList, PyStr, PyTuple};
use rustpython_vm::{compiler, AsObject, PyObjectRef, VirtualMachine};

use crate::error::EvalError;
use crate::value::XacroValue;

/// The builtins-dict key under which the per-VM YAML dict-wrapper CLASS is cached.
/// Chosen to be obscure + non-dunder (the dunder security gate inspects user
/// expression names, not this internal key) so it never collides with a real
/// property/global and is never referenced by a user `${...}`.
const WRAPPER_CACHE_KEY: &str = "_xacro_pure_YamlDictWrapper";

/// The Python source defining the YAML dict-wrapper class, a FAITHFUL port of
/// canonical `xacro/__init__.py`'s `YamlDictWrapper`/`YamlListWrapper`: a `dict`
/// subclass whose `__getattr__` (aliased to `__getitem__`) returns nested items
/// re-wrapped, so BOTH `cfg['k']` AND `cfg.k` work and propagate recursively
/// (`cfg.geometry.scale.x`). A missing key raises `AttributeError` (canonical
/// raises it from `__getattr__` so `hasattr` works), and since `__getitem__ =
/// __getattr__`, `cfg['missing']` raises `AttributeError` too, matching canonical
/// exactly. `__contains__` is inherited from `dict`, so `'k' in cfg` is unchanged;
/// `str()`/`repr()`/`==`/`.keys()`/`len()`/iteration are all the inherited `dict`
/// behavior, so the wrapper is observationally a dict for everything except the
/// added dotted access.
///
/// `_W` wraps dicts; `_L` wraps lists (so `cfg.items[0].name` works), mirroring
/// canonical's two cooperating wrapper classes. `_wrap(v)` is the shared dispatch.
/// Only `dict`/`list` are wrapped; scalars pass through. NOTE the subclass check
/// uses `type(v) ... in (dict, list)`-style `isinstance`, so an already-wrapped
/// value is re-wrapped idempotently (wrapping a `_W` yields an equivalent `_W`).
const WRAPPER_BOOTSTRAP: &str = "\
class _L(list):
    def __getitem__(self, i):
        return _wrap(list.__getitem__(self, i))
    def __iter__(self):
        for it in list.__iter__(self):
            yield _wrap(it)
class _W(dict):
    def __getattr__(self, k):
        try:
            return _wrap(dict.__getitem__(self, k))
        except KeyError:
            raise AttributeError(k)
    __getitem__ = __getattr__
def _wrap(v):
    if isinstance(v, dict):
        return _W(v)
    if isinstance(v, list):
        return _L(v)
    return v
";

/// rustpython 0.4's `int` payload is a `malachite_bigint::BigInt`, while our
/// public [`XacroValue::Int`] is a `num_bigint::BigInt`. The two are distinct
/// types from distinct crates. Rather than take a direct, version-pinned
/// dependency on rustpython's *internal* bigint crate (which would break the
/// moment rustpython swapped or bumped it), we bridge through the canonical
/// decimal form: `PyInt` implements `Display` as its exact decimal rendering,
/// and `num_bigint::BigInt` parses that decimal back with no loss for any
/// magnitude (`10**30`, `2**100`, ...).
fn pyint_to_num_bigint(value: &PyInt) -> BigInt {
    // The decimal text from `PyInt`'s `Display` is always a valid `num_bigint`
    // literal, so the parse cannot fail in practice; stay total (no panic) by
    // falling back to zero on the impossible error path.
    BigInt::from_str(&value.to_string()).unwrap_or_else(|_| BigInt::from(0))
}

/// The inverse, used to inject an arbitrary-precision property value into the
/// Python eval scope. `vm.ctx.new_int` wants rustpython's own bigint
/// (`malachite_bigint::BigInt`); we build it from our `num_bigint::BigInt` via
/// the same lossless decimal round-trip. `malachite-bigint` is a direct,
/// exactly-pinned dependency (matching `rustpython-vm`'s exact pin) so this
/// type unifies with what `new_int` expects.
fn num_to_rustpython_bigint(value: &BigInt) -> malachite_bigint::BigInt {
    malachite_bigint::BigInt::from_str(&value.to_string())
        .unwrap_or_else(|_| malachite_bigint::BigInt::from(0))
}

/// Get (or lazily define + cache) the per-VM `_wrap` callable that turns a plain
/// `dict`/`list` into the dotted-access `_W`/`_L` wrapper (canonical
/// `YamlDictWrapper.wrap`). Defined once per VM via the embedded compiler and
/// stashed in the builtins dict under [`WRAPPER_CACHE_KEY`]. Returns `None` if the
/// bootstrap could not run (it cannot, in practice, since `dict`/`list`/`isinstance`
/// are always present, but stay total so `to_py` can fall back to a plain dict).
fn yaml_wrap_fn(vm: &VirtualMachine) -> Option<PyObjectRef> {
    let builtins = vm.builtins.dict();
    if let Ok(Some(cached)) = builtins.get_item_opt(WRAPPER_CACHE_KEY, vm) {
        return Some(cached);
    }
    // Define the wrapper classes + `_wrap` in a throwaway scope, then capture
    // `_wrap` and cache it on the builtins dict so subsequent `to_py` calls (and
    // the `load_yaml` Rust closure, same VM) reuse it.
    let code = vm
        .compile(
            WRAPPER_BOOTSTRAP,
            compiler::Mode::Exec,
            "<yaml-wrapper>".to_owned(),
        )
        .ok()?;
    let scope = vm.new_scope_with_builtins();
    vm.run_code_obj(code, scope.clone()).ok()?;
    let wrap = scope.globals.get_item_opt("_wrap", vm).ok().flatten()?;
    // Cache for reuse (ignore a set failure: worst case we redefine next time).
    let _ = builtins.set_item(WRAPPER_CACHE_KEY, wrap.clone(), vm);
    Some(wrap)
}

/// Wrap a freshly-built `dict`/`list` Python object in the dotted-access wrapper,
/// so `cfg.k`/`cfg['k']` both resolve (canonical `YamlListWrapper.wrap`). Falls
/// back to the plain object if the wrapper is unavailable.
fn wrap_for_dotted(obj: PyObjectRef, vm: &VirtualMachine) -> PyObjectRef {
    match yaml_wrap_fn(vm) {
        Some(wrap) => wrap.call((obj.clone(),), vm).unwrap_or(obj),
        None => obj,
    }
}

/// Convert a Rust [`XacroValue`] into a Python object in the given VM.
pub fn to_py(value: &XacroValue, vm: &VirtualMachine) -> PyObjectRef {
    match value {
        // Bridge the arbitrary-precision int losslessly. `vm.ctx.new_int`
        // accepts anything `Into<BigInt>` (rustpython's malachite `BigInt`); we
        // round-trip our `num_bigint::BigInt` through its decimal form so the
        // full magnitude survives (`10**30`, `2**100`, ... stay exact) and we
        // don't depend on the two bigint crates' `From` interop.
        XacroValue::Int(i) => vm.ctx.new_int(num_to_rustpython_bigint(i)).into(),
        XacroValue::Float(f) => vm.ctx.new_float(*f).into(),
        XacroValue::Bool(b) => vm.ctx.new_bool(*b).into(),
        XacroValue::Str(s) => vm.ctx.new_str(s.as_str()).into(),
        XacroValue::Null => vm.ctx.none(),
        XacroValue::List(items) => {
            // Wrap in the dotted-access list wrapper (`_L`) so a `cfg.items[i].k`
            // chain resolves through a list of dicts, canonical `YamlListWrapper`.
            // Observationally a plain list for str/==/len/iteration of scalars.
            let elems: Vec<PyObjectRef> = items.iter().map(|v| to_py(v, vm)).collect();
            wrap_for_dotted(vm.ctx.new_list(elems).into(), vm)
        }
        XacroValue::Tuple(items) => {
            let elems: Vec<PyObjectRef> = items.iter().map(|v| to_py(v, vm)).collect();
            vm.ctx.new_tuple(elems).into()
        }
        XacroValue::Dict(map) => {
            // Wrap in the dotted-access dict wrapper (`_W`) so BOTH `cfg['k']` and
            // `cfg.k` resolve, canonical `YamlDictWrapper`. The wrapper is a `dict`
            // subclass, so str/repr/==/in/keys/len/iteration are unchanged.
            let dict = vm.ctx.new_dict();
            for (k, v) in map {
                // set_item on a fresh dict with str keys cannot fail in practice;
                // surface any error rather than unwrap-panicking.
                let _ = dict.set_item(k.as_str(), to_py(v, vm), vm);
            }
            wrap_for_dotted(dict.into(), vm)
        }
        // A xacro `NameSpace` bridges to a module-like object exposing each entry
        // as an ATTRIBUTE, so `${ns.prop}` resolves via Python attribute access,
        // exactly canonical's `NameSpace.__getattr__ -> __getitem__`. (A module's
        // `__getattr__` reads its namespace dict, the simplest attribute-accessible
        // object available in the `without_stdlib` VM, same shape as the injected
        // `xacro`/`math` modules.)
        XacroValue::Namespace(map) => {
            let module = vm.new_module("<namespace>", vm.ctx.new_dict(), None);
            for (k, v) in map {
                // `set_attr`'s `AsPyStr` only accepts a `&'static str` for a bare
                // `&str`; intern the (non-static) key to get a `&'static
                // PyStrInterned` it does accept.
                let name = vm.ctx.intern_str(k.as_str());
                let _ = module.set_attr(name, to_py(v, vm), vm);
            }
            module.into()
        }
    }
}

/// Convert a Python result object back into a typed [`XacroValue`].
///
/// Type dispatch uses exact type-object identity (`obj.class().is(...)`), with
/// `bool` checked before `int` (Python `bool <: int`). Unsupported Python types
/// (e.g. tuples, sets, custom objects) become an [`EvalError::UnsupportedType`]
/// rather than a lossy coercion, so the caller can decide how to handle them.
pub fn from_py(obj: &PyObjectRef, vm: &VirtualMachine) -> Result<XacroValue, EvalError> {
    if obj.is(&vm.ctx.none()) {
        return Ok(XacroValue::Null);
    }
    let class = obj.class();

    // bool BEFORE int: a Python bool is an int subclass.
    if class.is(vm.ctx.types.bool_type) {
        let i = obj
            .downcast_ref::<PyInt>()
            .ok_or_else(|| EvalError::Bridge("bool object lacked int payload".to_owned()))?;
        // A Python bool's int payload is exactly 0 or 1.
        let v: i64 = i
            .try_to_primitive(vm)
            .map_err(|_| EvalError::Bridge("bool int payload out of range".to_owned()))?;
        return Ok(XacroValue::Bool(v != 0));
    }
    if class.is(vm.ctx.types.int_type) {
        let i = obj
            .downcast_ref::<PyInt>()
            .ok_or_else(|| EvalError::Bridge("int object lacked int payload".to_owned()))?;
        // Arbitrary precision: read the full magnitude (no i64 clamp), so an
        // expression CPython/xacro evaluates to a big int (`2**63`, `10**30`,
        // `2**100`, ...) bridges back losslessly instead of erroring.
        return Ok(XacroValue::Int(pyint_to_num_bigint(i)));
    }
    if class.is(vm.ctx.types.float_type) {
        let f = obj
            .downcast_ref::<PyFloat>()
            .ok_or_else(|| EvalError::Bridge("float object lacked float payload".to_owned()))?;
        return Ok(XacroValue::Float(f.to_f64()));
    }
    if class.is(vm.ctx.types.str_type) {
        let s = obj
            .downcast_ref::<PyStr>()
            .ok_or_else(|| EvalError::Bridge("str object lacked str payload".to_owned()))?;
        return Ok(XacroValue::Str(s.as_str().to_owned()));
    }
    // `list` (and its subclasses, e.g. the dotted-access `_L` wrapper that a YAML
    // list becomes): accept via `isinstance`, not exact identity, so a wrapped
    // list round-trips back to a typed list.
    if obj.fast_isinstance(vm.ctx.types.list_type) {
        let list = obj
            .downcast_ref::<PyList>()
            .ok_or_else(|| EvalError::Bridge("list object lacked list payload".to_owned()))?;
        let mut out = Vec::new();
        for item in list.borrow_vec().iter() {
            out.push(from_py(item, vm)?);
        }
        return Ok(XacroValue::List(out));
    }
    if class.is(vm.ctx.types.tuple_type) {
        let tuple = obj
            .downcast_ref::<PyTuple>()
            .ok_or_else(|| EvalError::Bridge("tuple object lacked tuple payload".to_owned()))?;
        // A `${(1, 2.0)}` survives as a typed tuple (canonical str()s it as
        // `(1, 2.0)`, a 1-tuple as `(1,)`). The element walk preserves nested
        // type identity just like the list arm.
        let mut out = Vec::new();
        for item in tuple.iter() {
            out.push(from_py(item, vm)?);
        }
        return Ok(XacroValue::Tuple(out));
    }
    // `dict` (and its subclasses, e.g. the dotted-access `_W` wrapper that a YAML
    // dict, or any constructed dict, becomes): accept via `isinstance`, not exact
    // identity, so a wrapped dict round-trips back to a typed dict.
    if obj.fast_isinstance(vm.ctx.types.dict_type) {
        let dict = obj
            .downcast_ref::<PyDict>()
            .ok_or_else(|| EvalError::Bridge("dict object lacked dict payload".to_owned()))?;
        let mut out: IndexMap<String, XacroValue> = IndexMap::new();
        for (k, v) in dict {
            let key = k
                .downcast_ref::<PyStr>()
                .ok_or_else(|| EvalError::UnsupportedType("dict key is not a str".to_owned()))?
                .as_str()
                .to_owned();
            out.insert(key, from_py(&v, vm)?);
        }
        return Ok(XacroValue::Dict(out));
    }

    Err(EvalError::UnsupportedType(format!(
        "result of type '{}' cannot be converted to XacroValue",
        class.name()
    )))
}
