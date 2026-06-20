//! `safe_eval`: the canonical `xacro/__init__.py::safe_eval` equivalent.
//!
//! Canonical xacro (re-read live at `__init__.py:239`):
//! ```python
//! def safe_eval(expr, globals, locals=None):
//!     code = compile(expr.strip(), "<expression>", "eval")
//!     invalid_names = [n for n in code.co_names if n.startswith("__")]
//!     if invalid_names: raise XacroException("Use of invalid name(s): ", ...)
//!     globals.update(__builtins__={})   # disable default builtins
//!     return eval(code, globals, locals)
//! ```
//! i.e. **full Python expression semantics** over a namespace = the property
//! table + a curated builtin set (`math.*`, allowed builtins, `xacro.*`,
//! `python.*`), with `__builtins__` blanked and dunder names rejected.
//!
//! This module reproduces that exactly on top of `rustpython-vm`:
//!   1. compile in **eval mode**,
//!   2. inspect the compiled code object's `names` and reject any starting with
//!      `__` (the same `co_names` security gate),
//!   3. build a fresh scope, inject the namespace (each property -> its value as
//!      a Python object) and the registered Rust functions, set `__builtins__`
//!      to an EMPTY dict (so default builtins are unreachable),
//!   4. run the code object and bridge the result back to an [`XacroValue`].
//!
//! The curated set is fully populated: `math.*` (bare names + the `math`
//! namespace object), the canonical `xacro` namespace object (`load_yaml`/
//! `arg`/`dotify`/`abs_filename`), and the allowed-builtins list
//! ([`ALLOWED_BUILTINS`]); everything else is shadowed to a raising sentinel
//! (see [`shadow_disallowed_builtins`]).

use std::collections::HashMap;

use rustpython_vm::bytecode::{BorrowedConstant, Constant};
use rustpython_vm::{self as vm, AsObject};

/// The substitution-args snapshot used to build the `xacro.arg` callable in the
/// eval scope. Empty when no document/substitution context is attached.
pub(crate) type ArgsSnapshot = HashMap<String, String>;

use crate::bridge::{from_py, to_py};
use crate::error::EvalError;
use crate::namespace::Namespace;
use crate::value::XacroValue;

thread_local! {
    /// ONE interpreter, REUSED across every evaluation on this thread.
    ///
    /// Building a rustpython VM (`Interpreter::without_stdlib`) constructs the whole runtime (type system,
    /// builtins) and is BY FAR the dominant cost of an eval. Doing it per `${...}` made xacro expansion
    /// O(expressions) interpreter-builds, seconds for a real robot, ~24s for OpenArm v2.0's recursive
    /// macros (the in-browser import "hang"). Reuse is the idiomatic rustpython pattern (create once,
    /// `enter()` many times) and is SAFE here: each eval builds a FRESH scope (`new_scope_with_builtins`)
    /// and injects its properties/functions there, so no state leaks between expressions (same expr +
    /// namespace -> same result); an eval-mode expression cannot assign globals or import, so it cannot
    /// mutate shared VM state. `enter` DOES nest on one path: a `${load_yaml(...)}` runs its Rust reader
    /// callback inside `run_code_obj` (inside `enter`), and a unit-tagged scalar (`!degrees 90`) then
    /// `safe_eval`s its expression, a second `enter` on the same interpreter. That is safe by design:
    /// rustpython's `enter` pushes/pops an explicit per-thread VM stack (`thread::enter_vm`), restoring
    /// the outer VM on exit, and the inner eval isolates itself in its own fresh scope like any other.
    /// Covered by `tests/pipeline.rs::load_yaml_subscript_and_units`.
    static INTERP: vm::Interpreter = vm::Interpreter::without_stdlib(Default::default());
}

/// Run `f` with the thread-local reused interpreter's VM (see [`INTERP`]).
fn with_vm<R>(f: impl FnOnce(&vm::VirtualMachine) -> R) -> R {
    INTERP.with(|interp| interp.enter(f))
}

/// Evaluate a Python expression `expr` against `namespace`, returning a typed
/// [`XacroValue`]. Mirrors canonical xacro's `safe_eval`: dunder-name rejection,
/// blanked `__builtins__`, full Python eval semantics over the injected scope.
///
/// Never panics; every failure path is an [`EvalError`].
pub fn safe_eval(expr: &str, namespace: &Namespace) -> Result<XacroValue, EvalError> {
    // Evaluation is isolated by the fresh per-call scope built in `eval_in_vm`, not by a fresh VM, so the
    // interpreter is reused across calls (see [`INTERP`]): same expr + namespace -> same result.
    let empty_args = HashMap::new();
    with_vm(|vm| eval_in_vm(vm, expr, namespace, None, &HashMap::new(), &empty_args))
}

/// Like [`safe_eval`] but injects the registered functions from a SECOND
/// namespace alongside the property bindings of the first. The property
/// model resolves the small set of properties an expression references into a
/// throwaway `props` namespace, then evaluates against that PLUS the persistent
/// registered functions (the `xacro.load_yaml` / `radians` seam), without having
/// to move the boxed function builders out of their owning namespace.
///
/// Property bindings in `props` take precedence over any same-named function in
/// `functions` (properties are injected last), matching canonical xacro where a
/// property shadows a same-named global.
pub fn safe_eval_with(
    expr: &str,
    props: &Namespace,
    functions: &Namespace,
) -> Result<XacroValue, EvalError> {
    let empty_args = HashMap::new();
    with_vm(|vm| eval_in_vm(vm, expr, props, Some(functions), &HashMap::new(), &empty_args))
}

/// Like [`safe_eval_with`] but with a `deferred_errors` map of `name -> original
/// resolution error` AND a substitution-args snapshot.
///
/// `deferred_errors` is the live-globals fidelity seam: canonical xacro passes
/// the live `Table` as `globals`, so a referenced name is `_resolve_`d only when
/// the Python bytecode actually LOADS it; a name sitting in a short-circuited /
/// dead branch (`x if True else circular`, `ok or circular`) is never forced.
/// The port resolves the referenced names up front into a flat snapshot,
/// which would prematurely force a circular/erroring name even in a dead branch.
///
/// We recover canonical's lazy behavior WITHOUT the unsafe live-VM-callback: a
/// name whose resolution ERRORED is OMITTED from the snapshot and recorded here.
/// If the VM never loads it (dead branch), the eval succeeds, matching the
/// oracle. If the VM DOES load it (live branch), Python raises `NameError` whose
/// `.name` attribute is that name; we intercept it and re-surface the ORIGINAL
/// resolution error (e.g. the exact `circular variable definition: ...` chain)
/// instead of a bare `NameError`.
///
/// `args` makes the `xacro.arg(name)` callable in the `xacro` namespace resolve
/// against the LIVE arg table (canonical `substitution_args_context['arg']`).
/// The document pipeline passes the current arg table; a pure property eval
/// passes an empty one (then `xacro.arg` raises on use, matching an absent arg).
pub fn safe_eval_deferred_with_args(
    expr: &str,
    props: &Namespace,
    functions: &Namespace,
    deferred_errors: &HashMap<String, EvalError>,
    args: &ArgsSnapshot,
) -> Result<XacroValue, EvalError> {
    with_vm(|vm| eval_in_vm(vm, expr, props, Some(functions), deferred_errors, args))
}

/// Compile `expr` in eval mode and return the global names it references (its
/// `co_names`, minus dunder names which are rejected anyway). This is the seam
/// the property model uses: canonical xacro passes the live `Table` as
/// `globals` so ONLY the names an expression touches get `_resolve_`d; we mirror
/// that by resolving exactly these names against the `Table` before building the
/// flat [`Namespace`] handed to [`safe_eval`]. Resolving only the referenced
/// names (not the whole table) is what keeps an unrelated lazy/circular property
/// from being forced, matching the live-globals behavior.
///
/// Returns the dunder-rejection error if the expression references an invalid
/// name, so the same security gate fires here as in `safe_eval`. A compile error
/// becomes [`EvalError::Compile`].
pub fn referenced_names(expr: &str) -> Result<Vec<String>, EvalError> {
    with_vm(|vm| {
        let code = vm
            .compile(expr.trim(), vm::compiler::Mode::Eval, "<expression>".to_owned())
            .map_err(|e| EvalError::Compile(format!("{e}")))?;
        let mut invalid: Vec<String> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        collect_code_names(&code.code, &mut names, &mut invalid);
        if !invalid.is_empty() {
            return Err(EvalError::InvalidName(invalid.join(", ")));
        }
        Ok(names)
    })
}

/// Recursively collect a code object's referenced global/attribute names
/// (`co_names`), DESCENDING into nested code objects held in `co_consts`.
///
/// This is load-bearing for OpenArm: canonical xacro passes the LIVE `Table` as
/// `globals`, so a name LOADed *anywhere* (including inside a list/dict/set/gen
/// comprehension or a lambda) is resolved against the property table. In CPython
/// bytecode each comprehension/lambda is compiled to its OWN nested code object
/// (a `<listcomp>`/`<genexpr>`/`<lambda>` constant), and a free variable it
/// references (e.g. `robot_preset` in `[name for name in components if name in
/// robot_preset]`) lands in the NESTED code's `co_names`, NOT the outer one. The
/// previous single-level scan missed those, so the up-front name-resolution
/// snapshot omitted them and the eval hit `NameError`. Walking the nested code
/// objects recovers canonical's behavior. Comprehension iteration variables are
/// locals (`co_varnames`/`co_cellvars`), never in `co_names`, so they are not
/// captured here (correct: they must not shadow an outer property snapshot key).
///
/// Dunder names (`__...`) are routed to `invalid` so the same `co_names` security
/// gate fires regardless of nesting depth. Names are de-duplicated, first-seen
/// order preserved.
fn collect_code_names<C: Constant>(
    code: &vm::bytecode::CodeObject<C>,
    names: &mut Vec<String>,
    invalid: &mut Vec<String>,
) {
    for n in code.names.iter() {
        let n = n.as_ref();
        if n.starts_with("__") {
            if !invalid.iter().any(|x| x == n) {
                invalid.push(n.to_owned());
            }
        } else if !names.iter().any(|x| x == n) {
            names.push(n.to_owned());
        }
    }
    for constant in code.constants.iter() {
        if let BorrowedConstant::Code { code: nested } = constant.borrow_constant() {
            collect_code_names(nested, names, invalid);
        }
    }
}

fn eval_in_vm(
    vm: &vm::VirtualMachine,
    expr: &str,
    namespace: &Namespace,
    extra_functions: Option<&Namespace>,
    deferred_errors: &HashMap<String, EvalError>,
    args: &ArgsSnapshot,
) -> Result<XacroValue, EvalError> {
    // (1) compile in eval mode. `expr.strip()` == Rust trim, matching canonical.
    let code = vm
        .compile(expr.trim(), vm::compiler::Mode::Eval, "<expression>".to_owned())
        .map_err(|e| EvalError::Compile(format!("{e}")))?;

    // (2) the security gate: reject any referenced name beginning with `__`.
    // `code.code.names` is the rustpython equivalent of CPython's `co_names`;
    // descend into nested code objects (comprehensions/lambdas) so a dunder name
    // hidden inside one is still rejected.
    let mut all_names: Vec<String> = Vec::new();
    let mut invalid: Vec<String> = Vec::new();
    collect_code_names(&code.code, &mut all_names, &mut invalid);
    if !invalid.is_empty() {
        return Err(EvalError::InvalidName(invalid.join(", ")));
    }

    // (3) build the scope and enforce the SECURITY POSTURE.
    //
    // Canonical xacro does `globals.update(__builtins__={})` so that ALL default
    // builtins become unreachable and only the curated allowlist (`list`, `dict`,
    // `len`, `str`, `int`, `float`, `bool`, `min`, `max`, `round`, `map`,
    // `sorted`, `range`, plus `math.*`/`xacro.*`/`python.*`) is exposed.
    //
    // In rustpython 0.4 this is NOT achievable by overwriting the globals'
    // `__builtins__` key: `VirtualMachine::run_code_obj` hardcodes the frame's
    // builtins to `self.builtins.dict()` and never consults the globals key. So
    // a `${open(...)}` would otherwise still reach the real `open`. Name
    // resolution is globals-first (frame `load_global_or_builtin` =
    // `globals.get_chain(builtins, ...)`), so we enforce the allowlist by
    // SHADOWING every actual VM builtin that is NOT on the allowlist with a
    // sentinel that raises on use, reproducing the security effect of
    // `__builtins__={}` (escape vectors `open`/`eval`/`exec`/`getattr`/
    // `__import__`/... are all neutralized) while keeping the allowlisted
    // computational builtins reachable. Combined with the dunder-name rejection
    // above, every documented sandbox-escape path is closed.
    let scope = vm.new_scope_with_builtins();
    shadow_disallowed_builtins(vm, &scope)?;

    // Inject the `math.*` symbols (bare `pi`/`radians`/`sin`/... AND a `math`
    // namespace object) exactly as canonical `create_global_symbols` does. These
    // go in BEFORE the registered functions / properties so a same-named property
    // still shadows them (a `<xacro:property name="pi" .../>` wins, matching
    // canonical where a property shadows a global). math is ubiquitous in real
    // robot xacros (`${pi/2}`, `${radians(45)}`), so this is what makes the
    // pipeline able to process actual SO-ARM / OpenArm files.
    {
        let globals = scope.globals.clone();
        crate::math_builtins::install_math(vm, |name, obj| {
            let _ = globals.set_item(name, obj, vm);
        });
    }

    // Inject the canonical `xacro` namespace object (`xacro.load_yaml`,
    // `xacro.arg`, `xacro.dotify`, `xacro.abs_filename`), the modern API form
    // (the bare `load_yaml(...)` injected below is the deprecated alias). The YAML
    // reader is taken from whichever namespace installed it; `xacro.arg` reads the
    // live arg snapshot. Injected before properties so a same-named property still
    // shadows it.
    {
        let reader = extra_functions
            .and_then(Namespace::yaml_reader)
            .or_else(|| namespace.yaml_reader());
        let xacro_mod = crate::xacro_builtins::build_xacro_module(vm, reader, args);
        scope
            .globals
            .set_item("xacro", xacro_mod, vm)
            .map_err(|e| EvalError::Bridge(format!("failed to inject 'xacro': {}", fmt_exc(vm, &e))))?;
    }

    // Inject the registered Rust functions FIRST (the load_yaml / radians seam),
    // from both the primary namespace and the optional extra-functions namespace,
    // so that a same-named property injected below shadows them (matching
    // canonical xacro, where a property shadows a same-named global).
    for reg in extra_functions
        .into_iter()
        .flat_map(Namespace::functions)
        .chain(namespace.functions())
    {
        let f = (reg.build)(vm);
        scope
            .globals
            .set_item(reg.name.as_str(), f, vm)
            .map_err(|e| {
                EvalError::Bridge(format!(
                    "failed to inject fn '{}': {}",
                    reg.name,
                    fmt_exc(vm, &e)
                ))
            })?;
    }

    // Inject the namespace properties LAST: each property name -> its value as a
    // Python object (so properties win over any same-named function above).
    for (name, value) in namespace.properties() {
        let obj = to_py(value, vm);
        scope
            .globals
            .set_item(name.as_str(), obj, vm)
            .map_err(|e| EvalError::Bridge(format!("failed to inject '{name}': {}", fmt_exc(vm, &e))))?;
    }

    // Also set `__builtins__` to an empty dict for good measure / parity with
    // the canonical source (harmless even though rustpython ignores it for name
    // resolution; the shadowing above is what actually enforces the sandbox).
    let empty = vm.ctx.new_dict();
    scope
        .globals
        .set_item("__builtins__", empty.into(), vm)
        .map_err(|e| EvalError::Bridge(format!("failed to blank __builtins__: {}", fmt_exc(vm, &e))))?;

    // (4) run + bridge the result back. If the run raises a `NameError` whose
    // `.name` is a name whose RESOLUTION we deferred (it errored during the
    // up-front snapshot and was omitted), the VM has actually LOADED that name
    // (i.e. it was in a LIVE branch, not short-circuited), so re-surface the
    // ORIGINAL resolution error (the exact circular/runtime message) instead of
    // the bare `NameError`. A name in a dead branch is never loaded, so this path
    // is not hit and the eval succeeds (matching canonical's live-globals lazy
    // resolution).
    let result = vm.run_code_obj(code, scope).map_err(|e| {
        if !deferred_errors.is_empty() && e.class().is(vm.ctx.exceptions.name_error) {
            if let Some(orig) = name_error_name(vm, &e).and_then(|n| deferred_errors.get(&n)) {
                return orig.clone();
            }
        }
        EvalError::Runtime(fmt_exc(vm, &e))
    })?;

    from_py(&result, vm)
}

/// Extract the offending name from a `NameError`'s `.name` attribute (rustpython,
/// like CPython 3.10+, sets it), so a live-branch load of a deferred name can be
/// mapped back to its original resolution error. Returns `None` if the attribute
/// is absent or not a `str`.
fn name_error_name(vm: &vm::VirtualMachine, e: &vm::builtins::PyBaseExceptionRef) -> Option<String> {
    let name_obj = e.as_object().get_attr("name", vm).ok()?;
    let s = name_obj.downcast_ref::<vm::builtins::PyStr>()?;
    Some(s.as_str().to_owned())
}

/// The bare-name builtins canonical xacro exposes globally (the directly-exposed
/// allowlist from `create_global_symbols`: `list/dict/map/len/str/float/int/bool/
/// min/max/round` + the `sorted/range` it also keeps global). The Python keyword
/// constants `True`/`False`/`None` are not dict entries and are always available.
/// Everything in the VM's builtins NOT here is shadowed to a raising sentinel.
const ALLOWED_BUILTINS: &[&str] = &[
    "list", "dict", "map", "len", "str", "float", "int", "bool", "min", "max", "round", "sorted",
    "range",
    // Constants that may appear as builtins-dict entries; harmless to keep.
    "True", "False", "None", "NotImplemented", "Ellipsis", "__debug__",
];

/// Shadow every VM builtin that is not on [`ALLOWED_BUILTINS`] with a sentinel
/// that raises when used, reproducing canonical xacro's `__builtins__={}` sandbox
/// (since rustpython's frame ignores the globals `__builtins__` key, see the note
/// at the call site). Name resolution is globals-first, so the shadow wins.
fn shadow_disallowed_builtins(
    vm: &vm::VirtualMachine,
    scope: &vm::scope::Scope,
) -> Result<(), EvalError> {
    use vm::builtins::PyStr;

    // Snapshot the builtin names so we don't mutate while iterating the source.
    let builtin_names: Vec<String> = vm
        .builtins
        .dict()
        .into_iter()
        .filter_map(|(k, _v)| k.downcast_ref::<PyStr>().map(|s| s.as_str().to_owned()))
        .collect();

    for name in builtin_names {
        if ALLOWED_BUILTINS.contains(&name.as_str()) {
            continue;
        }
        // A sentinel callable that raises a NameError mirroring an undefined name;
        // the disallowed builtin is reachable as a reference but unusable, so a
        // `${open(...)}`/`${eval(...)}`/`${getattr(...)}` escape attempt fails.
        let disabled_name = name.clone();
        let sentinel = vm.new_function(
            "<disabled-builtin>",
            move |_args: vm::function::FuncArgs, vm: &vm::VirtualMachine| -> vm::PyResult {
                Err(vm.new_name_error(
                    format!("name '{disabled_name}' is not defined"),
                    vm.ctx.new_str(disabled_name.as_str()),
                ))
            },
        );
        scope
            .globals
            .set_item(name.as_str(), sentinel.into(), vm)
            .map_err(|e| {
                EvalError::Bridge(format!("failed to shadow builtin '{name}': {}", fmt_exc(vm, &e)))
            })?;
    }
    Ok(())
}

/// Render a Python exception to a single-line message for an [`EvalError`].
fn fmt_exc(vm: &vm::VirtualMachine, e: &vm::builtins::PyBaseExceptionRef) -> String {
    let mut s = String::new();
    // write_exception cannot fail into a String sink in practice; ignore the
    // formatting Result and fall back to the type name if it produced nothing.
    let _ = vm.write_exception(&mut s, e);
    let s = s.trim().to_owned();
    if s.is_empty() {
        e.class().name().to_owned()
    } else {
        s
    }
}
