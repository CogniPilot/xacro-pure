//! `$(...)` substitution args: a faithful port of canonical xacro's
//! `xacro/substitution_args.py` (`resolve_args` + `_collect_args` + the
//! per-command handlers), plus the `$(cwd)` short-circuit from
//! `xacro/__init__.py::eval_extension`.
//!
//! ## The grammar (canonical `_collect_args` + `_resolve_args`)
//! A `$(...)` extension's inner text is split on whitespace; the first token is
//! the COMMAND, the rest are its ARGS:
//!   * `$(arg NAME)`:      look NAME up in the substitution-args table (the
//!     `mappings`/declared-`xacro:arg` values). Missing -> [`ArgError`].
//!   * `$(find PKG)`:      the share directory of PKG, via a pluggable
//!     [`PackageResolver`] (default: an env/AMENT-based lookup; tests + wasm
//!     inject a fake). Resolved in a SECOND pass so any earlier `$(...)` in the
//!     string is already expanded (the canonical two-pass ordering).
//!   * `$(env VAR)`:       environment variable VAR (error if unset).
//!   * `$(optenv VAR D)`:  environment variable VAR, or the space-joined rest
//!     `D...` as default if unset (default defaults to empty).
//!   * `$(dirname)`:       the absolute directory of the current file (from the
//!     [`SubstContext::filename`]).
//!   * `$(cwd)`:           the absolute current working / root directory.
//!   * `$(eval EXPR)`,     ONLY when the WHOLE input is `$(eval ...)`: safe-eval
//!     EXPR (with `arg`/`dirname`/`env`/`optenv`/`find`/`math.*` in scope) and
//!     `str()` the result.
//!
//! ## The `$(...)`-resolved-as-TEXT-before-`${...}` rule
//! This module resolves `$(...)` to a plain string. In [`crate::eval_text`] a
//! `${BODY}` expression first re-`eval_text`s BODY (so a `$(arg x)` inside the
//! `${...}` is an EXTENSION segment resolved to text FIRST), then python-evals
//! the resulting string, the OpenArm `load_yaml(cfg + "/" + $(arg x))` pattern.
//!
//! ## Canonical fidelity notes
//!   * The EXTENSION lexer body is `[^)]*` (stops at the first `)`), so a
//!     `$(...)` CANNOT contain a `)`; `$(eval int(arg('n')))` breaks in canonical
//!     too. We reproduce that by only special-casing `$(eval ...)` when the whole
//!     string is one, matching `arg_str.startswith('$(eval ')`.
//!   * `_eval` rejects any `__` for safety, mirroring the canonical guard.

use std::collections::HashMap;

use crate::error::EvalError;
use crate::namespace::Namespace;
use crate::value::XacroValue;

/// An error from resolving a `$(...)` substitution arg. Mirrors canonical
/// `SubstitutionException` / `ArgException`, kept as a dedicated type so a
/// missing `$(arg)` (the recoverable `ArgError`) is distinguishable from a hard
/// substitution failure; canonical xacro wraps the two differently in
/// `eval_extension`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubstError {
    /// A `$(arg NAME)` whose NAME is not in the substitution-args table.
    /// Canonical raises `ArgException(name)`.
    Arg(String),
    /// Any other substitution failure (unknown command, bad arity, unset env,
    /// `$(eval)` failure, ...). Carries the canonical-style message.
    Subst(String),
}

impl std::fmt::Display for SubstError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubstError::Arg(name) => write!(f, "Undefined substitution argument {name}"),
            SubstError::Subst(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for SubstError {}

impl From<SubstError> for EvalError {
    /// Bridge a substitution error into the eval-error channel so an EXTENSION
    /// segment failing inside `eval_text` surfaces like any other eval failure
    /// (canonical re-raises `SubstitutionException` as a `XacroException`).
    fn from(e: SubstError) -> EvalError {
        EvalError::Runtime(e.to_string())
    }
}

/// Resolves a ROS package name to its share directory, the seam behind
/// `$(find PKG)`. Canonical xacro calls `ament_index_python`'s
/// `get_package_share_directory`; that is an impure, environment-dependent,
/// non-wasm lookup. We make it INJECTABLE so:
///   * the default native resolver consults `AMENT_PREFIX_PATH` (the same data
///     ament uses); see [`AmentPackageResolver`];
///   * tests and the wasm build inject a deterministic fake / a closure.
pub trait PackageResolver {
    /// The share directory for `pkg`, or an error message if it cannot be found.
    fn share_directory(&self, pkg: &str) -> Result<String, String>;
}

/// A [`PackageResolver`] backed by a closure, so a test can write
/// `FnPackageResolver(|p| Ok(format!("/fake/{p}")))` inline.
pub struct FnPackageResolver<F>(pub F)
where
    F: Fn(&str) -> Result<String, String>;

impl<F> PackageResolver for FnPackageResolver<F>
where
    F: Fn(&str) -> Result<String, String>,
{
    fn share_directory(&self, pkg: &str) -> Result<String, String> {
        (self.0)(pkg)
    }
}

/// The default native resolver: walks `AMENT_PREFIX_PATH` (colon-separated
/// prefixes, as populated by a sourced ROS environment) looking for
/// `<prefix>/share/<pkg>`. This is the same data `ament_index_python` indexes,
/// so `$(find pkg)` resolves identically to canonical in a sourced workspace,
/// without depending on the Python `ament_index_python` package. On wasm (no
/// filesystem / env) a caller should inject a [`FnPackageResolver`] instead.
pub struct AmentPackageResolver;

impl PackageResolver for AmentPackageResolver {
    fn share_directory(&self, pkg: &str) -> Result<String, String> {
        let prefix_path = std::env::var("AMENT_PREFIX_PATH")
            .map_err(|_| "AMENT_PREFIX_PATH is not set".to_owned())?;
        for prefix in prefix_path.split(':').filter(|s| !s.is_empty()) {
            let candidate = format!("{prefix}/share/{pkg}");
            if std::path::Path::new(&candidate).is_dir() {
                return Ok(candidate);
            }
        }
        Err(format!("package '{pkg}' not found in AMENT_PREFIX_PATH"))
    }
}

/// The mutable resolution context threaded through a document expansion, the
/// analogue of canonical's `substitution_args_context`. Holds the substitution
/// `args` table (`$(arg ...)`), the current `filename` (for `$(dirname)`), the
/// `cwd`/root dir (for `$(cwd)`), and the injected [`PackageResolver`].
///
/// The resolver is OWNED (an `Rc<dyn PackageResolver>`) rather than borrowed so
/// the context can live inside a `PropertyTables` (which property resolution
/// threads through `eval_text`) without infecting that type with a lifetime;
/// the OpenArm `${load_yaml(... $(arg x))}` pattern resolves `$(...)` during
/// lazy property resolution, so the substitution state must be reachable there.
/// It is an `Rc` (not a `Box`) so the `find(...)` callable inside `$(eval ...)`,
/// which the embedded interpreter builds and may hold past this call, can share
/// the same resolver by cloning the handle.
pub struct SubstContext {
    /// The substitution-args table: `mappings` plus any declared `<xacro:arg>`
    /// defaults. Canonical stores this under `context['arg']`.
    pub args: HashMap<String, String>,
    /// The path of the file being processed (for `$(dirname)`), or `None`.
    pub filename: Option<String>,
    /// The root / current-working directory (for `$(cwd)`).
    pub cwd: String,
    /// The injected package resolver for `$(find ...)`.
    pub resolver: std::rc::Rc<dyn PackageResolver>,
}

impl SubstContext {
    /// A context with the given resolver, an empty arg table, no filename, and
    /// `cwd = "."` (matching canonical's default `root_dir`).
    pub fn new(resolver: Box<dyn PackageResolver>) -> Self {
        SubstContext {
            args: HashMap::new(),
            filename: None,
            cwd: ".".to_owned(),
            resolver: std::rc::Rc::from(resolver),
        }
    }
}

/// Port of canonical `eval_extension(s)` for the already-extracted INNER text
/// of one `$(...)` (i.e. `arg_str` without the surrounding `$(` `)`). The caller
/// in `eval_text` passes the inner body; we re-wrap it as `$(body)` to reuse the
/// `resolve_args` machinery, exactly as canonical's `handle_extension` does
/// (`eval_extension("$(%s)" % ...)`).
pub fn eval_extension(inner: &str, ctx: &mut SubstContext) -> Result<String, SubstError> {
    let arg_str = format!("$({inner})");
    resolve_args(&arg_str, ctx)
}

/// Port of canonical `resolve_args(arg_str, context)`.
///
/// 1. If the WHOLE string is `$(eval ...)`, safe-eval the inner expression.
/// 2. Otherwise two passes: pass 1 resolves `env`/`optenv`/`dirname`/`arg`/
///    `find`; pass 2 resolves `find` again (so a `find` whose path was produced
///    by an earlier substitution is expanded last). We fold both `find` passes
///    by running all commands in pass 1 and `find` again in pass 2, identical to
///    canonical.
pub fn resolve_args(arg_str: &str, ctx: &mut SubstContext) -> Result<String, SubstError> {
    if arg_str.is_empty() {
        return Ok(arg_str.to_owned());
    }
    // `$(cwd)` short-circuit (canonical `eval_extension` checks this first, before
    // ever reaching `resolve_args`'s command table; `cwd` is NOT a valid
    // `_resolve_args` command). Returns the absolute root/cwd directory.
    if arg_str == "$(cwd)" {
        let abs = std::path::Path::new(&ctx.cwd);
        let resolved = if abs.is_absolute() {
            ctx.cwd.clone()
        } else {
            std::env::current_dir()
                .map(|d| d.join(&ctx.cwd).to_string_lossy().into_owned())
                .unwrap_or_else(|_| ctx.cwd.clone())
        };
        return Ok(resolved);
    }
    // Whole-string `$(eval ...)` special case.
    if let Some(rest) = arg_str.strip_prefix("$(eval ") {
        if let Some(expr) = rest.strip_suffix(')') {
            return eval_substitution(expr, ctx);
        }
    }

    // Pass 1: all five commands.
    let resolved = resolve_pass(arg_str, ctx, &["find", "env", "optenv", "dirname", "arg"])?;
    // Pass 2: find only.
    let resolved = resolve_pass(&resolved, ctx, &["find"])?;
    Ok(resolved)
}

/// One `_resolve_args` pass: for every `$(...)` collected from `arg_str`, if its
/// command is enabled in `enabled`, apply it (replacing every `$(<inner>)` with
/// its value). An unknown command is always an error (matching canonical's
/// `valid` check, which fires regardless of which pass).
fn resolve_pass(
    arg_str: &str,
    ctx: &mut SubstContext,
    enabled: &[&str],
) -> Result<String, SubstError> {
    const VALID: &[&str] = &["find", "env", "optenv", "dirname", "arg"];
    let mut resolved = arg_str.to_owned();
    for a in collect_args(arg_str)? {
        let splits: Vec<&str> = a.split_whitespace().collect();
        let command = match splits.first() {
            Some(c) => *c,
            // An empty `$()` has no command token; canonical would index
            // `splits[0]` and IndexError. Treat as an unknown-command error.
            None => {
                return Err(SubstError::Subst(format!(
                    "Unknown substitution command [{a}]. Valid commands are {VALID:?}"
                )))
            }
        };
        if !VALID.contains(&command) {
            return Err(SubstError::Subst(format!(
                "Unknown substitution command [{a}]. Valid commands are {VALID:?}"
            )));
        }
        if !enabled.contains(&command) {
            continue;
        }
        let args = &splits[1..];
        let value = apply_command(command, &a, args, ctx)?;
        resolved = resolved.replace(&format!("$({a})"), &value);
    }
    Ok(resolved)
}

/// Apply one resolved command, returning the replacement string.
fn apply_command(
    command: &str,
    a: &str,
    args: &[&str],
    ctx: &mut SubstContext,
) -> Result<String, SubstError> {
    match command {
        "arg" => {
            if args.is_empty() {
                return Err(SubstError::Subst(format!(
                    "$(arg var) must specify a variable name [{a}]"
                )));
            }
            if args.len() > 1 {
                return Err(SubstError::Subst(format!(
                    "$(arg var) may only specify one arg [{a}]"
                )));
            }
            ctx.args
                .get(args[0])
                .cloned()
                .ok_or_else(|| SubstError::Arg(args[0].to_owned()))
        }
        "env" => {
            if args.len() != 1 {
                return Err(SubstError::Subst(format!(
                    "$(env var) command only accepts one argument [{a}]"
                )));
            }
            std::env::var(args[0]).map_err(|_| {
                SubstError::Subst(format!("environment variable '{}' is not set", args[0]))
            })
        }
        "optenv" => {
            if args.is_empty() {
                return Err(SubstError::Subst(format!(
                    "$(optenv var) must specify an environment variable [{a}]"
                )));
            }
            match std::env::var(args[0]) {
                Ok(v) => Ok(v),
                // default = ' '.join(args[1:])
                Err(_) => Ok(args[1..].join(" ")),
            }
        }
        "dirname" => {
            let filename = ctx.filename.as_deref().ok_or_else(|| {
                SubstError::Subst(
                    "Cannot substitute $(dirname), no file/directory information available."
                        .to_owned(),
                )
            })?;
            Ok(abs_dirname(filename))
        }
        "find" => {
            if args.len() != 1 {
                return Err(SubstError::Subst(format!(
                    "$(find pkg) accepts exactly one argument [{a}]"
                )));
            }
            ctx.resolver
                .share_directory(args[0])
                .map_err(SubstError::Subst)
        }
        // `cwd` is handled by resolve_args' whole-string check in canonical
        // (eval_extension), never reaching the command table; guard anyway.
        other => Err(SubstError::Subst(format!(
            "Unknown substitution command [{other}]"
        ))),
    }
}

/// The absolute directory of `filename` (`os.path.abspath(os.path.dirname(f))`).
/// Pure path arithmetic so it stays wasm-safe (no filesystem access): take the
/// parent, and if it is relative, join it onto the process cwd.
fn abs_dirname(filename: &str) -> String {
    let path = std::path::Path::new(filename);
    let dir = path.parent().unwrap_or_else(|| std::path::Path::new(""));
    if dir.is_absolute() {
        return dir.to_string_lossy().into_owned();
    }
    // Relative: join onto the current dir (best-effort; "." if unavailable).
    let base = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    base.join(dir).to_string_lossy().into_owned()
}

/// Port of canonical `_eval(s, context)` (`substitution_args.py:264`):
/// ```python
/// functions = {'arg': _eval_arg_context, 'dirname': _eval_dirname_context}
/// functions.update(_eval_dict)           # math.* + True/False/true/false + env/optenv/find
/// if s.find('__') >= 0: raise SubstitutionException(...)
/// return str(eval(s, {}, _DictWrapper(context['arg'], functions)))
/// ```
/// `_DictWrapper.__getitem__` returns `functions[key]` if present, ELSE
/// `convert_value(args[key], 'auto')`, i.e. a BARE arg name resolves directly to
/// its auto-typed value. So `$(eval count + 10)` (with arg `count=3`) is `13`,
/// `$(eval pi)` resolves the math constant, `$(eval flag)` (arg `flag=true`)
/// resolves to the bool `True`. The previous port evaluated against an EMPTY
/// namespace with only `arg()`, so every bare name / math symbol raised.
///
/// We rebuild that namespace for `safe_eval`: the args table is injected as bare
/// auto-typed PROPERTIES (so `$(eval count + 10)` resolves), and the `_eval_dict`
/// callables (`arg`/`dirname`/`env`/`optenv`/`find`) as functions. Because
/// `_DictWrapper` checks functions BEFORE args, a function must WIN a name
/// collision with an arg: we drop any arg whose name matches a function name
/// before injecting it as a property, so the function binding stands.
/// `math.*` is supplied by `safe_eval` itself (it injects the math namespace).
/// `True`/`False` are Python keyword constants; we add `true`/`false` as bare
/// aliases.
fn eval_substitution(expr: &str, ctx: &mut SubstContext) -> Result<String, SubstError> {
    if expr.contains("__") {
        return Err(SubstError::Subst(
            "$(eval ...) may not contain double underscore expressions".to_owned(),
        ));
    }

    // (a) functions: the canonical `_eval_dict` callables. `arg`/`dirname` cover
    // the direct property/dirname access; `env`/`optenv`/`find` reuse the same
    // resolution the `$(env)`/`$(optenv)`/`$(find)` commands use. `math.*` is
    // injected by safe_eval directly.
    let args_snapshot = ctx.args.clone();
    let filename = ctx.filename.clone();
    let mut functions = Namespace::new();
    functions.register_raw("arg", {
        let table = args_snapshot.clone();
        move |vm| {
            let table = table.clone();
            vm.new_function(
                "arg",
                move |name: String, vm: &rustpython_vm::VirtualMachine| -> rustpython_vm::PyResult {
                    match table.get(&name) {
                        Some(raw) => Ok(crate::bridge::to_py(&convert_value_auto(raw), vm)),
                        None => Err(vm.new_key_error(vm.ctx.new_str(name).into())),
                    }
                },
            )
            .into()
        }
    });
    functions.register_raw("dirname", {
        let filename = filename.clone();
        move |vm| {
            let filename = filename.clone();
            vm.new_function(
                "dirname",
                move |vm: &rustpython_vm::VirtualMachine| -> rustpython_vm::PyResult {
                    match &filename {
                        Some(f) => Ok(vm.ctx.new_str(abs_dirname(f)).into()),
                        None => Err(vm.new_runtime_error(
                            "Cannot substitute $(dirname), no file/directory information available."
                                .to_owned(),
                        )),
                    }
                },
            )
            .into()
        }
    });
    // `env(name)`: the environment variable, erroring if unset (same as the
    // `$(env)` command handler).
    functions.register_raw("env", move |vm| {
        vm.new_function(
            "env",
            move |name: String, vm: &rustpython_vm::VirtualMachine| -> rustpython_vm::PyResult {
                std::env::var(&name)
                    .map(|v| vm.ctx.new_str(v).into())
                    .map_err(|_| {
                        vm.new_runtime_error(format!("environment variable '{name}' is not set"))
                    })
            },
        )
        .into()
    });
    // `optenv(name, *default)`: the environment variable, or the space-joined
    // default arguments if unset (same as the `$(optenv)` command handler).
    functions.register_raw("optenv", move |vm| {
        vm.new_function(
            "optenv",
            move |name: String,
                  default: rustpython_vm::function::PosArgs<String>,
                  vm: &rustpython_vm::VirtualMachine|
                  -> rustpython_vm::PyResult {
                match std::env::var(&name) {
                    Ok(v) => Ok(vm.ctx.new_str(v).into()),
                    Err(_) => Ok(vm.ctx.new_str(default.into_vec().join(" ")).into()),
                }
            },
        )
        .into()
    });
    // `find(pkg)`: the package share directory via the injected resolver (same as
    // the `$(find)` command handler). The `Rc` resolver is cloned into the
    // callable so it can outlive this call inside the interpreter.
    functions.register_raw("find", {
        let resolver = ctx.resolver.clone();
        move |vm| {
            let resolver = resolver.clone();
            vm.new_function(
                "find",
                move |pkg: String, vm: &rustpython_vm::VirtualMachine| -> rustpython_vm::PyResult {
                    resolver
                        .share_directory(&pkg)
                        .map(|p| vm.ctx.new_str(p).into())
                        .map_err(|e| vm.new_runtime_error(e))
                },
            )
            .into()
        }
    });

    // (b) properties: True/False/true/false as bare names, then EVERY arg as a
    // bare auto-typed name (the _DictWrapper args fallback). Injected as
    // properties so safe_eval binds them in the eval scope.
    let mut props = Namespace::new();
    props.set("True", XacroValue::Bool(true));
    props.set("False", XacroValue::Bool(false));
    props.set("true", XacroValue::Bool(true));
    props.set("false", XacroValue::Bool(false));
    for (name, raw) in &args_snapshot {
        // A function of the same name must WIN (canonical checks functions first),
        // so an arg colliding with a function name is NOT injected as a property.
        if EVAL_FUNCTION_NAMES.contains(&name.as_str()) {
            continue;
        }
        props.set(name.clone(), convert_value_auto(raw));
    }

    let value = crate::eval::safe_eval_with(expr, &props, &functions)
        .map_err(|e| SubstError::Subst(format!("$(eval {expr}) failed: {e}")))?;
    // `_eval` returns `str(eval(...))`.
    Ok(value.to_python_str())
}

/// The `_eval_dict` callable names exposed inside `$(eval ...)`. An arg colliding
/// with one of these is dropped from the property injection so the function wins
/// the binding (canonical `_DictWrapper` checks functions before args).
const EVAL_FUNCTION_NAMES: &[&str] = &["arg", "dirname", "env", "optenv", "find"];

/// Canonical `convert_value(value, 'auto')`: numeric if it parses (float if it
/// has a `.`, else int), bool for `true`/`false` (case-insensitive), else the
/// raw string. Used to type a `$(arg)` value inside `$(eval)`.
fn convert_value_auto(value: &str) -> XacroValue {
    if value.contains('.') {
        if let Ok(f) = value.parse::<f64>() {
            return XacroValue::Float(f);
        }
    } else if let Ok(i) = value.parse::<num_bigint::BigInt>() {
        return XacroValue::Int(i);
    }
    match value.to_ascii_lowercase().as_str() {
        "true" => return XacroValue::Bool(true),
        "false" => return XacroValue::Bool(false),
        _ => {}
    }
    XacroValue::Str(value.to_owned())
}

/// State-machine port of canonical `_collect_args(arg_str)`: returns the inner
/// contents of each top-level `$(...)` (the text between `$(` and the matching (actually
/// the FIRST) `)`). `$(...)` cannot contain `)` (the `[^)]*` lexer
/// rule), and a `$` inside an open `$(...)` is an error, both matching canonical.
fn collect_args(arg_str: &str) -> Result<Vec<String>, SubstError> {
    // States mirror canonical's _OUT/_DOLLAR/_LP/_IN.
    #[derive(PartialEq)]
    enum St {
        Out,
        Dollar,
        Lp,
        In,
    }
    let mut buff = String::new();
    let mut args: Vec<String> = Vec::new();
    let mut state = St::Out;
    for c in arg_str.chars() {
        match c {
            '$' => match state {
                St::Out => state = St::Dollar,
                St::Dollar => {}
                _ => {
                    return Err(SubstError::Subst(format!(
                        "Dollar signs \"$\" cannot be inside of substitution args [{arg_str}]"
                    )))
                }
            },
            '(' => match state {
                St::Dollar => state = St::Lp,
                St::Out => {}
                _ => {
                    return Err(SubstError::Subst(format!(
                        "Invalid left parenthesis \"(\" in substitution args [{arg_str}]"
                    )))
                }
            },
            ')' => {
                if state == St::In {
                    args.push(std::mem::take(&mut buff));
                }
                state = St::Out;
            }
            _ => {
                // canonical: in _DOLLAR a non-'(' char drops back to _OUT; in _LP
                // the first char enters _IN.
                if state == St::Dollar {
                    state = St::Out;
                } else if state == St::Lp {
                    state = St::In;
                }
            }
        }
        if state == St::In {
            buff.push(c);
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A context with a fake resolver mapping `pkg -> /share/pkg`.
    fn fake_ctx() -> SubstContext {
        SubstContext::new(Box::new(FnPackageResolver(|p: &str| {
            Ok(format!("/share/{p}"))
        })))
    }

    #[test]
    fn collect_args_basic() {
        assert_eq!(
            collect_args("$(find pkg)/x/$(arg y)").unwrap(),
            vec!["find pkg".to_owned(), "arg y".to_owned()]
        );
    }

    #[test]
    fn collect_args_stops_at_first_paren() {
        // EXTENSION body is [^)]*; the first `)` closes it, so the trailing
        // text after a `$(...)` is NOT part of the collected arg.
        assert_eq!(
            collect_args("$(find pkg)/sub)dir").unwrap(),
            vec!["find pkg".to_owned()]
        );
    }

    #[test]
    fn collect_args_rejects_inner_open_paren() {
        // canonical `_collect_args` raises on a `(` while already inside a
        // `$(...)` (state `_IN`): a nested `(` is invalid. The whole-string
        // `$(eval ...)` special-case in `resolve_args` avoids this for eval; a
        // bare `collect_args` over a nested-paren arg errors, matching canonical.
        assert!(matches!(
            collect_args("$(eval f(x))"),
            Err(SubstError::Subst(_))
        ));
    }

    #[test]
    fn arg_resolves_from_table() {
        let mut ctx = fake_ctx();
        ctx.args.insert("robot".to_owned(), "so100".to_owned());
        assert_eq!(resolve_args("a_$(arg robot)", &mut ctx).unwrap(), "a_so100");
    }

    #[test]
    fn arg_missing_is_arg_error() {
        let mut ctx = fake_ctx();
        assert_eq!(
            resolve_args("$(arg nope)", &mut ctx),
            Err(SubstError::Arg("nope".to_owned()))
        );
    }

    #[test]
    fn find_uses_injected_resolver() {
        let mut ctx = fake_ctx();
        assert_eq!(
            resolve_args("$(find my_pkg)/meshes", &mut ctx).unwrap(),
            "/share/my_pkg/meshes"
        );
    }

    #[test]
    fn optenv_default_when_unset() {
        let mut ctx = fake_ctx();
        // A name that is essentially never set.
        assert_eq!(
            resolve_args("v_$(optenv XACRO_PURE_NOPE_VAR fb word)", &mut ctx).unwrap(),
            "v_fb word"
        );
    }

    #[test]
    fn eval_whole_string_arithmetic() {
        let mut ctx = fake_ctx();
        assert_eq!(resolve_args("$(eval 2 * 5 + 1)", &mut ctx).unwrap(), "11");
    }

    #[test]
    fn eval_reads_arg() {
        let mut ctx = fake_ctx();
        ctx.args.insert("n".to_owned(), "3".to_owned());
        assert_eq!(
            resolve_args("$(eval arg('n') + 10)", &mut ctx).unwrap(),
            "13"
        );
    }

    #[test]
    fn eval_env_reads_environment() {
        let mut ctx = fake_ctx();
        std::env::set_var("XACRO_PURE_X3_ENV", "hello");
        assert_eq!(
            resolve_args("$(eval env('XACRO_PURE_X3_ENV'))", &mut ctx).unwrap(),
            "hello"
        );
        std::env::remove_var("XACRO_PURE_X3_ENV");
    }

    #[test]
    fn eval_optenv_default_and_set() {
        let mut ctx = fake_ctx();
        // Unset variable -> the default argument.
        assert_eq!(
            resolve_args(
                "$(eval optenv('XACRO_PURE_X3_UNSET_VAR', 'fallback'))",
                &mut ctx
            )
            .unwrap(),
            "fallback"
        );
        // Set variable -> its value.
        std::env::set_var("XACRO_PURE_X3_OPT", "here");
        assert_eq!(
            resolve_args("$(eval optenv('XACRO_PURE_X3_OPT', 'fallback'))", &mut ctx).unwrap(),
            "here"
        );
        std::env::remove_var("XACRO_PURE_X3_OPT");
    }

    #[test]
    fn eval_find_uses_resolver() {
        // `find(...)` inside $(eval) resolves through the injected resolver, the
        // same one $(find ...) uses.
        let mut ctx = fake_ctx();
        assert_eq!(
            resolve_args("$(eval find('my_pkg'))", &mut ctx).unwrap(),
            "/share/my_pkg"
        );
    }

    #[test]
    fn eval_function_wins_arg_name_collision() {
        // An arg named the same as a function must NOT shadow the function inside
        // $(eval): canonical's _DictWrapper checks functions first. An arg `find`
        // collides with the find() function; `find(...)` must call the function.
        let mut ctx = fake_ctx();
        ctx.args.insert("find".to_owned(), "SOME_ARG_VALUE".to_owned());
        assert_eq!(
            resolve_args("$(eval find('collision_pkg'))", &mut ctx).unwrap(),
            "/share/collision_pkg"
        );
    }

    #[test]
    fn eval_rejects_dunder() {
        let mut ctx = fake_ctx();
        assert!(matches!(
            resolve_args("$(eval __import__('os'))", &mut ctx),
            Err(SubstError::Subst(_))
        ));
    }

    #[test]
    fn unknown_command_errors() {
        let mut ctx = fake_ctx();
        assert!(matches!(
            resolve_args("$(bogus x)", &mut ctx),
            Err(SubstError::Subst(_))
        ));
    }
}
