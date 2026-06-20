//! THE PROPERTY MODEL.
//!
//! A faithful port of canonical xacro's `Table` / `NameSpace` property system
//! (`xacro/__init__.py`: `Table`, `NameSpace`, `_resolve_`, `_setitem`,
//! `grab_property`). This is the piece every other Rust xacro crate gets wrong:
//!
//!   * `xacro-rs` stores raw RHS strings and re-evaluates eagerly, hitting a
//!     self-reference / circular wall.
//!   * `xurdf` evaluates uniformly eager, so late vs early binding is wrong.
//!
//! Canonical xacro instead uses a **per-scope `Table`** (a `dict` subclass) with
//! a **per-property lazy/eager flag** and **lazy resolution-on-first-access**:
//!
//!   * DEFINITION (`grab_property` -> `_setitem`): the raw RHS is `_eval_literal`
//!     typed at def time; `lazy_eval` defaults true; if eager (`lazy_eval=false`,
//!     or `scope=global`/`scope=parent` which FORCE eager) the value is
//!     `eval_text`-evaluated immediately against the current table. If it is
//!     still an unevaluated string it is added to the `unevaluated` set (a
//!     deferred lazy property); otherwise it is final.
//!   * ACCESS (`_resolve_`): on the first read of an `unevaluated` key, its
//!     stored string is `eval_text`-evaluated, the result is CACHED back into the
//!     table, and the key is removed from `unevaluated`. Re-entrancy is tracked
//!     in a `recursive` list; re-entering the SAME key raises the EXACT canonical
//!     error `circular variable definition: a -> b -> a\nConsider disabling lazy
//!     evaluation via lazy_eval="false"`.
//!
//! ## Behaviors this reproduces (cross-checked vs canonical `xacro`)
//!   * (A) lazy = LATE binding: `derived=${base*2}`; redefine `base` later;
//!     reading `derived` uses the NEW `base`.
//!   * (B) eager = EARLY binding frozen at def.
//!   * (C) eager SELF-REFERENCE: `prop=${10}` then `prop=${prop+5}` eager -> 15
//!     (the RHS reads the PRIOR binding at def time); the dict form
//!     `d=${dict(x=1.0)}` then `d=${dict(x=d['x']+100.0)}` -> `{x: 101.0}`.
//!   * SCOPES: `local` / `parent` / `global`. `global` and `parent` FORCE eager
//!     and RETARGET the destination table (`global` -> the top user table;
//!     `parent` -> the caller's scope, skipping intervening `NameSpace`s when in
//!     macro scope).
//!   * `default=` (only-if-absent), `remove=` (delete up to root),
//!     `is_valid_name` (reject python keywords + `__` prefix),
//!     redefine-global-symbol warning.
//!
//! ## Rust modeling note
//! Canonical Python mutates `Table`s in place during `_resolve_` (lazy caching)
//! and rewires parent links for scope traversal, all over a graph of `dict`
//! subclasses. We mirror that with an **arena of [`Scope`]s** (a `Vec` owned by
//! the [`PropertyTables`]) addressed by [`ScopeId`]. A scope holds its own
//! property map, the `unevaluated` set, the `recursive` re-entrancy guard, and a
//! `parent: Option<ScopeId>` link. Lazy resolution + caching needs to *read while
//! mutating* the same arena, which the `ScopeId` indirection makes safe (we split
//! borrows by id rather than aliasing `&mut`). This is the direct structural
//! analogue of Python's `self`/`self.parent`/`self.root` pointers.

use std::collections::HashSet;

use num_bigint::BigInt;

use crate::error::EvalError;
use crate::eval_text::eval_text;
use crate::literal::eval_literal;
use crate::namespace::Namespace;
use crate::substitution_args::SubstContext;
use crate::value::XacroValue;

/// An index into the [`PropertyTables`] scope arena. The structural analogue of
/// a Python `Table` object reference (`self` / `self.parent` / `self.root`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(usize);

/// A single property binding's stored form: either a fully-resolved typed value,
/// or a still-unevaluated RHS string awaiting lazy resolution.
///
/// In canonical Python both live in the same `dict` slot: a resolved value is a
/// typed Python object, an unevaluated one is the raw `str` whose key is also in
/// `self.unevaluated`. We make the distinction explicit so resolve-on-access has
/// no ambiguity about which slots still need evaluating.
#[derive(Debug, Clone, PartialEq)]
enum Binding {
    /// A fully-resolved, typed value. No further evaluation on access.
    Resolved(XacroValue),
    /// A deferred lazy property: the raw RHS string to be `eval_text`-evaluated
    /// on first access, then cached as [`Binding::Resolved`]. The key is also
    /// present in the owning scope's `unevaluated` set.
    Unevaluated(String),
    /// A `NameSpace` binding: `symbols[ns] = NameSpace(parent=symbols)` from a
    /// `<xacro:include ns="..">`. Points to the namespace's own property
    /// [`ScopeId`]. On access (`${ns.prop}`) it materializes to a
    /// [`XacroValue::Namespace`] of the namespace's reachable properties (own +
    /// inherited), so dotted access resolves like canonical's `NameSpace`. Stored
    /// as a binding (not a resolved value) because the namespace scope is populated
    /// AFTER the binding is created (the include's body is processed later), so it
    /// must be materialized lazily at access time, not snapshotted at push time.
    Namespace(ScopeId),
}

/// A single lexical scope: the property bindings plus the lazy-resolution
/// bookkeeping, with a parent link for scope traversal.
///
/// Mirrors one canonical `Table`/`NameSpace` instance: the `dict` payload
/// (`bindings`), the `unevaluated` set, the `recursive` re-entrancy guard, and
/// the `parent` pointer. `is_namespace` distinguishes a `NameSpace` (a macro's
/// argument scope / a declared namespace) from a plain macro-body `Table`, which
/// matters for `scope=parent` retargeting (see [`PropertyTables::set_property`]).
#[derive(Debug)]
struct Scope {
    /// `name -> Binding`. Insertion order is irrelevant for lookup; a plain map
    /// suffices (unlike [`XacroValue::Dict`], property scopes are not iterated
    /// in definition order anywhere that order is observable).
    bindings: indexmap::IndexMap<String, Binding>,
    /// Keys whose `Binding` is still [`Binding::Unevaluated`]: the set of
    /// properties that need lazy resolution on first access. Mirrors
    /// `Table.unevaluated`.
    unevaluated: HashSet<String>,
    /// Re-entrancy guard: keys currently being resolved, in order, so a circular
    /// definition produces the exact `a -> b -> a` chain. Mirrors
    /// `Table.recursive`.
    recursive: Vec<String>,
    /// Parent scope for lookup fallback / traversal, or `None` for the root.
    parent: Option<ScopeId>,
    /// Whether this scope is a `NameSpace` (vs a plain macro-body `Table`).
    /// `scope="parent"` from inside a macro skips intervening `NameSpace`s.
    is_namespace: bool,
}

impl Scope {
    fn new(parent: Option<ScopeId>, is_namespace: bool) -> Self {
        Scope {
            bindings: indexmap::IndexMap::new(),
            unevaluated: HashSet::new(),
            recursive: Vec::new(),
            parent,
            is_namespace,
        }
    }
}

/// The arena of scopes that together form a xacro property environment, plus the
/// `root` scope id (the equivalent of `_global_symbols`: the table whose contents
/// trigger the "redefining global symbol" warning and which `del` cannot cross).
///
/// The top user scope (`symbols` in canonical xacro) is created as a child of the
/// root, exactly as `symbols = Table(_global_symbols)`.
pub struct PropertyTables {
    scopes: Vec<Scope>,
    root: ScopeId,
    /// The top user scope (child of root). Where top-level `<xacro:property>`s
    /// land and where `scope="global"` retargets to.
    top: ScopeId,
    /// Non-fatal diagnostics collected during definition (redefinition,
    /// no-parent-at-global-scope, default-on-non-local-scope). Canonical xacro
    /// emits these via `warning(...)`; we collect rather than print so a caller
    /// (and the tests) can assert on them without capturing stderr.
    warnings: Vec<String>,
    /// Registered Rust functions (the `xacro.load_yaml` / `radians` seam) that
    /// every `eval_text`/`safe_eval` over these tables should see. The
    /// [`Namespace`] owns these; we thread a borrow at eval time.
    functions: Namespace,
    /// The substitution-args context (canonical's `substitution_args_context`)
    /// for `$(...)` EXTENSION segments (`$(arg)`/`$(find)`/`$(eval)`/...) that
    /// may appear inside a property value (the OpenArm
    /// `${load_yaml(cfg + '/' + $(arg x))}` pattern resolves `$(...)` during lazy
    /// property resolution). `None` when no document context has been attached
    /// (a pure property-model use); then an EXTENSION inside a value is an error.
    subst: Option<SubstContext>,
}

impl Default for PropertyTables {
    fn default() -> Self {
        Self::new()
    }
}

/// The destination scope for a property definition plus the (possibly forced)
/// lazy flag, the result of resolving the `scope=` attribute. Mirrors the
/// `target_table` / `lazy_eval` pair computed in canonical `grab_property`.
struct SetTarget {
    table: ScopeId,
    lazy: bool,
}

/// The raw `<xacro:property>` attributes, bundled so the definition entry point
/// takes one descriptor rather than a long positional argument list. Each field
/// is the RAW attribute text (pre-`eval_text`), `None` meaning the attribute is
/// absent, exactly the shape `grab_property`'s `check_attrs` produces for
/// `['name'], ['value', 'default', 'remove', 'scope', 'lazy_eval']`.
#[derive(Debug, Default, Clone, Copy)]
pub struct PropertyDef<'a> {
    /// `name=` (required). May itself be an expression (`eval_text`-evaluated).
    pub name: &'a str,
    /// `value=`: the property RHS (mutually exclusive with `default`/`remove`).
    pub value: Option<&'a str>,
    /// `default=`: only-if-absent RHS (local scope only).
    pub default: Option<&'a str>,
    /// `remove=`: when truthy, delete the property up to root.
    pub remove: Option<&'a str>,
    /// `scope=`: `local` (default) / `parent` / `global`.
    pub scope: Option<&'a str>,
    /// `lazy_eval=`: `true` (default) / `false`; `parent`/`global` force false.
    pub lazy_eval: Option<&'a str>,
}

impl<'a> PropertyDef<'a> {
    /// A `name=N value=V` property (lazy default), the overwhelmingly common
    /// shape. Convenience constructor.
    pub fn value(name: &'a str, value: &'a str) -> Self {
        PropertyDef {
            name,
            value: Some(value),
            ..PropertyDef::default()
        }
    }
}

impl PropertyTables {
    /// Build an environment with a fresh root (`_global_symbols`) and a top user
    /// scope (`symbols`) as its child, mirroring `symbols = Table(_global_symbols)`.
    pub fn new() -> Self {
        let root = Scope::new(None, false);
        let mut scopes = vec![root];
        let root_id = ScopeId(0);
        let top = Scope::new(Some(root_id), false);
        scopes.push(top);
        let top_id = ScopeId(1);
        PropertyTables {
            scopes,
            root: root_id,
            top: top_id,
            warnings: Vec::new(),
            functions: Namespace::new(),
            subst: None,
        }
    }

    /// Attach a substitution-args [`SubstContext`] so `$(...)` EXTENSION segments
    /// inside property values resolve. The document pipeline installs this before
    /// expansion; a pure property-model caller may leave it absent.
    pub fn set_subst_context(&mut self, ctx: SubstContext) -> &mut Self {
        self.subst = Some(ctx);
        self
    }

    /// Borrow the attached substitution context, if any (so the pipeline can
    /// read/mutate the `arg` table, e.g. for `<xacro:arg>` declarations).
    pub fn subst_context_mut(&mut self) -> Option<&mut SubstContext> {
        self.subst.as_mut()
    }

    /// The top user scope id: where top-level properties live and where new
    /// macro / namespace child scopes attach.
    pub fn top_scope(&self) -> ScopeId {
        self.top
    }

    /// Collected non-fatal warnings, in emission order.
    ///
    /// NOTE: the document pipeline (`process_document`/`process_document_with`)
    /// currently collects these but does NOT surface them; it returns only the
    /// expanded XML, so they are dropped on the floor (canonical xacro prints
    /// them to stderr). Only a caller driving the property model directly can
    /// read them today.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Register a Rust function visible to every expression evaluated against
    /// these tables (the `xacro.load_yaml` / `radians` seam).
    pub fn register_fn_i64(
        &mut self,
        name: &'static str,
        f: impl Fn(i64) -> i64 + Copy + 'static,
    ) -> &mut Self {
        self.functions.register_fn_i64(name, f);
        self
    }

    /// Install a whole [`Namespace`] of registered functions (e.g. the document
    /// pipeline's `load_yaml` seam). Replaces the current function set.
    pub fn set_functions(&mut self, functions: Namespace) -> &mut Self {
        self.functions = functions;
        self
    }

    /// Push a new child scope under `parent`, returning its id. `is_namespace`
    /// marks it as a `NameSpace` (a macro argument scope / declared namespace)
    /// vs a plain macro-body `Table`.
    pub fn push_scope(&mut self, parent: ScopeId, is_namespace: bool) -> ScopeId {
        let id = ScopeId(self.scopes.len());
        self.scopes.push(Scope::new(Some(parent), is_namespace));
        id
    }

    /// Create a namespaced child `NameSpace` scope `ns` under `parent` and bind it
    /// as a property in `parent` under the name `ns` (so `${ns.prop}` dotted access
    /// resolves). Mirrors `symbols[ns] = NameSpace(parent=symbols)` in
    /// `process_include`. Returns the new scope id (the namespace's own table).
    pub fn push_namespace(&mut self, parent: ScopeId, ns: &str) -> ScopeId {
        // The namespace's symbol table is a NameSpace child of the parent so its
        // contents fall back to the parent on lookup, matching canonical's
        // `NameSpace(parent=symbols)`.
        let id = self.push_scope(parent, true);
        // Bind it under `ns` in the parent as a Namespace binding, so a later
        // `${ns.prop}` reads it and materializes the namespace's properties for
        // dotted access, canonical's `symbols[ns] = NameSpace(...)`.
        if !ns.is_empty() {
            self.scopes[parent.0]
                .bindings
                .insert(ns.to_owned(), Binding::Namespace(id));
            // A Namespace binding is never an unevaluated string.
            self.scopes[parent.0].unevaluated.remove(ns);
        }
        id
    }

    /// Bind `name` to an already-typed `value` in `scope` as a FINAL (non-lazy)
    /// property, `_setitem(name, value, unevaluated=False)`. Used to bind macro
    /// call parameters (which are evaluated in the caller scope before binding).
    pub fn set_property_value(&mut self, scope: ScopeId, name: &str, value: XacroValue) {
        self.setitem(scope, name, value, false);
    }

    // ---- low-level scope ops (mirror dict.__getitem__/__setitem__/etc.) ----

    /// `dict.__contains__` for a single scope (no parent fallback).
    fn scope_has(&self, scope: ScopeId, key: &str) -> bool {
        self.scopes[scope.0].bindings.contains_key(key)
    }

    /// `Table.__contains__`: this scope OR any ancestor.
    pub fn contains(&self, scope: ScopeId, key: &str) -> bool {
        let mut cur = Some(scope);
        while let Some(id) = cur {
            if self.scope_has(id, key) {
                return true;
            }
            cur = self.scopes[id.0].parent;
        }
        false
    }

    /// `Table.top()`: walk parent links until the child of root (the top user
    /// scope as seen from `scope`). Used by `scope="global"`.
    fn top_of(&self, scope: ScopeId) -> ScopeId {
        let mut p = scope;
        // while p.parent is not p.root: p = p.parent
        while let Some(parent) = self.scopes[p.0].parent {
            if parent == self.root {
                break;
            }
            p = parent;
        }
        p
    }

    /// `Table.__getitem__` -> `_resolve_`: look up `key` starting at `scope`,
    /// walking to parents, resolving a lazy binding on first access (and caching
    /// the result). Returns `None` if unbound anywhere up the chain.
    ///
    /// This is the heart of the model: late binding falls out of resolving the
    /// stored RHS string *at access time* against the live table, and the
    /// circular-definition guard lives here.
    ///
    /// The public entry point TAKES the functions + subst context out of `self`
    /// once, then threads them (borrowed) through the whole resolution recursion;
    /// see [`PropertyTables::get_env`] for why re-taking per nested call would
    /// lose them.
    pub fn get(&mut self, scope: ScopeId, key: &str) -> Result<Option<XacroValue>, EvalError> {
        let functions = std::mem::take(&mut self.functions);
        let mut subst = self.subst.take();
        let result = self.get_env(scope, key, &functions, &mut subst);
        self.functions = functions;
        self.subst = subst;
        result
    }

    /// `get` threaded with an already-taken evaluation environment (`functions` +
    /// `subst`). Walks the scope chain and resolves a found key. Reusing the
    /// SAME borrowed env on every nested call is what fixes the re-entrancy bug
    /// where a nested `eval_text` (a lazy property whose RHS calls `load_yaml` /
    /// reads another lazy property) would otherwise re-`mem::take` the (now
    /// already-taken, hence EMPTY) functions/subst from `self`.
    pub(crate) fn get_env(
        &mut self,
        scope: ScopeId,
        key: &str,
        functions: &Namespace,
        subst: &mut Option<SubstContext>,
    ) -> Result<Option<XacroValue>, EvalError> {
        let mut cur = Some(scope);
        while let Some(id) = cur {
            if self.scope_has(id, key) {
                return self.resolve(id, key, functions, subst).map(Some);
            }
            cur = self.scopes[id.0].parent;
        }
        Ok(None)
    }

    /// Public entry point: evaluate `text` (resolving `${...}`/`$(...)`) at
    /// `scope` against these tables, returning a typed [`XacroValue`] per the
    /// typed-single-result rule. Takes the functions + subst env out of `self`
    /// once and threads it through the whole (possibly re-entrant) evaluation.
    pub fn eval_text_in(&mut self, scope: ScopeId, text: &str) -> Result<XacroValue, EvalError> {
        let functions = std::mem::take(&mut self.functions);
        let mut subst = self.subst.take();
        let result = self.eval_text_here(text, scope, &functions, &mut subst);
        self.functions = functions;
        self.subst = subst;
        result
    }

    /// Run `eval_text` over `raw` at `scope` against `self`, threading the
    /// already-taken evaluation environment (`functions` + `subst`). The env was
    /// moved out of `self` ONCE by the top-level entry point, so it is reused by
    /// every nested `eval_text`/resolution without re-taking (which would yield
    /// the emptied fields). `eval_text` may MUTATE `subst` (e.g. `$(eval)` reads
    /// the arg table; an `<xacro:arg>` writes it), so it is threaded as `&mut`.
    fn eval_text_here(
        &mut self,
        raw: &str,
        scope: ScopeId,
        functions: &Namespace,
        subst: &mut Option<SubstContext>,
    ) -> Result<XacroValue, EvalError> {
        eval_text(raw, self, scope, functions, subst)
    }

    /// `Table._resolve_(key)` for a key KNOWN to live in `scope`: if it is
    /// unevaluated, eval its stored string against `scope`, cache the typed
    /// result, and clear it from `unevaluated`; detect re-entrancy (circular
    /// definition) with the exact canonical error. Threaded with the evaluation
    /// environment so the nested `eval_text` sees the registered functions /
    /// substitution context.
    fn resolve(
        &mut self,
        scope: ScopeId,
        key: &str,
        functions: &Namespace,
        subst: &mut Option<SubstContext>,
    ) -> Result<XacroValue, EvalError> {
        // A `NameSpace` binding (`symbols[ns] = NameSpace(...)`): materialize the
        // namespace's reachable properties into a `XacroValue::Namespace` so a
        // `${ns.prop}` dotted access resolves. Materialized fresh per access (not
        // cached) because the namespace scope's contents can change between reads.
        if let Some(Binding::Namespace(ns_scope)) = self.scopes[scope.0].bindings.get(key) {
            let ns_scope = *ns_scope;
            let map = self.materialize_namespace(ns_scope, functions, subst, &mut Vec::new())?;
            return Ok(XacroValue::Namespace(map));
        }
        if self.scopes[scope.0].unevaluated.contains(key) {
            // Circular: re-entering a key already on the recursive stack.
            if self.scopes[scope.0].recursive.iter().any(|k| k == key) {
                let mut chain = self.scopes[scope.0].recursive.clone();
                chain.push(key.to_owned());
                return Err(EvalError::Runtime(format!(
                    "circular variable definition: {}\nConsider disabling lazy evaluation via lazy_eval=\"false\"",
                    chain.join(" -> ")
                )));
            }
            self.scopes[scope.0].recursive.push(key.to_owned());

            // Pull the stored RHS (it must be an Unevaluated string here).
            let raw = match self.scopes[scope.0].bindings.get(key) {
                Some(Binding::Unevaluated(s)) => s.clone(),
                // An unevaluated key whose binding is already Resolved is an
                // internal inconsistency; surface rather than silently diverge.
                _ => {
                    return Err(EvalError::Bridge(format!(
                        "'{key}' marked unevaluated but holds a resolved value"
                    )))
                }
            };

            // eval_text against THIS scope, then re-apply _eval_literal, exactly
            // `self._eval_literal(eval_text(stored, self))`. The eval_text can
            // recurse into this same `resolve` for nested property reads, which
            // is what the `recursive` guard above protects.
            let evaluated = self.eval_text_here(&raw, scope, functions, subst)?;
            let cached = match evaluated {
                // _eval_literal only re-coerces a *string* result; typed results
                // (dict/list/float/...) pass through untouched.
                XacroValue::Str(s) => eval_literal(&s),
                other => other,
            };

            self.scopes[scope.0]
                .bindings
                .insert(key.to_owned(), Binding::Resolved(cached));
            self.scopes[scope.0].unevaluated.remove(key);
            // pop the recursive guard for this key
            if let Some(pos) = self.scopes[scope.0].recursive.iter().position(|k| k == key) {
                self.scopes[scope.0].recursive.remove(pos);
            }
        }

        // Return the (now) resolved value.
        match self.scopes[scope.0].bindings.get(key) {
            Some(Binding::Resolved(v)) => Ok(v.clone()),
            // Still unevaluated here would be a logic error (we just resolved).
            _ => Err(EvalError::Bridge(format!(
                "'{key}' failed to resolve to a value"
            ))),
        }
    }

    /// Materialize a `NameSpace` scope into the [`IndexMap`] of properties exposed
    /// by `${ns.prop}` dotted access. Mirrors canonical `NameSpace.__getitem__`
    /// (which walks the parent chain): the result is the UNION of `ns_scope`'s own
    /// keys and all ANCESTOR keys, with nearer scopes shadowing farther ones (so
    /// `ns.prop` falls back to the including file's properties, verified vs the
    /// oracle). Each property value is resolved (lazy ones evaluated); a key whose
    /// resolution ERRORS is omitted (accessing it then raises, matching canonical's
    /// per-access lazy resolution rather than eagerly forcing an unrelated
    /// circular/erroring property). Nested `NameSpace` bindings are skipped to avoid
    /// re-entrant self-materialization (the parent scope contains the namespace's
    /// OWN binding); `seen` guards any residual cycle.
    fn materialize_namespace(
        &mut self,
        ns_scope: ScopeId,
        functions: &Namespace,
        subst: &mut Option<SubstContext>,
        seen: &mut Vec<ScopeId>,
    ) -> Result<indexmap::IndexMap<String, XacroValue>, EvalError> {
        if seen.contains(&ns_scope) {
            return Ok(indexmap::IndexMap::new());
        }
        seen.push(ns_scope);

        // Collect the reachable key set: own scope first, then ancestors, keeping
        // the FIRST (nearest) occurrence of each key so a nearer scope shadows a
        // farther one. Skip Namespace-typed bindings (a sibling/own namespace) to
        // avoid recursing back into this same materialization.
        let mut order: Vec<(ScopeId, String)> = Vec::new();
        let mut chosen: HashSet<String> = HashSet::new();
        let mut cur = Some(ns_scope);
        while let Some(id) = cur {
            // Iterate the scope's own keys in insertion order.
            let keys: Vec<String> = self.scopes[id.0].bindings.keys().cloned().collect();
            for k in keys {
                if chosen.contains(&k) {
                    continue;
                }
                if matches!(self.scopes[id.0].bindings.get(&k), Some(Binding::Namespace(_))) {
                    // A nested/sibling namespace binding: not a plain property,
                    // and including it risks self-recursion. Skip it.
                    continue;
                }
                chosen.insert(k.clone());
                order.push((id, k));
            }
            cur = self.scopes[id.0].parent;
        }

        let mut map = indexmap::IndexMap::new();
        for (id, k) in order {
            // Resolve each property at the scope it lives in; a resolution error
            // means the attribute is simply absent (canonical only raises on a
            // LIVE access of a bad property, which would then surface there).
            match self.resolve(id, &k, functions, subst) {
                Ok(v) => {
                    map.insert(k, v);
                }
                Err(_) => { /* omit erroring keys */ }
            }
        }
        seen.pop();
        Ok(map)
    }

    /// `Table._setitem(key, value, unevaluated)`: store a binding, applying
    /// `_eval_literal` to the (string) value, marking it `unevaluated` iff it is
    /// still a string after literal coercion and the caller requested lazy
    /// storage. Emits the "redefining global symbol" warning when `key` already
    /// exists in the root.
    ///
    /// `value` here is the value as canonical xacro hands it to `_setitem`: for a
    /// lazy property the raw RHS *string*; for an eager one the already
    /// `eval_text`-evaluated [`XacroValue`].
    fn setitem(&mut self, scope: ScopeId, key: &str, value: XacroValue, unevaluated: bool) {
        // redefining-global-symbol warning: `if key in self.root`.
        if self.scope_has(self.root, key) {
            self.warnings
                .push(format!("redefining global symbol: {key}"));
        }

        // value = self._eval_literal(value): only strings are coerced.
        let coerced = match value {
            XacroValue::Str(s) => eval_literal(&s),
            other => other,
        };

        match &coerced {
            // `if unevaluated and isinstance(value, str)`: a still-string lazy
            // value is stored as the deferred RHS and added to `unevaluated`.
            XacroValue::Str(s) if unevaluated => {
                let s = s.clone();
                self.scopes[scope.0]
                    .bindings
                    .insert(key.to_owned(), Binding::Unevaluated(s));
                self.scopes[scope.0].unevaluated.insert(key.to_owned());
            }
            // Otherwise it is a final typed value; ensure it is not lingering in
            // `unevaluated` (mirrors the `elif key in self.unevaluated: remove`).
            _ => {
                self.scopes[scope.0]
                    .bindings
                    .insert(key.to_owned(), Binding::Resolved(coerced));
                self.scopes[scope.0].unevaluated.remove(key);
            }
        }
    }

    /// `Table.__delitem__`: remove `key` from `scope` and every ancestor UP TO
    /// (not including) the root; if it exists in root, warn that a global symbol
    /// cannot be removed.
    fn delitem(&mut self, scope: ScopeId, key: &str) {
        let mut p = Some(scope);
        while let Some(id) = p {
            if id == self.root {
                break;
            }
            self.scopes[id.0].bindings.shift_remove(key);
            self.scopes[id.0].unevaluated.remove(key);
            p = self.scopes[id.0].parent;
        }
        if self.scope_has(self.root, key) {
            self.warnings
                .push(format!("Cannot remove global symbol: {key}"));
        }
    }

    // ---- the definition path: grab_property / set_property ----

    /// Port of canonical `grab_property`'s *property-model* core (the XML element
    /// handling (`check_attrs`, comment removal, node replacement) lives in the
    /// document pipeline). Defines a property from an already-extracted [`PropertyDef`]
    /// attribute bundle.
    ///
    /// `scope_id` is the scope the `<xacro:property>` is evaluated in (the macro
    /// body / top scope). Returns `Ok(())` on success, an [`EvalError`] on the
    /// fatal cases (`is_valid_name` rejects, mutual exclusion). Non-fatal issues
    /// are pushed to [`PropertyTables::warnings`].
    pub fn set_property(&mut self, scope_id: ScopeId, def: &PropertyDef) -> Result<(), EvalError> {
        // Take the evaluation environment (functions + subst) out of `self` ONCE
        // and thread it through every `eval_text` this definition triggers (the
        // name/remove/lazy_eval/value evals, plus any nested lazy resolution they
        // recurse into). See `get` for why re-taking per call would lose it.
        let functions = std::mem::take(&mut self.functions);
        let mut subst = self.subst.take();
        let result = self.set_property_env(scope_id, def, &functions, &mut subst);
        self.functions = functions;
        self.subst = subst;
        result
    }

    /// `set_property` threaded with an already-taken evaluation environment.
    fn set_property_env(
        &mut self,
        scope_id: ScopeId,
        def: &PropertyDef,
        functions: &Namespace,
        subst: &mut Option<SubstContext>,
    ) -> Result<(), EvalError> {
        let PropertyDef {
            name,
            value,
            default,
            remove,
            scope,
            lazy_eval,
        } = *def;

        // name = eval_text(name, table); the name may itself be an expression.
        let name = self
            .eval_text_here(name, scope_id, functions, subst)?
            .to_python_str();

        // is_valid_name + double-underscore guard (FATAL, like canonical).
        if !is_valid_name(&name) {
            return Err(EvalError::Runtime(format!(
                "Property names must be valid python identifiers: {name}"
            )));
        }
        if name.starts_with("__") {
            return Err(EvalError::Runtime(format!(
                "Property names must not start with double underscore:{name}"
            )));
        }

        // remove = get_boolean_value(eval_text(remove or 'false', table), remove)
        // Keep the TYPED eval_text result (so a non-str like ${1.5}/${[1]}/${None}
        // uses Python truthiness, not a stringified form) and pass the ORIGINAL
        // raw attribute text as the condition slot (canonical passes `remove`,
        // which is the raw attribute, `None` when absent; we mirror that with the
        // empty string, the value used for the absent default `'false'`).
        let remove_value =
            self.eval_text_here(remove.unwrap_or("false"), scope_id, functions, subst)?;
        let remove_flag =
            get_boolean_value(&remove_value, remove.unwrap_or_default()).map_err(EvalError::Runtime)?;

        // mutual exclusion of value / default / remove.
        let n_set =
            usize::from(value.is_some()) + usize::from(default.is_some()) + usize::from(remove_flag);
        if n_set > 1 {
            return Err(EvalError::Runtime(format!(
                "Property attributes default, value, and remove are mutually exclusive: {name}"
            )));
        }

        // remove= : delete up to root if present, then done.
        if remove_flag {
            if self.contains(scope_id, &name) {
                self.delitem(scope_id, &name);
            }
            return Ok(());
        }

        // default= : only-if-absent, local scope only; warn if scope given.
        // The effective RHS string to store (`value` chosen below).
        let mut rhs: Option<String> = value.map(str::to_owned);
        if let Some(def) = default {
            if scope.is_some() {
                self.warnings.push(format!(
                    "{name}: default property value can only be defined on local scope"
                ));
            }
            if !self.contains(scope_id, &name) {
                rhs = Some(def.to_owned());
            } else {
                // already present -> leave untouched, done.
                return Ok(());
            }
        }

        // If neither value nor default produced an RHS, canonical xacro stores a
        // sentinel ('**'+name -> elt) for debug; in the property-only port there
        // is no element, so treat a value-less, default-less, non-remove property
        // as a no-op store of the empty string under the '**'-prefixed key is not
        // observable. We instead require an RHS to define a usable property; a
        // missing one is a defined-but-unusable marker we skip storing.
        let rhs = match rhs {
            Some(r) => r,
            None => return Ok(()),
        };

        // lazy_eval default 'true'; resolve to a bool against the TYPED eval_text
        // result (so `lazy_eval="${0.0}"` -> bool(0.0)=False=eager, defined &
        // frozen, instead of erroring and silently losing the property). Condition
        // slot is the raw `lazy_eval` attribute, mirroring canonical.
        let lazy_value =
            self.eval_text_here(lazy_eval.unwrap_or("true"), scope_id, functions, subst)?;
        let mut lazy =
            get_boolean_value(&lazy_value, lazy_eval.unwrap_or_default()).map_err(EvalError::Runtime)?;

        // scope= : retarget the destination table; global/parent FORCE eager.
        let target = match scope {
            Some("global") => {
                lazy = false;
                SetTarget {
                    table: self.top_of(scope_id),
                    lazy,
                }
            }
            Some("parent") => {
                let parent = self.scopes[scope_id.0].parent;
                match parent {
                    Some(mut target) => {
                        lazy = false;
                        // If the current scope is NOT a NameSpace (i.e. a macro
                        // body), skip intervening NameSpaces to reach the
                        // caller's scope: `while isinstance(target, NameSpace):
                        // target = target.parent`.
                        if !self.scopes[scope_id.0].is_namespace {
                            while self.scopes[target.0].is_namespace {
                                match self.scopes[target.0].parent {
                                    Some(p) => target = p,
                                    None => break,
                                }
                            }
                        }
                        SetTarget {
                            table: target,
                            lazy,
                        }
                    }
                    None => {
                        // no parent scope at global scope -> warn, cannot store.
                        self.warnings
                            .push(format!("{name}: no parent scope at global scope "));
                        return Ok(());
                    }
                }
            }
            _ => SetTarget {
                table: scope_id,
                lazy,
            },
        };

        // `if not lazy_eval and isinstance(value, str): value = eval_text(...)`.
        // Eager: evaluate the RHS NOW against the *defining* scope (`table`, i.e.
        // scope_id), so early binding freezes against the current environment and
        // an eager self-reference reads the prior binding.
        if !target.lazy {
            let evaluated = self.eval_text_here(&rhs, scope_id, functions, subst)?;
            // _setitem(name, evaluated, unevaluated=False)
            self.setitem(target.table, &name, evaluated, false);
        } else {
            // Lazy: store the raw RHS string for resolution-on-first-access.
            // _setitem(name, rhs_string, unevaluated=True)
            self.setitem(target.table, &name, XacroValue::Str(rhs), true);
        }

        Ok(())
    }
}

/// `is_valid_name(name)`: a valid property/macro identifier is one whose Python
/// AST is exactly a single bare `Name`, i.e. a syntactically valid identifier
/// that is NOT a python keyword (keywords don't parse as a `Name`).
///
/// We reproduce this without an AST by applying Python's identifier grammar
/// directly: the first char is `XID_Start` (or `_`, which the Unicode spec lists
/// as `XID_Continue` but not `XID_Start`, yet Python allows to lead an
/// identifier), and every remaining char is `XID_Continue`. This is the SAME
/// grammar `ast.parse` accepts, so non-ASCII identifiers Python (and the live
/// xacro oracle) accept (`café`, `über`, `résumé`, `λ`, `你好`) validate here
/// too. Restricting to ASCII (the previous bug) would FATAL-reject a legal
/// canonical xacro file. A name is then valid iff it matches that grammar AND is
/// not a Python keyword (keywords / `True`/`False`/`None` don't parse as a bare
/// `Name` node, so canonical rejects them).
///
/// The double-underscore rejection is a SEPARATE check in `grab_property`
/// (callers apply it after this), matching canonical structure.
pub fn is_valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        // `_` is a valid identifier start in Python though not `XID_Start`.
        Some(c) if c == '_' || unicode_ident::is_xid_start(c) => {}
        _ => return false,
    }
    if !chars.all(unicode_ident::is_xid_continue) {
        return false;
    }
    !PYTHON_KEYWORDS.contains(&name)
}

/// Python 3 hard keywords plus the constant names that `ast.parse` does NOT
/// surface as a plain `Name` node (`True`/`False`/`None` parse as `Constant`,
/// not `Name`, so canonical `is_valid_name` rejects them too).
const PYTHON_KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield",
];

/// `get_boolean_value(value, condition)`: a faithful port of canonical xacro's
/// `get_boolean_value` (`xacro/__init__.py:851`):
/// ```python
/// def get_boolean_value(value, condition):
///     try:
///         if isinstance(value, str):
///             if value == 'true' or value == 'True':   return True
///             elif value == 'false' or value == 'False': return False
///             else:                                      return bool(int(value))
///         else:
///             return bool(value)        # Python truthiness for non-str
///     except Exception:
///         raise XacroException('Xacro conditional "%s" evaluated to "%s", '
///                              'which is not a boolean expression.'
///                              % (condition, value))
/// ```
///
/// The crucial distinction the previous port lost: `value` is the ALREADY-TYPED
/// [`XacroValue`] that `eval_text` produced, NOT a pre-stringified `String`. A
/// `str` value follows the `true`/`false`/`int` rule; EVERY OTHER variant
/// (`${1.5}`, `${[1]}`, `${None}`, `${dict(...)}`, ...) follows Python truthiness
/// `bool(value)`. Stringifying first (the old bug) collapsed `1.5`/`[1]`/`None`
/// to their `str()` and ran only the string rule, so `remove="${1.5}"` /
/// `lazy_eval="${0.0}"` etc. diverged from canonical (verified against the live
/// `/opt/ros/jazzy/bin/xacro` oracle).
///
/// `condition` is the ORIGINAL raw attribute text (e.g. `${'yes'}`), used as the
/// first slot of the error message exactly as canonical does (the second slot is
/// the evaluated value's `str()`).
fn get_boolean_value(value: &XacroValue, condition: &str) -> Result<bool, String> {
    let err = || {
        Err(format!(
            "Xacro conditional \"{}\" evaluated to \"{}\", which is not a boolean expression.",
            condition,
            value.to_python_str()
        ))
    };
    match value {
        // isinstance(value, str): the true/True/false/False else bool(int(v)) rule.
        XacroValue::Str(s) => match s.as_str() {
            "true" | "True" => Ok(true),
            "false" | "False" => Ok(false),
            // bool(int(value)): Python int() strips surrounding whitespace; a
            // non-integer string raises -> the canonical error.
            other => match other.trim().parse::<i64>() {
                Ok(i) => Ok(i != 0),
                Err(_) => err(),
            },
        },
        // else: bool(value), Python truthiness for every non-str typed value.
        XacroValue::Int(i) => Ok(*i != BigInt::from(0)),
        XacroValue::Float(f) => Ok(*f != 0.0),
        XacroValue::Bool(b) => Ok(*b),
        XacroValue::List(items) => Ok(!items.is_empty()),
        XacroValue::Tuple(items) => Ok(!items.is_empty()),
        // `NameSpace` is a `dict` subclass: truthy iff non-empty.
        XacroValue::Dict(map) | XacroValue::Namespace(map) => Ok(!map.is_empty()),
        XacroValue::Null => Ok(false),
    }
}
