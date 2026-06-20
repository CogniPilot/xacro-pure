//! MACROS: `xacro:macro` definitions + macro CALL expansion.
//!
//! A faithful port of canonical xacro's macro machinery (`xacro/__init__.py`):
//! `Macro`, `parse_macro_arg` / `re_macro_arg`, `grab_macro`, `resolve_macro`,
//! `handle_macro_call`, `handle_dynamic_macro_call`, `eval_default_arg`, and the
//! `xacro:insert_block` branch of `eval_all`.
//!
//! ## The parameter spec grammar (`parse_macro_arg`)
//! A `params="..."` string is a whitespace-separated list of parameter specs,
//! each of the form `<param>[:=|=][^|]<default>`:
//!   * a bare `name`: a required positional/keyword param;
//!   * `name:=default` / `name=default`: a param with a default expression;
//!   * `name:=^`: FORWARD: bind from the CALLER's scope (`symbols[name]`),
//!     erroring if undefined;
//!   * `name:=^|default`: forward if defined in the caller, else use `default`;
//!   * `*block`: a single positional BLOCK param (one child element of the call);
//!   * `**content`: a multi-block param (ALL child elements of the call).
//!
//! Canonical parses this with one regex (`re_macro_arg`) applied repeatedly to
//! the front of the string. We reproduce the regex's behavior directly (see
//! [`parse_macro_arg`]): the alternation
//! `\$\{.*?\}|\$\(.*?\)|(?:'.*?'|".*?"|[^\s'"]+)+|` for the default value, the
//! optional `:?=`, the optional `\^\|?` forward marker.
//!
//! ## Macro tables + namespaces
//! Canonical keeps a `macros` `Table` parallel to the `symbols` `Table`. A
//! namespaced include (`<xacro:include ns="foo">`) creates `macros['foo']` as a
//! nested `NameSpace`, so `<xacro:foo.bar/>` resolves by splitting on `.`
//! (`resolve_macro`). We model this with a [`MacroTables`] arena of
//! [`MacroScope`]s addressed by [`MacroScopeId`], structurally parallel to the
//! property [`crate::table::PropertyTables`] arena: a scope holds its own
//! `name -> Macro` map plus a map of child NAMESPACE scopes and a `parent` link
//! for lookup fallback.
//!
//! ## Macro body storage
//! A [`Macro`] stores its body as an owned [`xmltree::Element`] (the
//! `<xacro:macro>` element itself; its children are the body). Each expansion
//! deep-clones the body (`xmltree::Element: Clone`), mirroring
//! `m.body.cloneNode(deep=True)`, so the same macro instantiates repeatedly.

use std::collections::HashMap;

use xmltree::Element;

use crate::table::ScopeId;

/// One parsed macro definition: the body element, the ordered parameter names,
/// and the per-parameter `(forward, default)` map. Mirrors canonical `Macro`.
#[derive(Debug, Clone)]
pub struct Macro {
    /// The `<xacro:macro>` element; its `children` are the macro body. Cloned per
    /// expansion (`cloneNode(deep=True)`).
    pub body: Element,
    /// Parameter names in declaration order (including `*block`/`**content`,
    /// whose names retain the leading `*`/`**`).
    pub params: Vec<String>,
    /// `param -> (forward_variable, default)`. Present only for params that have a
    /// default spec. `forward_variable` is `Some(param)` iff the `^` marker was
    /// given (forward from caller scope), else `None`. `default` is the default
    /// string, or `None` if only `^` (no `|default`) was given.
    pub defaultmap: HashMap<String, (Option<String>, Option<String>)>,
}

/// A parsed default spec for one parameter: `(forward, default)`.
///   * `forward`: `Some(param)` if `^`/`^|` was given, else `None`.
///   * `default`: the default string, or `None` if absent.
pub type DefaultSpec = (Option<String>, Option<String>);

/// The result of parsing one parameter spec off the front of a `params` string:
/// the param name, its optional `(forward, default)` spec, and the REST of the
/// string to continue parsing.
struct ParsedArg {
    param: String,
    spec: Option<DefaultSpec>,
    rest: String,
}

/// Parse the first parameter spec from a macro parameter string `s`, a faithful
/// port of canonical `parse_macro_arg` + its `re_macro_arg` regex.
///
/// The regex is:
/// ```text
/// ^\s*([^\s:=]+?)\s*:?=\s*(\^\|?)?(DEFAULT)(?:\s+|$)(.*)
/// DEFAULT = \$\{.*?\}|\$\(.*?\)|(?:'.*?'|".*?"|[^\s'"]+)+|
/// ```
/// i.e.: optional leading space, the param name (non-greedy run of chars that are
/// not space/`:`/`=`), optional space, an OPTIONAL `:?=` (so `name:=d` and
/// `name=d` both parse, and a bare `name` with no `=` falls through to the
/// else-branch), optional space, an optional `^`/`^|` forward marker, the default
/// value, then trailing space-or-end and the rest.
///
/// When the regex does NOT match (no `=` present, a bare parameter), canonical
/// splits off the first whitespace-delimited token as the param with no spec.
fn parse_macro_arg(s: &str) -> ParsedArg {
    if let Some(parsed) = try_re_macro_arg(s) {
        return parsed;
    }
    // else branch: `result = s.lstrip().split(None, 1)`; param = result[0],
    // rest = result[1] if present else ''. `split(None, 1)` splits on the first
    // run of whitespace.
    let trimmed = s.trim_start();
    let mut it = trimmed.splitn(2, char::is_whitespace);
    let param = it.next().unwrap_or("").to_owned();
    // splitn keeps the remainder verbatim after the FIRST whitespace char, but
    // Python `split(None, 1)` collapses the leading whitespace run of the
    // remainder. Re-trim the remainder's leading whitespace to match.
    let rest = it.next().map(str::trim_start).unwrap_or("").to_owned();
    ParsedArg {
        param,
        spec: None,
        rest,
    }
}

/// Try to match the `re_macro_arg` regex against the front of `s`. Returns the
/// parsed arg on a match, `None` if the regex would not match (no `=`).
fn try_re_macro_arg(s: &str) -> Option<ParsedArg> {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut i = 0;

    // ^\s*: leading whitespace.
    while i < n && chars[i].is_whitespace() {
        i += 1;
    }

    // ([^\s:=]+?): the param name, a NON-GREEDY run of chars that are not space/:/=.
    // Because the next token is `\s*:?=`, the non-greedy `+?` consumes the
    // SHORTEST name such that the rest still matches; with the `[^\s:=]` class the
    // name simply runs up to the first space/`:`/`=`. (`+?` differs from `+` only
    // when the following pattern could match earlier, which it cannot here since
    // the name class already excludes the delimiters.)
    let name_start = i;
    while i < n && chars[i] != ':' && chars[i] != '=' && !chars[i].is_whitespace() {
        i += 1;
    }
    if i == name_start {
        return None; // empty name -> no match
    }
    let param: String = chars[name_start..i].iter().collect();

    // \s*: optional space before the (optional) `:?=`.
    while i < n && chars[i].is_whitespace() {
        i += 1;
    }

    // :?= is an optional `:` then a REQUIRED `=`. If there is no `=` here, the
    // whole regex fails to match (the bare-parameter case).
    if i < n && chars[i] == ':' {
        i += 1;
    }
    if i >= n || chars[i] != '=' {
        return None; // no `=` -> regex does not match
    }
    i += 1; // consume `=`

    // \s*: optional space after `=`.
    while i < n && chars[i].is_whitespace() {
        i += 1;
    }

    // (\^\|?)? is the optional forward marker: `^` or `^|`.
    let mut forward = false;
    if i < n && chars[i] == '^' {
        forward = true;
        i += 1;
        if i < n && chars[i] == '|' {
            i += 1;
        }
    }

    // (DEFAULT): the default value alternation.
    let (default, after_default) = match_default_value(&chars, i);
    i = after_default;

    // (?:\s+|$): trailing whitespace run OR end-of-string. The default
    // alternation's `[^\s'"]+` stops at whitespace, and `${...}`/`$(...)`/quoted
    // forms are followed by either space or end, so this is satisfied unless the
    // default ran into a non-space non-end char (impossible here). Consume the
    // trailing whitespace run.
    while i < n && chars[i].is_whitespace() {
        i += 1;
    }

    // (.*): the rest of the string.
    let rest: String = chars[i..].iter().collect();

    // Mirror canonical: `if not default: default = None`. An empty-string default
    // (the trailing `|` empty alternative) becomes `None`.
    let default_opt = if default.is_empty() {
        None
    } else {
        Some(default)
    };
    let spec = Some((forward.then(|| param.clone()), default_opt));

    Some(ParsedArg { param, spec, rest })
}

/// Match the DEFAULT-value alternation `\$\{.*?\}|\$\(.*?\)|(?:'.*?'|".*?"|[^\s'"]+)+|`
/// at position `start`, returning `(matched_text, end_index)`. The final empty
/// alternative means this ALWAYS matches (possibly the empty string).
fn match_default_value(chars: &[char], start: usize) -> (String, usize) {
    let n = chars.len();
    let i = start;

    // `\$\{.*?\}`: `${` ... first `}` (non-greedy).
    if i + 1 < n && chars[i] == '$' && chars[i + 1] == '{' {
        if let Some(close) = (i + 2..n).find(|&k| chars[k] == '}') {
            return (chars[i..=close].iter().collect(), close + 1);
        }
    }
    // `\$\(.*?\)`: `$(` ... first `)` (non-greedy).
    if i + 1 < n && chars[i] == '$' && chars[i + 1] == '(' {
        if let Some(close) = (i + 2..n).find(|&k| chars[k] == ')') {
            return (chars[i..=close].iter().collect(), close + 1);
        }
    }

    // `(?:'.*?'|".*?"|[^\s'"]+)+` matches one or more of: a single-quoted run, a
    // double-quoted run, or a run of non-space non-quote chars. Greedy `+`.
    let mut j = i;
    loop {
        if j >= n {
            break;
        }
        let c = chars[j];
        if c == '\'' {
            // '.*?': up to the next single quote.
            if let Some(close) = (j + 1..n).find(|&k| chars[k] == '\'') {
                j = close + 1;
                continue;
            }
            break; // unterminated quote: stop (this `'` is not consumed)
        } else if c == '"' {
            if let Some(close) = (j + 1..n).find(|&k| chars[k] == '"') {
                j = close + 1;
                continue;
            }
            break;
        } else if !c.is_whitespace() {
            // [^\s'"]+: a run of non-space non-quote chars.
            j += 1;
            continue;
        } else {
            break; // whitespace terminates the default value
        }
    }
    // The trailing empty alternative: if nothing matched, `j == i` and we return
    // the empty string.
    (chars[i..j].iter().collect(), j)
}

/// Parse a whole `params` attribute string into the ordered param list + the
/// default map, the loop body of canonical `grab_macro`.
pub fn parse_params(params: &str) -> (Vec<String>, HashMap<String, DefaultSpec>) {
    let mut names = Vec::new();
    let mut defaultmap = HashMap::new();
    let mut rest = params.to_owned();
    while !rest.is_empty() {
        let parsed = parse_macro_arg(&rest);
        if parsed.param.is_empty() {
            break; // defensive: no progress -> stop (only on pathological input)
        }
        names.push(parsed.param.clone());
        if let Some(spec) = parsed.spec {
            defaultmap.insert(parsed.param, spec);
        }
        if parsed.rest == rest {
            break; // no progress guard
        }
        rest = parsed.rest;
    }
    (names, defaultmap)
}

/// An index into the [`MacroTables`] scope arena. Parallel to a property
/// `ScopeId`: the structural analogue of a canonical `macros` `Table`/`NameSpace`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroScopeId(usize);

/// A single macro scope: its own `name -> Macro` map, a map of child NAMESPACE
/// scopes (created by `<xacro:include ns=..>`), a parent link, and the PARALLEL
/// property [`ScopeId`] this macro scope was created alongside.
///
/// Canonical xacro keeps the `macros` `Table` and the `symbols` `Table`
/// structurally in lockstep: `process_include` creates `macros[ns]` and
/// `symbols[ns]` together, and `resolve_macro` traverses BOTH on the SAME `.`-path
/// so a resolved macro yields both its defining `scoped_macros` AND
/// `scoped_symbols`. We mirror that by storing, on each macro scope, the property
/// scope it parallels: the top macro scope parallels the top property scope, and a
/// namespace macro scope parallels the namespace's [`crate::table::PropertyTables`]
/// `NameSpace`. [`MacroTables::resolve`] returns this so a namespaced macro body is
/// parented on its OWN namespace's property scope (not the caller's).
#[derive(Debug, Default)]
struct MacroScope {
    macros: HashMap<String, Macro>,
    /// Child namespace scopes by name (`macros[ns] = NameSpace()`).
    namespaces: HashMap<String, MacroScopeId>,
    parent: Option<MacroScopeId>,
    /// The parallel property [`ScopeId`] (the lockstep partner of this macro
    /// scope). `None` only transiently before it is wired up.
    symbol_scope: Option<ScopeId>,
}

/// The arena of macro scopes, parallel to [`crate::table::PropertyTables`].
/// Macro lookup walks the parent chain (a macro defined in an outer scope is
/// visible to an inner one), and namespaced names traverse the `namespaces` maps.
#[derive(Debug)]
pub struct MacroTables {
    scopes: Vec<MacroScope>,
    top: MacroScopeId,
}

impl MacroTables {
    /// A fresh macro environment with a single top scope, paralleling the property
    /// `top_symbol_scope` (so a top-level macro's body is parented on the top
    /// property scope, matching canonical's `scoped_symbols = symbols`).
    pub fn new(top_symbol_scope: ScopeId) -> Self {
        MacroTables {
            scopes: vec![MacroScope {
                symbol_scope: Some(top_symbol_scope),
                ..MacroScope::default()
            }],
            top: MacroScopeId(0),
        }
    }

    /// The top macro scope id.
    pub fn top_scope(&self) -> MacroScopeId {
        self.top
    }

    /// Define (or overwrite) a macro `name` in `scope`. Mirrors `macros[name] = m`.
    pub fn define(&mut self, scope: MacroScopeId, name: &str, m: Macro) {
        self.scopes[scope.0].macros.insert(name.to_owned(), m);
    }

    /// Create (or fetch) a child NAMESPACE macro scope `ns` under `scope`, paired
    /// with its parallel property `symbol_scope`, returning its id. Mirrors
    /// `macros[ns] = NameSpace()` alongside `symbols[ns] = NameSpace(...)` in
    /// `process_include` (re-using an existing one if `ns` was already declared).
    pub fn namespace(&mut self, scope: MacroScopeId, ns: &str, symbol_scope: ScopeId) -> MacroScopeId {
        if let Some(&existing) = self.scopes[scope.0].namespaces.get(ns) {
            return existing;
        }
        let id = MacroScopeId(self.scopes.len());
        self.scopes.push(MacroScope {
            parent: Some(scope),
            symbol_scope: Some(symbol_scope),
            ..MacroScope::default()
        });
        self.scopes[scope.0].namespaces.insert(ns.to_owned(), id);
        id
    }

    /// Push a FRESH child macro scope under `parent`, paired with the body's
    /// property `symbol_scope`, returning its id. This is canonical's
    /// `scoped_macros = Table(scoped_macros)` for a macro body: a throwaway child
    /// in which any `<xacro:macro>` grabbed during body expansion is CONFINED (it
    /// lives in this child and is discarded after the call), while sibling-macro
    /// lookup still walks up to the defining/namespace scope via `parent`.
    pub fn push_child(&mut self, parent: MacroScopeId, symbol_scope: ScopeId) -> MacroScopeId {
        let id = MacroScopeId(self.scopes.len());
        self.scopes.push(MacroScope {
            parent: Some(parent),
            symbol_scope: Some(symbol_scope),
            ..MacroScope::default()
        });
        id
    }

    /// Look up a single (non-namespaced) macro name in `scope` or any ancestor.
    fn lookup_direct(&self, scope: MacroScopeId, name: &str) -> Option<&Macro> {
        let mut cur = Some(scope);
        while let Some(id) = cur {
            if let Some(m) = self.scopes[id.0].macros.get(name) {
                return Some(m);
            }
            cur = self.scopes[id.0].parent;
        }
        None
    }

    /// Look up a namespace name in `scope` or any ancestor (for the `.`-traversal).
    fn lookup_namespace(&self, scope: MacroScopeId, ns: &str) -> Option<MacroScopeId> {
        let mut cur = Some(scope);
        while let Some(id) = cur {
            if let Some(&child) = self.scopes[id.0].namespaces.get(ns) {
                return Some(child);
            }
            cur = self.scopes[id.0].parent;
        }
        None
    }

    /// `resolve_macro(fullname, macros, symbols)`: resolve a macro name, trying the
    /// FULL name first, then splitting on `.` into namespaces + a final name.
    /// Returns the resolved [`Macro`] (cloned) PLUS its DEFINING scopes: both the
    /// macro [`MacroScopeId`] (`scoped_macros`) and the parallel property
    /// [`ScopeId`] (`scoped_symbols`), exactly as canonical's `resolve_macro`
    /// returns `(scoped_macros, scoped_symbols, m)`. `None` if unresolved.
    ///
    /// Two cases, matching canonical's `_resolve`:
    ///   * DIRECT hit (the full name resolves in `caller_mac` or an ancestor): the
    ///     defining scopes are the CALLER's own scopes (`return macros, symbols,
    ///     macros[fullname]`). So a plain top-level macro's body is parented on the
    ///     caller's symbol scope, which is why a plain macro sees a property the
    ///     CALL SITE defined (verified vs the oracle).
    ///   * NAMESPACE traversal (`macros[ns]...[name]`): the defining scopes are the
    ///     final namespace's macro scope AND its parallel property scope, so a
    ///     namespaced macro's body sees its OWN namespace's properties (not the
    ///     caller's).
    pub fn resolve(
        &self,
        caller_mac: MacroScopeId,
        caller_prop: ScopeId,
        fullname: &str,
    ) -> Option<(Macro, MacroScopeId, ScopeId)> {
        // try the full name directly (in `caller_mac` or an ancestor). The defining
        // scopes are the CALLER's own scopes, mirroring `_resolve([], fullname,
        // macros, symbols)` returning the passed-in `macros`/`symbols`.
        if let Some(m) = self.lookup_direct(caller_mac, fullname) {
            return Some((m.clone(), caller_mac, caller_prop));
        }
        // split into namespaces + name, traverse both macro + symbol structures.
        if !fullname.contains('.') {
            return None;
        }
        let mut parts: Vec<&str> = fullname.split('.').collect();
        let name = parts.pop()?;
        let mut cur = caller_mac;
        let mut first = true;
        for ns in parts {
            let next = if first {
                self.lookup_namespace(cur, ns)?
            } else {
                // after the first hop, only look in THAT namespace directly
                self.scopes[cur.0].namespaces.get(ns).copied()?
            };
            cur = next;
            first = false;
        }
        let m = self.scopes[cur.0].macros.get(name)?.clone();
        // the resolved namespace macro scope's parallel property scope is the
        // defining symbol scope (`scoped_symbols`).
        let sym = self.scopes[cur.0].symbol_scope?;
        Some((m, cur, sym))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(params: &str) -> Vec<String> {
        parse_params(params).0
    }

    #[test]
    fn bare_params() {
        assert_eq!(names("a b c"), vec!["a", "b", "c"]);
    }

    #[test]
    fn value_default_colon_eq() {
        let (n, d) = parse_params("a b:=1.0 c");
        assert_eq!(n, vec!["a", "b", "c"]);
        assert_eq!(d.get("b"), Some(&(None, Some("1.0".to_owned()))));
        assert!(!d.contains_key("a"));
    }

    #[test]
    fn value_default_plain_eq() {
        let (_n, d) = parse_params("b=world");
        assert_eq!(d.get("b"), Some(&(None, Some("world".to_owned()))));
    }

    #[test]
    fn forward_bare() {
        let (_n, d) = parse_params("x:=^");
        assert_eq!(d.get("x"), Some(&(Some("x".to_owned()), None)));
    }

    #[test]
    fn forward_with_default() {
        let (_n, d) = parse_params("x:=^|0.0");
        assert_eq!(d.get("x"), Some(&(Some("x".to_owned()), Some("0.0".to_owned()))));
    }

    #[test]
    fn quoted_default_with_spaces() {
        let (n, d) = parse_params("color:='1.0 0.82 0.12 1.0' parent:=world");
        assert_eq!(n, vec!["color", "parent"]);
        assert_eq!(
            d.get("color"),
            Some(&(None, Some("'1.0 0.82 0.12 1.0'".to_owned())))
        );
        assert_eq!(d.get("parent"), Some(&(None, Some("world".to_owned()))));
    }

    #[test]
    fn expr_default() {
        let (_n, d) = parse_params("a:=${b + 1}");
        assert_eq!(d.get("a"), Some(&(None, Some("${b + 1}".to_owned()))));
    }

    #[test]
    fn extension_default() {
        let (_n, d) = parse_params("a:=$(arg x)");
        assert_eq!(d.get("a"), Some(&(None, Some("$(arg x)".to_owned()))));
    }

    #[test]
    fn block_params() {
        let n = names("*origin **content");
        assert_eq!(n, vec!["*origin", "**content"]);
    }

    #[test]
    fn multiline_params() {
        // The SO-ARM macro's real param string (newline-separated, indented).
        let params = "\n      prefix\n      color:='1.0 0.82 0.12 1.0'\n      parent:=world\n      x:=0.0";
        let (n, d) = parse_params(params);
        assert_eq!(n, vec!["prefix", "color", "parent", "x"]);
        assert_eq!(
            d.get("color"),
            Some(&(None, Some("'1.0 0.82 0.12 1.0'".to_owned())))
        );
        assert_eq!(d.get("x"), Some(&(None, Some("0.0".to_owned()))));
        assert!(!d.contains_key("prefix"));
    }

    #[test]
    fn namespaced_resolution() {
        // Build real property scopes so the macro tables' parallel symbol-scope
        // ids are genuine (the structural lockstep canonical maintains).
        let mut props = crate::table::PropertyTables::new();
        let top_prop = props.top_scope();
        let ns_prop = props.push_namespace(top_prop, "foo");

        let mut t = MacroTables::new(top_prop);
        let top = t.top_scope();
        let ns = t.namespace(top, "foo", ns_prop);
        let m = Macro {
            body: Element::new("dummy"),
            params: vec![],
            defaultmap: HashMap::new(),
        };
        t.define(ns, "bar", m);
        // namespaced lookup returns the macro + the namespace's macro+symbol scopes.
        let (_m, mac_scope, sym_scope) = t.resolve(top, top_prop, "foo.bar").expect("foo.bar");
        assert_eq!(mac_scope, ns);
        assert_eq!(sym_scope, ns_prop);
        // a bare (non-namespaced) `bar` is not visible from the top scope.
        assert!(t.resolve(top, top_prop, "bar").is_none());
    }
}
