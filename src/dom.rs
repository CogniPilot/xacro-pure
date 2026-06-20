//! The DOCUMENT-PROCESSING PIPELINE: a faithful port of canonical xacro's
//! `process_doc` + `eval_all` (`xacro/__init__.py:902,1045`) over a MUTABLE XML
//! DOM (`xmltree`).
//!
//! `process_document` parses the source, builds a [`PropertyTables`] with a
//! [`SubstContext`] (the substitution-args / `$(...)` state) and a registered
//! `load_yaml`, then walks the tree with [`eval_all`], mutating it in place:
//!   * every TEXT node + non-`xacro:` ATTRIBUTE value is `eval_text`-evaluated
//!     and replaced with its `str()`;
//!   * `<xacro:property>` defines a property (via the property model);
//!   * `<xacro:arg name=.. default=..>` declares a substitution arg (default-if-
//!     absent), then is removed;
//!   * `<xacro:if>` / `<xacro:unless>` keep-or-drop their subtree on the boolean
//!     condition (recursing into a kept subtree, then SPLICING its children in
//!     place, `replace_node(content_only=True)`);
//!   * COMMENT nodes are passed through, unless an `xacro:eval-comments` pragma
//!     comment turns on comment evaluation for following siblings;
//!   * `xacro:*` attributes and the `xmlns:xacro` declaration are stripped from
//!     output.
//!
//! ## Macros + includes
//!   * `<xacro:macro name=.. params=..>`: [`crate::macros`] grabs the definition
//!     (parsed params, defaults, `^`/`^|` forwarding, `*block`/`**content`) into
//!     the [`MacroTables`] and removes the node;
//!   * a macro CALL element (`<ns:name .../>` or `<xacro:call macro=..>`) binds
//!     its params (positional/keyword + forwarded `^` + defaults), binds block
//!     params from its child elements, pushes a macro-body scope, expands the
//!     cloned body, and splices the result;
//!   * `<xacro:insert_block name=..>` inserts a bound `*`/`**` block;
//!   * `<xacro:include filename=.. [ns=..] [optional=..]>`: [`crate::includes`]
//!     resolves the filename (eval_text, relative to the current file, glob), reads
//!     it via the INJECTED reader, parses + recurses `eval_all` over it (honoring
//!     `ns=`/`optional=`/xmlns import), and splices the included children in place
//!     of the `<xacro:include>` element. Includes are processed IN ORDER inside
//!     `eval_all`, exactly as canonical's `ros2` branch does (no separate pass).
//!
//! ## DOM choice + wasm-cleanliness
//! `xmltree` (over `xml-rs`) is the same pure-Rust mutable tree `xacro-rs`/
//! `xurdf` use. Both build for `wasm32-unknown-unknown` (verified). It surfaces
//! the namespace PREFIX per element (so a `<xacro:property>` is `name="property",
//! prefix=Some("xacro")`) and the namespace-decl map (so `xmlns:xacro` can be
//! stripped). The faithful pretty-printer (`fixed_writexml`) and byte-parity live
//! in [`crate::serialize`]; the tree here is built with `xmltree`.
//!
//! ## File access seam (wasm-capable)
//! `xacro:include` reads files through an injected [`IncludeReader`] (the same
//! seam pattern as `load_yaml`'s): native callers supply a `std::fs`-backed
//! reader; a wasm caller supplies a virtual-FS map. The core never calls
//! `std::fs` directly in the include path.

use std::collections::HashMap;

use xmltree::{Element, XMLNode};

use crate::error::EvalError;
use crate::includes::{get_include_files, IncludeReader};
use crate::load_yaml::register_load_yaml_from_source;
use crate::macros::{parse_params, Macro, MacroScopeId, MacroTables};
use crate::namespace::Namespace;
use crate::substitution_args::{PackageResolver, SubstContext};
use crate::table::{PropertyDef, PropertyTables, ScopeId};
use crate::value::XacroValue;

/// The xacro namespace prefix and URI.
const XACRO_PREFIX: &str = "xacro";
const XACRO_NS_URI: &str = "http://www.ros.org/wiki/xacro";

/// An error from processing a xacro document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessError {
    /// The input was not well-formed XML.
    Parse(String),
    /// An evaluation (expression / property / substitution) failed.
    Eval(EvalError),
    /// A `xacro:*` element this pipeline does not implement. Carries the local name.
    Unsupported(String),
    /// Serializing the expanded tree back to XML failed.
    Serialize(String),
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessError::Parse(m) => write!(f, "XML parse error: {m}"),
            ProcessError::Eval(e) => write!(f, "{e}"),
            ProcessError::Unsupported(n) => {
                write!(f, "xacro:{n} is not implemented")
            }
            ProcessError::Serialize(m) => write!(f, "XML serialize error: {m}"),
        }
    }
}

impl std::error::Error for ProcessError {}

impl From<EvalError> for ProcessError {
    fn from(e: EvalError) -> Self {
        ProcessError::Eval(e)
    }
}

/// The expansion context threaded alongside [`PropertyTables`]: the macro
/// tables, the per-property-scope BLOCK-parameter store, the injected include
/// reader, and the filestack (for relative include resolution + cycle detection).
///
/// Kept separate from [`PropertyTables`] (which the eval layer already threads)
/// so the borrow split is clean: `eval_all` takes `&mut PropertyTables` AND
/// `&mut Ctx` and mutates both without aliasing.
struct Ctx<'r> {
    /// The macro definitions arena.
    macros: MacroTables,
    /// Block-parameter bindings: `(property_scope, block_name) -> child elements`.
    /// Canonical stores these in the `symbols` Table under `*name`/`**name`; we
    /// keep them out of the typed property table (which holds [`XacroValue`]s)
    /// since a block is XML, not a value. Keyed by the property [`ScopeId`] the
    /// macro body evaluates in plus the block name (without the `*`/`**` sigil).
    blocks: HashMap<(ScopeId, String), BlockBinding>,
    /// The injected include reader (native FS / wasm virtual map).
    reader: &'r dyn IncludeReader,
    /// The stack of currently-processed file paths (`filestack`): the last entry
    /// is the current file (relative-include base + `$(dirname)`), and the whole
    /// stack is checked for include cycles.
    filestack: Vec<Option<String>>,
    /// Property-scope parent links for scopes we create here (macro-body +
    /// namespace scopes). [`PropertyTables`] does not expose its internal parent
    /// chain, so block lookup (which keys on [`ScopeId`]) walks THIS mirror of the
    /// parent links we established when pushing those scopes.
    scope_parents: HashMap<ScopeId, Option<ScopeId>>,
}

/// The lockstep pair of scopes threaded through expansion: the PROPERTY scope and
/// the parallel MACRO scope. They move together (a namespaced include / macro
/// expansion descends both), so bundling them keeps signatures short and makes the
/// pairing explicit (canonical threads `symbols` + `macros` side by side).
#[derive(Debug, Clone, Copy)]
struct Scopes {
    /// The property [`ScopeId`] (`symbols`).
    prop: ScopeId,
    /// The macro [`MacroScopeId`] (`macros`).
    mac: MacroScopeId,
}

/// A bound block parameter: the child element(s) captured from a macro call, plus
/// whether it is a single `*`-block (insert content_only=False, the element
/// itself) or a `**`-block (content_only=True, the element's children).
#[derive(Clone)]
struct BlockBinding {
    /// The captured element(s). A `*`-block captures exactly one; a `**`-block
    /// captures one (the call node whose children are inserted).
    element: Element,
    /// `true` for a `**`-block (`content_only=True`: insert children),
    /// `false` for a `*`-block (`content_only=False`: insert the element itself).
    content_only: bool,
}

/// Process a xacro `source` document into expanded XML, given the `mappings`
/// (substitution `$(arg)` values), a `resolver` for `$(find)`, and a `read_yaml`
/// callback that `xacro.load_yaml(filename)` uses to obtain YAML source.
///
/// Includes are read through the default native [`crate::includes::FsIncludeReader`]
/// (so relative + glob includes work against the real filesystem). For a wasm /
/// hermetic caller that must control all file access, use [`process_document_with`].
///
/// `current_file` is the path of the source document (for relative include
/// resolution + `$(dirname)`); pass `None` if the source has no on-disk location
/// (then includes resolve relative to `.`).
///
/// NOTE: non-fatal warnings (see [`PropertyTables::warnings`]) are collected
/// during expansion but NOT surfaced by this entry point; only the expanded
/// XML is returned (canonical xacro prints them to stderr).
#[cfg(not(target_arch = "wasm32"))]
pub fn process_document<R>(
    source: &str,
    mappings: HashMap<String, String>,
    resolver: Box<dyn PackageResolver>,
    read_yaml: R,
) -> Result<String, ProcessError>
where
    R: Fn(&str) -> Result<String, String> + Clone + 'static,
{
    let reader = crate::includes::FsIncludeReader;
    process_document_with(source, None, mappings, resolver, read_yaml, &reader)
}

/// Process a xacro `source` with a CALLER-SUPPLIED include reader (the wasm /
/// hermetic entry point). `current_file` is the source's path for relative
/// include resolution (or `None`). All other arguments match [`process_document`],
/// including the note there that collected warnings are not surfaced.
pub fn process_document_with<R>(
    source: &str,
    current_file: Option<&str>,
    mappings: HashMap<String, String>,
    resolver: Box<dyn PackageResolver>,
    read_yaml: R,
    reader: &dyn IncludeReader,
) -> Result<String, ProcessError>
where
    R: Fn(&str) -> Result<String, String> + Clone + 'static,
{
    // Parse the FULL document node list (not just the root element) so any
    // document-level comments flanking the root (e.g. a license banner before
    // `<robot>`) survive into the output, exactly as canonical's minidom Document
    // preserves them. `parse_all` returns the prolog-less top-level nodes
    // (comments + the single root element) in order.
    let mut doc_nodes =
        Element::parse_all(source.as_bytes()).map_err(|e| ProcessError::Parse(format!("{e}")))?;
    // Locate the single root element among the top-level nodes.
    let root_idx = doc_nodes
        .iter()
        .position(|n| matches!(n, XMLNode::Element(_)))
        .ok_or_else(|| ProcessError::Parse("document has no root element".to_owned()))?;
    let mut root = match std::mem::replace(&mut doc_nodes[root_idx], XMLNode::Text(String::new())) {
        XMLNode::Element(e) => e,
        // Just matched Element above; unreachable in practice.
        _ => return Err(ProcessError::Parse("root element vanished".to_owned())),
    };

    // Build the property environment + substitution context.
    let mut tables = PropertyTables::new();
    let mut ctx = SubstContext::new(resolver);
    ctx.args = mappings;
    ctx.filename = current_file.map(str::to_owned);
    tables.set_subst_context(ctx);

    // Register the load_yaml seam (bare name, matching canonical's direct
    // exposure).
    let mut functions = Namespace::new();
    register_load_yaml_from_source(&mut functions, "load_yaml", read_yaml);
    tables.set_functions(functions);

    let top_prop = tables.top_scope();
    let mut pctx = Ctx {
        macros: MacroTables::new(top_prop),
        blocks: HashMap::new(),
        reader,
        filestack: vec![current_file.map(str::to_owned)],
        scope_parents: HashMap::new(),
    };

    let scopes = Scopes {
        prop: top_prop,
        mac: pctx.macros.top_scope(),
    };
    eval_all(&mut root, &mut tables, scopes, &mut pctx)?;

    // Put the processed root back into its document position, then serialize via
    // the faithful `fixed_writexml` port (xml header + autogenerated banner with
    // the source filename + 2-space indent + alphabetically-sorted attributes +
    // single-text-node collapse + whitespace-only-text skip), so the output is
    // BYTE-comparable to canonical `xacro` (which serializes with the monkey-
    // patched `toprettyxml`). `current_file` supplies the banner filename.
    doc_nodes[root_idx] = XMLNode::Element(root);
    Ok(crate::serialize::to_canonical_string(&doc_nodes, current_file))
}

/// Port of `eval_all` for one element: evaluate its attributes, then walk its
/// children, handling `xacro:*` elements + macro calls and evaluating
/// text/comments. Mutates `elt` in place.
///
/// `scopes` bundles the PROPERTY scope and the parallel MACRO scope (they move in
/// lockstep: a namespaced include / macro call descends both).
fn eval_all(
    elt: &mut Element,
    tables: &mut PropertyTables,
    scopes: Scopes,
    ctx: &mut Ctx,
) -> Result<(), ProcessError> {
    // (1) evaluate attributes: drop `xacro:*` attrs, eval_text the rest. Also
    // strip the xacro namespace declaration from THIS element's namespace map.
    eval_attributes(elt, tables, scopes.prop)?;
    strip_xacro_namespace(elt);

    // (2) walk children, rebuilding the child vector. `pending_ns` accumulates
    // `xmlns:*` declarations lifted from any included file / expanded macro body
    // root (canonical's `import_xml_namespaces(elt.parentNode, ...)`), to be merged
    // into THIS element (the parent) after the walk.
    let children = std::mem::take(&mut elt.children);
    let mut out: Vec<XMLNode> = Vec::with_capacity(children.len());
    let mut pending_ns = xmltree::Namespace::empty();
    let mut eval_comments = false;

    for node in children {
        match node {
            XMLNode::Element(child) => {
                eval_comments = false; // any tag disables comment evaluation
                handle_child_element(child, tables, scopes, ctx, &mut out, &mut pending_ns)?;
            }
            XMLNode::Text(text) => {
                let evaluated = eval_text_str(tables, scopes.prop, &text)?;
                if !evaluated.trim().is_empty() {
                    eval_comments = false;
                }
                out.push(XMLNode::Text(evaluated));
            }
            XMLNode::Comment(data) => {
                if data.contains("xacro:eval-comments") {
                    eval_comments = !data.contains("xacro:eval-comments:off");
                    // drop this pragma comment
                } else if eval_comments {
                    let evaluated = eval_text_str(tables, scopes.prop, &data)?;
                    out.push(XMLNode::Comment(evaluated));
                } else {
                    out.push(XMLNode::Comment(data));
                }
            }
            other => out.push(other),
        }
    }

    elt.children = out;
    // Merge the lifted `xmlns:*` declarations into this (parent) element's
    // namespace map, warning on inconsistent redefinition, canonical
    // `import_xml_namespaces`.
    merge_pending_namespaces(elt, &pending_ns);
    Ok(())
}

/// Merge `pending` `xmlns:*` declarations into `elt`'s namespace map, a faithful
/// port of canonical `import_xml_namespaces`: for each `xmlns:*` prefix, if the
/// parent already declares it with a DIFFERENT uri, leave the existing one (a
/// warning case in canonical); otherwise add it. The default namespace (`xmlns`)
/// and the xacro namespace are NOT imported (canonical only lifts `xmlns:*`, and
/// the included root's `xmlns:xacro` is already stripped before this runs).
fn merge_pending_namespaces(elt: &mut Element, pending: &xmltree::Namespace) {
    if pending.0.is_empty() {
        return;
    }
    let target = elt.namespaces.get_or_insert_with(xmltree::Namespace::empty);
    for (prefix, uri) in &pending.0 {
        if prefix.is_empty()
            || prefix == XACRO_PREFIX
            || uri == XACRO_NS_URI
            || prefix == "xmlns"
            || prefix == "xml"
        {
            continue;
        }
        match target.0.get(prefix) {
            // Already present with the SAME uri, or absent: (re)insert (idempotent).
            Some(existing) if existing == uri => {}
            Some(_) => { /* inconsistent redefinition: keep existing (canonical warns) */ }
            None => {
                target.0.insert(prefix.clone(), uri.clone());
            }
        }
    }
}

/// Collect the `xmlns:*` declarations from a (already xacro-stripped) root element
/// into `pending`, so the parent assembling the splice can lift them, canonical's
/// `import_xml_namespaces(parent, root.attributes)`.
fn collect_namespaces_to_import(root: &Element, pending: &mut xmltree::Namespace) {
    if let Some(ns) = root.namespaces.as_ref() {
        for (prefix, uri) in &ns.0 {
            if prefix.is_empty()
                || prefix == XACRO_PREFIX
                || uri == XACRO_NS_URI
                || prefix == "xmlns"
                || prefix == "xml"
            {
                continue;
            }
            pending.0.entry(prefix.clone()).or_insert_with(|| uri.clone());
        }
    }
}

/// Dispatch one child element: a `xacro:*` control/macro-def element, a macro
/// CALL (`<ns:name>`/`<xacro:call>`), or a plain element to recurse into.
fn handle_child_element(
    mut child: Element,
    tables: &mut PropertyTables,
    scopes: Scopes,
    ctx: &mut Ctx,
    out: &mut Vec<XMLNode>,
    pending_ns: &mut xmltree::Namespace,
) -> Result<(), ProcessError> {
    if is_xacro(&child) {
        return handle_xacro_element(child, tables, scopes, ctx, out, pending_ns);
    }
    // A plain (non-`xacro:`-prefixed) element is never a macro call in canonical
    // xacro; only `xacro:`-prefixed tags are. Recurse normally.
    eval_all(&mut child, tables, scopes, ctx)?;
    out.push(XMLNode::Element(child));
    Ok(())
}

/// Handle one `xacro:*` element (prefix `xacro:`), pushing its expansion (if any)
/// onto `out`. Besides the control elements, a `xacro:NAME` whose `NAME` is not a
/// known control keyword is a MACRO CALL.
fn handle_xacro_element(
    mut elt: Element,
    tables: &mut PropertyTables,
    scopes: Scopes,
    ctx: &mut Ctx,
    out: &mut Vec<XMLNode>,
    pending_ns: &mut xmltree::Namespace,
) -> Result<(), ProcessError> {
    match elt.name.as_str() {
        "property" => {
            remove_previous_comments(out);
            grab_property(&elt, tables, scopes.prop)?;
        }
        "arg" => {
            remove_previous_comments(out);
            grab_arg(&elt, tables, scopes.prop)?;
        }
        "macro" => {
            remove_previous_comments(out);
            grab_macro(&elt, ctx, scopes.mac)?;
        }
        "include" => {
            remove_previous_comments(out);
            handle_include(&elt, tables, scopes, ctx, out, pending_ns)?;
        }
        "insert_block" => {
            handle_insert_block(&elt, tables, scopes, ctx, out)?;
        }
        "if" | "unless" => {
            remove_previous_comments(out);
            let raw_cond = elt
                .attributes
                .get("value")
                .cloned()
                .ok_or_else(|| missing_attr("value", &elt.name))?;
            let value = tables.eval_text_in(scopes.prop, &raw_cond)?;
            let mut keep = get_boolean_value(&value, &raw_cond)?;
            if elt.name == "unless" {
                keep = !keep;
            }
            if keep {
                eval_all(&mut elt, tables, scopes, ctx)?;
                out.extend(std::mem::take(&mut elt.children));
            }
        }
        "call" => {
            // Dynamic dispatch: resolve the `macro` attribute to a name, then
            // expand as a macro call.
            handle_dynamic_macro_call(elt, tables, scopes, ctx, out, pending_ns)?;
        }
        // Any other `xacro:NAME` is a MACRO CALL.
        _ => {
            let macro_name = elt.name.clone();
            handle_macro_call(&macro_name, &mut elt, tables, scopes, ctx, out, pending_ns)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------

/// Port of `grab_macro`: parse the `<xacro:macro name=.. params=..>` element into
/// a [`Macro`] and register it in the current macro scope. The body is the
/// element itself (its children form the macro body); it is stored owned for
/// per-call deep-cloning.
fn grab_macro(elt: &Element, ctx: &mut Ctx, mscope: MacroScopeId) -> Result<(), ProcessError> {
    let mut name = elt
        .attributes
        .get("name")
        .map(String::as_str)
        .ok_or_else(|| missing_attr("name", "macro"))?
        .to_owned();
    let params = elt
        .attributes
        .get("params")
        .map(String::as_str)
        .unwrap_or("");

    if name == "call" {
        return Err(runtime("Invalid use of macro name 'call'"));
    }
    if name.contains('.') {
        return Err(runtime(format!(
            "macro names must not contain '.' (reserved for namespaces): {name}"
        )));
    }
    if let Some(stripped) = name.strip_prefix("xacro:") {
        // canonical warns + drops the prefix.
        name = stripped.to_owned();
    }

    let (param_names, defaultmap) = parse_params(params);
    let m = Macro {
        // store the macro body element (children are the body).
        body: elt.clone(),
        params: param_names,
        defaultmap,
    };
    ctx.macros.define(mscope, &name, m);
    Ok(())
}

/// Port of `handle_dynamic_macro_call`: `<xacro:call macro="NAME" .../>`, resolve
/// the `macro` attribute (which may itself be an expression), remove it, then
/// expand the node as if it were `<xacro:NAME .../>`.
fn handle_dynamic_macro_call(
    mut elt: Element,
    tables: &mut PropertyTables,
    scopes: Scopes,
    ctx: &mut Ctx,
    out: &mut Vec<XMLNode>,
    pending_ns: &mut xmltree::Namespace,
) -> Result<(), ProcessError> {
    let raw = elt
        .attributes
        .get("macro")
        .cloned()
        .ok_or_else(|| runtime("xacro:call is missing the 'macro' attribute"))?;
    let name = tables.eval_text_in(scopes.prop, &raw)?.to_python_str();
    if name.is_empty() {
        return Err(runtime("xacro:call is missing the 'macro' attribute"));
    }
    elt.attributes.shift_remove("macro");
    handle_macro_call(&name, &mut elt, tables, scopes, ctx, out, pending_ns)
}

/// Port of `handle_macro_call`: resolve `name`, bind params (keyword attributes +
/// `^`-forwarding + defaults) and block params (`*`/`**` from the call's child
/// elements), expand the cloned body in a fresh macro-body scope, and splice it.
fn handle_macro_call(
    name: &str,
    node: &mut Element,
    tables: &mut PropertyTables,
    scopes: Scopes,
    ctx: &mut Ctx,
    out: &mut Vec<XMLNode>,
    pending_ns: &mut xmltree::Namespace,
) -> Result<(), ProcessError> {
    // `resolve_macro` returns the macro PLUS its DEFINING scopes: both the macro
    // scope (`scoped_macros`) and the parallel property scope (`scoped_symbols`).
    // For a plain top-level macro these are the CALLER's own scopes; for a
    // namespaced macro they are the namespace's macro + property scopes.
    let (m, macro_def_scope, prop_def_scope) = ctx
        .macros
        .resolve(scopes.mac, scopes.prop, name)
        .ok_or_else(|| runtime(format!("unknown macro name: xacro:{name}")))?;

    // A fresh local property scope for macro evaluation: `scoped_symbols =
    // Table(scoped_symbols)`. Canonical scopes the new symbol table under the
    // macro's DEFINING property scope (`prop_def_scope`), so a namespaced macro's
    // body sees its OWN namespace's properties and a plain macro sees the caller's.
    let body_scope = tables.push_scope(prop_def_scope, false);
    ctx.scope_parents.insert(body_scope, Some(prop_def_scope));

    let mut remaining: Vec<String> = m.params.clone();

    // (1) keyword attributes -> bind in the body scope, evaluated in the CALLER's
    // scope. Each consumes a matching param. An unknown attribute is an error.
    let attr_pairs: Vec<(String, String)> = node
        .attributes
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for (attr, raw) in attr_pairs {
        // xmlns:* declarations on the call node are not params.
        if attr.starts_with("xmlns:") || attr == "xmlns" {
            continue;
        }
        if !remaining.iter().any(|p| p == &attr) {
            return Err(runtime(format!("Invalid parameter \"{attr}\"")));
        }
        remaining.retain(|p| p != &attr);
        let value = tables.eval_text_in(scopes.prop, &raw)?;
        tables.set_property_value(body_scope, &attr, value);
    }

    // (2) evaluate the call node's children IN THE CALLER's scope first
    // (`eval_all(node, macros, symbols)`) so block params are pre-expanded, then
    // capture the child ELEMENTS in order for `*`/`**` binding.
    eval_all(node, tables, scopes, ctx)?;
    let child_elements: Vec<Element> = node
        .children
        .iter()
        .filter_map(|n| match n {
            XMLNode::Element(e) => Some(e.clone()),
            _ => None,
        })
        .collect();
    let mut block_idx = 0;

    // (3) bind block params (those whose name starts with `*`), in order, from the
    // captured child elements.
    let block_params: Vec<String> = remaining
        .iter()
        .filter(|p| p.starts_with('*'))
        .cloned()
        .collect();
    for param in block_params {
        if block_idx >= child_elements.len() {
            return Err(runtime(format!("Not enough blocks (macro {name})")));
        }
        remaining.retain(|p| p != &param);
        let block = child_elements[block_idx].clone();
        block_idx += 1;
        bind_block(ctx, body_scope, &param, block);
    }
    // Unused block: canonical walks `block = first_child_element` and advances it
    // once per declared `*`/`**` param; if `block is not None` afterwards (an
    // unconsumed child element remains), it raises "Unused block". This fires both
    // when more child elements than block params exist AND when a macro with NO
    // block params is given child elements. Replicate exactly: any leftover child
    // element past the consumed block params is an error.
    if block_idx < child_elements.len() {
        let leftover = &child_elements[block_idx];
        return Err(runtime(format!("Unused block \"{}\"", leftover.name)));
    }

    // (4) defaults / forwarding for remaining non-block params.
    let still: Vec<String> = remaining.clone();
    for param in still {
        if param.starts_with('*') {
            continue;
        }
        if let Some((forward, default)) = m.defaultmap.get(&param).cloned() {
            let value = eval_default_arg(&forward, default.as_deref(), tables, scopes.prop)?;
            tables.set_property_value(body_scope, &param, value);
            remaining.retain(|p| p != &param);
        }
    }

    // (5) any still-unbound params are undefined.
    if !remaining.is_empty() {
        return Err(runtime(format!(
            "Undefined parameters [{}]",
            remaining.join(",")
        )));
    }

    // (6) expand the cloned body in the body property scope + a FRESH child macro
    // scope (`scoped_macros = Table(scoped_macros)`), then splice its children in
    // place of the call node. The fresh child macro scope CONFINES any
    // `<xacro:macro>` grabbed during body expansion to this call (it is discarded
    // afterwards), while sibling-macro lookup still walks up to `macro_def_scope`.
    let body_mac = ctx.macros.push_child(macro_def_scope, body_scope);
    let body_scopes = Scopes {
        prop: body_scope,
        mac: body_mac,
    };
    let mut body = m.body.clone();
    eval_all(&mut body, tables, body_scopes, ctx)?;

    remove_previous_comments(out);
    // Lift the expanded body root's `xmlns:*` declarations to the call's parent,
    // canonical `import_xml_namespaces(node.parentNode, body.attributes)`.
    collect_namespaces_to_import(&body, pending_ns);
    out.extend(std::mem::take(&mut body.children));
    Ok(())
}

/// `eval_default_arg(forward_variable, default, symbols, macro)`: if no forward,
/// `eval_text(default)`; else read `symbols[forward]`, falling back to the default
/// if absent (and erroring if neither is available).
fn eval_default_arg(
    forward: &Option<String>,
    default: Option<&str>,
    tables: &mut PropertyTables,
    scope: ScopeId,
) -> Result<XacroValue, ProcessError> {
    match forward {
        None => {
            let d = default.unwrap_or("");
            Ok(tables.eval_text_in(scope, d)?)
        }
        Some(var) => match tables.get(scope, var)? {
            Some(v) => Ok(v),
            None => match default {
                Some(d) => Ok(tables.eval_text_in(scope, d)?),
                None => Err(runtime(format!("Undefined property to forward: {var}"))),
            },
        },
    }
}

/// Bind a block param `param` (with its leading `*`/`**` sigil) to `block` in the
/// body scope's block store. A `**name` captures the block's children
/// (content_only=True); a `*name` captures the element itself.
fn bind_block(ctx: &mut Ctx, body_scope: ScopeId, param: &str, block: Element) {
    let (key, content_only) = if let Some(stripped) = param.strip_prefix("**") {
        (stripped.to_owned(), true)
    } else {
        let stripped = param.strip_prefix('*').unwrap_or(param);
        (stripped.to_owned(), false)
    };
    ctx.blocks.insert(
        (body_scope, key),
        BlockBinding {
            element: block,
            content_only,
        },
    );
}

/// Port of the `xacro:insert_block` branch of `eval_all`: look up the bound block
/// (`*name` single, `**name` multi, by checking the block store up the scope
/// chain), clone it, recursively eval it, and splice.
fn handle_insert_block(
    elt: &Element,
    tables: &mut PropertyTables,
    scopes: Scopes,
    ctx: &mut Ctx,
    out: &mut Vec<XMLNode>,
) -> Result<(), ProcessError> {
    let raw_name = elt
        .attributes
        .get("name")
        .ok_or_else(|| missing_attr("name", "insert_block"))?;
    let name = tables.eval_text_in(scopes.prop, raw_name)?.to_python_str();

    let binding = lookup_block(ctx, scopes.prop, &name)
        .ok_or_else(|| runtime(format!("Undefined block \"{name}\"")))?;

    let mut block = binding.element.clone();
    if binding.content_only {
        // `**` block: eval the wrapper and splice its children.
        eval_all(&mut block, tables, scopes, ctx)?;
        out.extend(std::mem::take(&mut block.children));
    } else {
        // `*` block: eval the element and splice the element itself.
        eval_all(&mut block, tables, scopes, ctx)?;
        out.push(XMLNode::Element(block));
    }
    Ok(())
}

/// Look up a bound block by `name`, walking the property-scope parent chain (a
/// block bound in an outer macro body is visible to a nested `insert_block`).
fn lookup_block(ctx: &Ctx, scope: ScopeId, name: &str) -> Option<BlockBinding> {
    let mut cur = Some(scope);
    while let Some(id) = cur {
        if let Some(b) = ctx.blocks.get(&(id, name.to_owned())) {
            return Some(b.clone());
        }
        cur = tables_parent(ctx, id);
    }
    None
}

/// The property-scope parent of `id`, as recorded by [`PropertyTables`]. The block
/// store keys on [`ScopeId`], so block lookup must walk the SAME parent chain the
/// property model uses. [`PropertyTables`] does not expose parent links, so the
/// macro-body scope is created as a direct child of the call scope and block
/// lookup only needs the immediate chain we built; we record parents in `Ctx`.
fn tables_parent(ctx: &Ctx, id: ScopeId) -> Option<ScopeId> {
    ctx.scope_parents.get(&id).copied().flatten()
}

// ---------------------------------------------------------------------------
// Includes
// ---------------------------------------------------------------------------

/// Port of `process_include`: resolve the filename (eval_text + relative + glob),
/// read via the injected reader, parse + recurse `eval_all`, honoring
/// `ns=`/`optional=`/xmlns import, and splice the included children.
fn handle_include(
    elt: &Element,
    tables: &mut PropertyTables,
    scopes: Scopes,
    ctx: &mut Ctx,
    out: &mut Vec<XMLNode>,
    pending_ns: &mut xmltree::Namespace,
) -> Result<(), ProcessError> {
    let raw_filename = elt
        .attributes
        .get("filename")
        .ok_or_else(|| missing_attr("filename", "include"))?;
    let filename_spec = tables.eval_text_in(scopes.prop, raw_filename)?.to_python_str();

    // Optional (swallow read error).
    let optional = match elt.attributes.get("optional") {
        Some(raw) => {
            let v = tables.eval_text_in(scopes.prop, raw)?;
            get_boolean_value(&v, raw)?
        }
        None => false,
    };

    // Namespace: a namespaced include creates a child macro + symbol scope.
    let child_scopes = match elt.attributes.get("ns") {
        Some(raw_ns) => {
            let ns = tables.eval_text_in(scopes.prop, raw_ns)?.to_python_str();
            let sym = tables.push_namespace(scopes.prop, &ns);
            ctx.scope_parents.insert(sym, Some(scopes.prop));
            // Pair the macro namespace scope with its parallel property scope, so
            // `resolve_macro` can return both `scoped_macros` and `scoped_symbols`
            // for a namespaced macro (canonical's lockstep `macros[ns]`/`symbols[ns]`).
            let mac = ctx.macros.namespace(scopes.mac, &ns, sym);
            Scopes { prop: sym, mac }
        }
        None => scopes,
    };

    let current = ctx.filestack.last().cloned().flatten();
    let files = get_include_files(&filename_spec, current.as_deref(), ctx.reader);
    if files.is_empty() {
        // canonical only warns ("matched no files"); we mirror (no error).
        return Ok(());
    }

    for filename in files {
        // Cycle detection: a file already on the stack would recurse forever.
        if ctx
            .filestack
            .iter()
            .any(|f| f.as_deref() == Some(filename.as_str()))
        {
            return Err(runtime(format!(
                "circular xacro include detected: {filename}"
            )));
        }

        let source = match ctx.reader.read(&filename) {
            Ok(s) => s,
            Err(e) => {
                if optional {
                    continue; // swallow the I/O error
                }
                return Err(runtime(format!("failed to read include {filename}: {e}")));
            }
        };

        let mut included = Element::parse(source.as_bytes())
            .map_err(|e| ProcessError::Parse(format!("{filename}: {e}")))?;

        // push the filestack + the subst-context filename so relative nested
        // includes + $(dirname) resolve against THIS file.
        ctx.filestack.push(Some(filename.clone()));
        let prev_filename = set_subst_filename(tables, Some(filename.clone()));

        let result = eval_all(&mut included, tables, child_scopes, ctx);

        set_subst_filename(tables, prev_filename);
        ctx.filestack.pop();
        result?;

        // Lift `xmlns:*` declarations from the (now xacro-stripped) included root
        // to the include's parent element (canonical
        // `import_xml_namespaces(elt.parentNode, include.attributes)`), then splice
        // the included root's CHILDREN (content_only=True) in place of the include.
        collect_namespaces_to_import(&included, pending_ns);
        out.extend(std::mem::take(&mut included.children));
    }
    Ok(())
}

/// Set the subst-context filename (for `$(dirname)`), returning the previous one
/// so the caller can restore it after the nested include completes.
fn set_subst_filename(tables: &mut PropertyTables, filename: Option<String>) -> Option<String> {
    match tables.subst_context_mut() {
        Some(ctx) => std::mem::replace(&mut ctx.filename, filename),
        None => None,
    }
}

// ---------------------------------------------------------------------------
// Comment removal / property / arg (unchanged, with the comment-skip
// guarded port).
// ---------------------------------------------------------------------------

/// Remove the comment node(s) immediately preceding a xacro control node from the
/// already-emitted `out` vector, a faithful port of canonical
/// `remove_previous_comments`.
fn remove_previous_comments(out: &mut Vec<XMLNode>) {
    let mut cut = out.len();
    loop {
        let mut idx = cut;
        if idx > 0 {
            if let XMLNode::Text(t) = &out[idx - 1] {
                if t.chars().all(char::is_whitespace) && t.matches('\n').count() <= 1 {
                    idx -= 1;
                }
            }
        }
        if idx > 0 && matches!(out[idx - 1], XMLNode::Comment(_)) {
            cut = idx - 1;
        } else {
            break;
        }
    }
    out.truncate(cut);
}

/// Port of `grab_property`'s XML-facing core.
fn grab_property(
    elt: &Element,
    tables: &mut PropertyTables,
    scope: ScopeId,
) -> Result<(), ProcessError> {
    let name = elt
        .attributes
        .get("name")
        .map(String::as_str)
        .ok_or_else(|| missing_attr("name", &elt.name))?;
    let def = PropertyDef {
        name,
        value: elt.attributes.get("value").map(String::as_str),
        default: elt.attributes.get("default").map(String::as_str),
        remove: elt.attributes.get("remove").map(String::as_str),
        scope: elt.attributes.get("scope").map(String::as_str),
        lazy_eval: elt.attributes.get("lazy_eval").map(String::as_str),
    };
    tables.set_property(scope, &def)?;
    Ok(())
}

/// Port of the `xacro:arg` branch of `eval_all`.
fn grab_arg(
    elt: &Element,
    tables: &mut PropertyTables,
    scope: ScopeId,
) -> Result<(), ProcessError> {
    let raw_name = elt
        .attributes
        .get("name")
        .ok_or_else(|| missing_attr("name", &elt.name))?;
    let raw_default = elt
        .attributes
        .get("default")
        .ok_or_else(|| missing_attr("default", &elt.name))?;
    let name = tables.eval_text_in(scope, raw_name)?.to_python_str();
    let default = tables.eval_text_in(scope, raw_default)?.to_python_str();

    let ctx = tables
        .subst_context_mut()
        .expect("document pipeline always installs a subst context");
    ctx.args.entry(name).or_insert(default);
    Ok(())
}

/// Evaluate an element's attributes in place.
fn eval_attributes(
    elt: &mut Element,
    tables: &mut PropertyTables,
    scope: ScopeId,
) -> Result<(), ProcessError> {
    let keys: Vec<String> = elt.attributes.keys().cloned().collect();
    let mut new_attrs = xmltree::AttributeMap::<String, String>::new();
    for key in keys {
        if key.starts_with("xacro:") || key == "xmlns:xacro" {
            continue;
        }
        let raw = elt
            .attributes
            .get(&key)
            .expect("key came from this map")
            .clone();
        let evaluated = eval_text_str(tables, scope, &raw)?;
        new_attrs.insert(key, evaluated);
    }
    elt.attributes = new_attrs;
    Ok(())
}

/// `str(eval_text(text, symbols))`.
fn eval_text_str(
    tables: &mut PropertyTables,
    scope: ScopeId,
    text: &str,
) -> Result<String, ProcessError> {
    Ok(tables.eval_text_in(scope, text)?.to_python_str())
}

/// Is `elt` in the xacro namespace (prefix `xacro:`)?
fn is_xacro(elt: &Element) -> bool {
    elt.prefix.as_deref() == Some(XACRO_PREFIX)
}

/// Remove the `xmlns:xacro` declaration from `elt`'s namespace map.
fn strip_xacro_namespace(elt: &mut Element) {
    if let Some(ns) = elt.namespaces.as_mut() {
        ns.0
            .retain(|prefix, uri| prefix.as_str() != XACRO_PREFIX && uri.as_str() != XACRO_NS_URI);
    }
}

/// `get_boolean_value(value, condition)` for the document pipeline.
fn get_boolean_value(value: &XacroValue, condition: &str) -> Result<bool, ProcessError> {
    let err = || {
        ProcessError::Eval(EvalError::Runtime(format!(
            "Xacro conditional \"{}\" evaluated to \"{}\", which is not a boolean expression.",
            condition,
            value.to_python_str()
        )))
    };
    match value {
        XacroValue::Str(s) => match s.as_str() {
            "true" | "True" => Ok(true),
            "false" | "False" => Ok(false),
            other => match other.trim().parse::<i64>() {
                Ok(i) => Ok(i != 0),
                Err(_) => Err(err()),
            },
        },
        XacroValue::Int(i) => Ok(*i != num_bigint::BigInt::from(0)),
        XacroValue::Float(f) => Ok(*f != 0.0),
        XacroValue::Bool(b) => Ok(*b),
        XacroValue::List(items) => Ok(!items.is_empty()),
        XacroValue::Tuple(items) => Ok(!items.is_empty()),
        // `NameSpace` is a `dict` subclass: truthy iff non-empty.
        XacroValue::Dict(map) | XacroValue::Namespace(map) => Ok(!map.is_empty()),
        XacroValue::Null => Ok(false),
    }
}

/// Build the "missing required attribute" error.
fn missing_attr(attr: &str, elt: &str) -> ProcessError {
    ProcessError::Eval(EvalError::Runtime(format!(
        "xacro:{elt} is missing the required '{attr}' attribute"
    )))
}

/// Build a runtime `ProcessError` from a message.
fn runtime(msg: impl Into<String>) -> ProcessError {
    ProcessError::Eval(EvalError::Runtime(msg.into()))
}
