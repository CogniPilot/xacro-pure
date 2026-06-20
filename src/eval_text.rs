//! `eval_text`: the FULL `QuickLexer` + the typed-single-result rule.
//!
//! Faithful port of canonical xacro's `eval_text` (`xacro/__init__.py:700`) and
//! its `LEXER` (`__init__.py:693`). The lexer's regex-precedence table is:
//! ```text
//! DOLLAR_DOLLAR_BRACE = ^\$\$+(\{|\()   # $$ (one or more) then { or (  -> de-escape
//! EXPR                = ^\$\{[^\}]*\}    # ${...}  (body cannot contain })
//! EXTENSION           = ^\$\([^\)]*\)    # $(...)  (body cannot contain ))
//! TEXT                = [^$]+|\$[^{($]+|\$$  # text without $, or $ not opening {/(/$
//! ```
//! Each `${...}` is `handle_expr`'d, each `$(...)` `handle_extension`'d, each
//! DOLLAR_DOLLAR_BRACE yields the match minus one `$` (so `$${x}` -> literal
//! `${x}`), each TEXT is verbatim. Then the **typed-single-result rule**:
//! ```python
//! if len(results) == 1: return results[0]          # single -> typed, AS IS
//! else:                 return ''.join(map(str, results))   # mixed -> str-join
//! ```
//!
//! ## The `$(...)`-resolved-as-TEXT-before-`${...}` rule (canonical, load-bearing)
//! `handle_expr(s)` is `safe_eval(eval_text(s, symbols), symbols)` and
//! `handle_extension(s)` is `eval_extension("$(%s)" % eval_text(s, symbols))`.
//! Both RE-`eval_text` their inner body FIRST. So a `$(arg x)` appearing *inside*
//! a `${...}` python expression is an EXTENSION segment of the recursive
//! `eval_text` and is resolved to TEXT before the surrounding expression is
//! python-evaluated, the OpenArm `${load_yaml(cfg + "/" + '$(arg x)')}` pattern.
//!
//! ## Substitution context
//! `eval_text` threads a [`SubstContext`] (canonical's global
//! `substitution_args_context`) so EXTENSION segments can resolve. When `None`
//! (a pure property-model eval with no document context), an EXTENSION segment is
//! a hard error, matching canonical, where `$(...)` always needs the context.

use std::collections::HashMap;

use crate::error::EvalError;
use crate::eval::{referenced_names, safe_eval_deferred_with_args};
use crate::namespace::Namespace;
use crate::substitution_args::{eval_extension, SubstContext};
use crate::table::{PropertyTables, ScopeId};
use crate::value::XacroValue;

/// One lexed segment of the input text, mirroring the `QuickLexer` token kinds.
enum Segment {
    /// Literal `TEXT` (verbatim) OR a de-escaped `DOLLAR_DOLLAR_BRACE` result
    /// (the match minus one leading `$`). Both end up as literal output text.
    Text(String),
    /// An `EXPR` segment: the inner `${...}` body (without the delimiters).
    Expr(String),
    /// An `EXTENSION` segment: the inner `$(...)` body (without the delimiters).
    Extension(String),
}

/// Evaluate `text`, resolving each `${...}` and `$(...)` against `tables`/`subst`
/// at `scope`, returning a typed [`XacroValue`] per the typed-single-result rule.
///
/// `functions` supplies the registered Rust callables (the `xacro.load_yaml` /
/// `radians` seam). `subst` is the substitution-args context for `$(...)`
/// EXTENSION segments (`None` => an EXTENSION is an error, as in a context-free
/// property eval).
pub fn eval_text(
    text: &str,
    tables: &mut PropertyTables,
    scope: ScopeId,
    functions: &Namespace,
    subst: &mut Option<SubstContext>,
) -> Result<XacroValue, EvalError> {
    let segments = lex(text)?;

    let mut results: Vec<XacroValue> = Vec::with_capacity(segments.len());
    for seg in segments {
        match seg {
            Segment::Text(t) => results.push(XacroValue::Str(t)),
            Segment::Expr(src) => {
                let v = handle_expr(&src, tables, scope, functions, subst)?;
                results.push(v);
            }
            Segment::Extension(src) => {
                let v = handle_extension(&src, tables, scope, functions, subst)?;
                results.push(XacroValue::Str(v));
            }
        }
    }

    // The typed-single-result rule.
    match results.len() {
        0 => Ok(XacroValue::Str(String::new())),
        1 => Ok(results.into_iter().next().expect("len checked == 1")),
        _ => {
            let joined: String = results.iter().map(XacroValue::to_python_str).collect();
            Ok(XacroValue::Str(joined))
        }
    }
}

/// `handle_expr(s)` = `safe_eval(eval_text(s, symbols), symbols)`: recursively
/// `eval_text` the body FIRST (so a nested `$(...)` resolves to text), then
/// python-evaluate the resulting STRING against the live tables.
fn handle_expr(
    body: &str,
    tables: &mut PropertyTables,
    scope: ScopeId,
    functions: &Namespace,
    subst: &mut Option<SubstContext>,
) -> Result<XacroValue, EvalError> {
    // (1) recursively eval_text the body (resolving any inner $(...) / $$) into
    // its string form (the source `safe_eval` will compile). A single nested
    // EXTENSION yields its str; pure expression text yields itself verbatim.
    let inner = eval_text(body, tables, scope, functions, subst)?;
    let expr_src = inner.to_python_str();
    // (2) safe_eval the resulting expression string against the live tables.
    eval_expr(&expr_src, tables, scope, functions, subst)
}

/// `handle_extension(s)` = `eval_extension("$(%s)" % eval_text(s, symbols))`:
/// recursively `eval_text` the body, then resolve the `$(...)` substitution.
fn handle_extension(
    body: &str,
    tables: &mut PropertyTables,
    scope: ScopeId,
    functions: &Namespace,
    subst: &mut Option<SubstContext>,
) -> Result<String, EvalError> {
    let inner = eval_text(body, tables, scope, functions, subst)?;
    let inner_str = inner.to_python_str();
    let ctx = subst.as_mut().ok_or_else(|| {
        EvalError::Runtime(format!(
            "substitution arg $({inner_str}) used without a substitution context"
        ))
    })?;
    Ok(eval_extension(&inner_str, ctx)?)
}

/// Evaluate a single resolved `${...}` expression string against the live tables
/// (the live-globals / deferred-error machinery, unchanged): resolve the
/// names it references (lazy resolution + circular detection in `tables.get`),
/// snapshot them into a flat [`Namespace`] alongside the registered functions,
/// then hand off to `safe_eval`. A name whose resolution ERRORS is omitted and
/// its error DEFERRED so a dead-branch reference is not forced.
fn eval_expr(
    src: &str,
    tables: &mut PropertyTables,
    scope: ScopeId,
    functions: &Namespace,
    subst: &mut Option<SubstContext>,
) -> Result<XacroValue, EvalError> {
    let names = referenced_names(src)?;

    let mut props = Namespace::new();
    let mut deferred_errors: HashMap<String, EvalError> = HashMap::new();
    for name in &names {
        // Resolve via `get_env` (NOT `get`): we are already inside an evaluation
        // whose env was taken out of `tables`, so `tables.get` would re-take the
        // now-empty functions/subst. `get_env` reuses the borrowed env, keeping
        // the registered functions (`load_yaml`) + subst context live for a
        // nested lazy-property resolution.
        match tables.get_env(scope, name, functions, subst) {
            Ok(Some(value)) => {
                props.set(name.clone(), value);
            }
            Ok(None) => {}
            Err(e) => {
                deferred_errors.insert(name.clone(), e);
            }
        }
    }

    // The live substitution-args snapshot, so `xacro.arg(name)` inside the
    // expression resolves against the current arg table (canonical
    // `substitution_args_context['arg']`). Empty when no document context.
    let args_snapshot = subst
        .as_ref()
        .map(|c| c.args.clone())
        .unwrap_or_default();

    safe_eval_deferred_with_args(src, &props, functions, &deferred_errors, &args_snapshot)
}

/// Lex `text` into [`Segment`]s, faithfully reproducing canonical `QuickLexer`,
/// including its **error behavior**.
///
/// At each position the lexer matches the remaining string against the four
/// regexes in precedence order (DOLLAR_DOLLAR_BRACE > EXPR > EXTENSION > TEXT),
/// where:
/// ```text
/// DOLLAR_DOLLAR_BRACE = ^\$\$+(\{|\()   # two or more $ then { or (
/// EXPR                = ^\$\{[^\}]*\}    # ${...}, body has no }
/// EXTENSION           = ^\$\([^\)]*\)    # $(...), body has no )
/// TEXT                = [^$]+|\$[^{($]+|\$$   # text w/o $, OR $ + non-{($ run, OR a lone trailing $
/// ```
/// CRITICALLY, when NONE matches, canonical raises
/// `XacroException('invalid expression: ' + remaining)` (`__init__.py:463`).
/// That happens exactly for a `$` that starts `${`/`$(` which never closes, or a
/// `$$+` run with no following `{`/`(`, or `$$`/`$$x` (a `$` followed by `$` with
/// no brace); NONE of which the TEXT regex can consume. The previous port
/// fabricated literal-text fallbacks for these, silently producing garbage where
/// canonical loudly errors (the exact silent-wrong-output class this port exists
/// to avoid). We now mirror the no-match-raises rule.
///
/// Adjacent literal output (TEXT runs + de-escaped `$$`) is coalesced into one
/// [`Segment::Text`], observationally identical (consecutive TEXT str-joins the
/// same as one) and keeping the typed-single rule correct (a value-bearing
/// segment is never merged with literal text).
fn lex(text: &str) -> Result<Vec<Segment>, EvalError> {
    let chars: Vec<char> = text.chars().collect();
    let mut segments: Vec<Segment> = Vec::new();
    let mut buf = String::new();
    let mut i = 0;
    let n = chars.len();

    macro_rules! flush {
        () => {
            if !buf.is_empty() {
                segments.push(Segment::Text(std::mem::take(&mut buf)));
            }
        };
    }

    /// The remaining string from index `k`, for the `invalid expression` message.
    fn remaining(chars: &[char], k: usize) -> String {
        chars[k..].iter().collect()
    }

    while i < n {
        let c = chars[i];
        if c != '$' {
            // TEXT alt `[^$]+`: consume the run of non-`$` chars.
            buf.push(c);
            i += 1;
            continue;
        }

        // c == '$'. Try the four regexes in precedence order.

        // (1) DOLLAR_DOLLAR_BRACE `^\$\$+(\{|\()`: `$` then one-or-more `$` then
        //     `{`/`(`. i.e. at least TWO `$` followed by a brace.
        if i + 1 < n && chars[i + 1] == '$' {
            let mut j = i;
            while j < n && chars[j] == '$' {
                j += 1;
            }
            let dollars = j - i; // >= 2 here
            if j < n && (chars[j] == '{' || chars[j] == '(') {
                // match = "$$...{" (dollars $ + brace); emit match[1:] as literal
                // (one fewer `$`, then the brace), continue AFTER the brace.
                for _ in 0..dollars - 1 {
                    buf.push('$');
                }
                buf.push(chars[j]);
                i = j + 1;
                continue;
            }
            // A `$$+` run with NO following brace: DOLLAR_DOLLAR_BRACE fails, and
            // TEXT cannot consume a `$` immediately followed by `$` (the `[^{($]`
            // class excludes `$`, and `\$$` needs the `$` to be string-final, but
            // here it is followed by another `$`). So NO regex matches -> canonical
            // raises. (`$$`, `$$x`, `lit$$`, `a$$$` all land here.)
            return Err(EvalError::Runtime(format!(
                "invalid expression: {}",
                remaining(&chars, i)
            )));
        }

        // (2) EXPR `^\$\{[^\}]*\}`: `${` ... `}` with no `}` inside.
        if i + 1 < n && chars[i + 1] == '{' {
            if let Some(close) = find_close(&chars, i + 2, '}') {
                flush!();
                let inner: String = chars[i + 2..close].iter().collect();
                segments.push(Segment::Expr(inner));
                i = close + 1;
                continue;
            }
            // Unterminated `${`: EXPR fails; TEXT cannot consume a `$` followed by
            // `{` (excluded by `[^{($]`, and not string-final). No match -> raise.
            return Err(EvalError::Runtime(format!(
                "invalid expression: {}",
                remaining(&chars, i)
            )));
        }

        // (3) EXTENSION `^\$\([^\)]*\)`: `$(` ... `)` with no `)` inside.
        if i + 1 < n && chars[i + 1] == '(' {
            if let Some(close) = find_close(&chars, i + 2, ')') {
                flush!();
                let inner: String = chars[i + 2..close].iter().collect();
                segments.push(Segment::Extension(inner));
                i = close + 1;
                continue;
            }
            // Unterminated `$(`: EXTENSION fails; TEXT cannot consume `$(`. Raise.
            return Err(EvalError::Runtime(format!(
                "invalid expression: {}",
                remaining(&chars, i)
            )));
        }

        // (4) TEXT for a lone `$`:
        //   * `\$$`:     a `$` at the very end of the string -> literal `$`;
        //   * `\$[^{($]+`: a `$` followed by a run of chars that are NOT `{`,`(`,
        //     `$` -> literal (`$5`, `$x`, `$b`).
        // A `$` immediately followed by `{`/`(`/`$` was already handled (and would
        // have closed-or-raised) above, so here the next char (if any) is a plain
        // char and TEXT always succeeds.
        if i + 1 == n {
            // lone trailing `$` (`\$$`).
            buf.push('$');
            i += 1;
            continue;
        }
        // `\$[^{($]+`: emit the `$` plus the following non-{($ run as literal.
        buf.push('$');
        i += 1;
        while i < n && chars[i] != '{' && chars[i] != '(' && chars[i] != '$' {
            buf.push(chars[i]);
            i += 1;
        }
    }

    flush!();
    Ok(segments)
}

/// Find the index of the first `close` char at or after `start`. `None` if none;
/// mirrors the `[^}]*`/`[^)]*` lexer bodies stopping at the first close.
fn find_close(chars: &[char], start: usize, close: char) -> Option<usize> {
    (start..chars.len()).find(|&k| chars[k] == close)
}
