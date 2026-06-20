//! The property model, each case cross-checked against canonical
//! `xacro` (/opt/ros/jazzy/bin/xacro) on a minimal `<xacro:property>` + a
//! `<link name="${...}">` probe snippet. The canonical expansion is recorded in
//! the comment beside each expectation (the oracle output of the `.xacro` shown).
//!
//! The behaviors here are the ones every other Rust xacro crate gets wrong:
//! late vs early binding, eager self-reference (scalar AND dict), lazy circular
//! detection (with the EXACT canonical error string), scope global/parent
//! retargeting + force-eager, `default=` only-if-absent, `remove=`,
//! `is_valid_name` rejecting python keywords and `__`-prefixed names, and the
//! `_eval_literal`-at-def typing.

use indexmap::IndexMap;
use xacro_pure::{is_valid_name, EvalError, PropertyDef, PropertyTables, ScopeId, XacroValue};

/// Define a property at `scope` from a [`PropertyDef`] (mirrors a
/// `<xacro:property ...>`), panicking on a fatal error so the happy-path tests
/// read cleanly.
fn def(t: &mut PropertyTables, scope: ScopeId, def: PropertyDef) {
    let name = def.name;
    t.set_property(scope, &def)
        .unwrap_or_else(|e| panic!("def `{name}` failed: {e}"));
}

/// The common shape: `<xacro:property name=N value=V/>` (lazy default) at the
/// top scope.
fn def_value(t: &mut PropertyTables, name: &str, value: &str) {
    let top = t.top_scope();
    def(t, top, PropertyDef::value(name, value));
}

/// `<xacro:property name=N value=V lazy_eval="false"/>` (eager) at the top scope.
fn def_eager(t: &mut PropertyTables, name: &str, value: &str) {
    let top = t.top_scope();
    def(
        t,
        top,
        PropertyDef {
            lazy_eval: Some("false"),
            ..PropertyDef::value(name, value)
        },
    );
}

/// Read a property from the top scope, panicking if unbound.
fn get(t: &mut PropertyTables, name: &str) -> XacroValue {
    let top = t.top_scope();
    t.get(top, name)
        .unwrap_or_else(|e| panic!("get `{name}` failed: {e}"))
        .unwrap_or_else(|| panic!("`{name}` is unbound"))
}

// ---------------------------------------------------------------------------
// (A) lazy = LATE binding.
// ---------------------------------------------------------------------------

#[test]
fn lazy_late_binding_uses_redefined_base() {
    // late.xacro (oracle): probe_200
    //   <xacro:property name="base" value="${10}"/>
    //   <xacro:property name="derived" value="${base*2}"/>
    //   <xacro:property name="base" value="${100}"/>
    //   <link name="probe_${derived}"/>
    // derived is LAZY (default), so it resolves at access time against the
    // REDEFINED base=100 -> 200, NOT the def-time base=10.
    let mut t = PropertyTables::new();
    def_value(&mut t, "base", "${10}");
    def_value(&mut t, "derived", "${base*2}");
    def_value(&mut t, "base", "${100}");
    assert_eq!(get(&mut t, "derived"), XacroValue::int(200));
}

// ---------------------------------------------------------------------------
// (B) eager = EARLY binding, frozen at def.
// ---------------------------------------------------------------------------

#[test]
fn eager_early_binding_frozen_at_def() {
    // eager.xacro (oracle): probe_20
    //   derived has lazy_eval="false" -> evaluated NOW against base=10 -> 20,
    //   and a later base=100 does NOT change it.
    let mut t = PropertyTables::new();
    def_value(&mut t, "base", "${10}");
    def_eager(&mut t, "derived", "${base*2}");
    def_value(&mut t, "base", "${100}");
    assert_eq!(get(&mut t, "derived"), XacroValue::int(20));
}

// ---------------------------------------------------------------------------
// (C) eager SELF-REFERENCE: scalar and dict.
// ---------------------------------------------------------------------------

#[test]
fn eager_self_reference_scalar() {
    // selfref.xacro (oracle): probe_15
    //   <xacro:property name="prop" value="${10}" lazy_eval="false"/>
    //   <xacro:property name="prop" value="${prop+5}" lazy_eval="false"/>
    // The second eager def reads the PRIOR binding (10) at def time -> 15.
    let mut t = PropertyTables::new();
    def_eager(&mut t, "prop", "${10}");
    def_eager(&mut t, "prop", "${prop+5}");
    assert_eq!(get(&mut t, "prop"), XacroValue::int(15));
}

#[test]
fn eager_self_reference_dict() {
    // selfdict.xacro (oracle): probe_101.0
    //   <xacro:property name="d" value="${dict(x=1.0)}" lazy_eval="false"/>
    //   <xacro:property name="d" value="${dict(x=d['x']+100.0)}" lazy_eval="false"/>
    //   <link name="probe_${d['x']}"/>  -> 101.0
    // This only works because the typed-single-result rule stores a REAL dict
    // (so d['x'] reads the prior 1.0, not a stringified value).
    let mut t = PropertyTables::new();
    def_eager(&mut t, "d", "${dict(x=1.0)}");
    def_eager(&mut t, "d", "${dict(x=d['x']+100.0)}");

    let mut expected = IndexMap::new();
    expected.insert("x".to_owned(), XacroValue::Float(101.0));
    assert_eq!(get(&mut t, "d"), XacroValue::Dict(expected));

    // And the probe `${d['x']}` -> 101.0 (a typed float, since it is the single
    // segment).
    let top = t.top_scope();
    let probe = eval_text_at(&mut t, top, "${d['x']}");
    assert_eq!(probe, XacroValue::Float(101.0));
}

// ---------------------------------------------------------------------------
// lazy circular -> the EXACT canonical error string.
// ---------------------------------------------------------------------------

#[test]
fn lazy_circular_definition_exact_error() {
    // circular.xacro (oracle stderr):
    //   error: circular variable definition: a -> b -> a
    //   <xacro:property name="a" value="${b}"/>
    //   <xacro:property name="b" value="${a}"/>
    //   <link name="probe_${a}"/>
    let mut t = PropertyTables::new();
    def_value(&mut t, "a", "${b}");
    def_value(&mut t, "b", "${a}");

    let top = t.top_scope();
    let err = t
        .get(top, "a")
        .expect_err("circular definition must error");
    let msg = match err {
        EvalError::Runtime(m) => m,
        other => panic!("expected Runtime error, got {other:?}"),
    };
    // EXACT canonical text: the chain `a -> b -> a` plus the lazy_eval hint.
    assert_eq!(
        msg,
        "circular variable definition: a -> b -> a\n\
         Consider disabling lazy evaluation via lazy_eval=\"false\""
    );
}

// ---------------------------------------------------------------------------
// SCOPES: global / parent retarget + force-eager.
// ---------------------------------------------------------------------------

#[test]
fn scope_global_retargets_to_top_and_forces_eager() {
    // scope_global.xacro (oracle): inner_42 then outer_42
    //   inside a macro: <xacro:property name="g" value="${42}" scope="global"/>
    //   the property is visible at the OUTER (top) scope, not just the macro body.
    let mut t = PropertyTables::new();
    let top = t.top_scope();
    // A macro body is a child scope of top.
    let macro_body = t.push_scope(top, /*is_namespace=*/ false);
    def(
        &mut t,
        macro_body,
        PropertyDef {
            scope: Some("global"),
            ..PropertyDef::value("g", "${42}")
        },
    );

    // Visible at the TOP scope (retargeted there).
    assert_eq!(get(&mut t, "g"), XacroValue::int(42));
    // Also visible from inside the macro body (lookup walks to parent/top).
    assert_eq!(
        t.get(macro_body, "g").unwrap().unwrap(),
        XacroValue::int(42)
    );
}

#[test]
fn scope_parent_retargets_to_caller_and_forces_eager() {
    // scope_parent.xacro (oracle): outer_7
    //   inside a macro: <xacro:property name="p" value="${7}" scope="parent"/>
    //   the property is visible in the CALLER (parent) scope after the call.
    let mut t = PropertyTables::new();
    let top = t.top_scope();
    let macro_body = t.push_scope(top, /*is_namespace=*/ false);
    def(
        &mut t,
        macro_body,
        PropertyDef {
            scope: Some("parent"),
            ..PropertyDef::value("p", "${7}")
        },
    );

    // Visible at the parent (top) scope.
    assert_eq!(get(&mut t, "p"), XacroValue::int(7));
}

#[test]
fn scope_parent_skips_intervening_namespaces_from_macro_body() {
    // A macro body (NOT a NameSpace) whose parent chain is
    //   macro_body -> namespace -> caller(top)
    // with scope="parent" must skip the intervening NameSpace and land in the
    // caller scope (`while isinstance(target, NameSpace): target = target.parent`).
    let mut t = PropertyTables::new();
    let top = t.top_scope();
    let ns = t.push_scope(top, /*is_namespace=*/ true);
    let macro_body = t.push_scope(ns, /*is_namespace=*/ false);
    def(
        &mut t,
        macro_body,
        PropertyDef {
            scope: Some("parent"),
            ..PropertyDef::value("q", "${9}")
        },
    );

    // Landed in `top` (the caller), NOT in the intervening NameSpace `ns`.
    assert_eq!(get(&mut t, "q"), XacroValue::int(9));
    // It is reachable from the NameSpace only by walking up to top, i.e. it is
    // not stored in `ns` itself; confirmed by reading from top directly above.
}

#[test]
fn scope_global_eager_freezes_value() {
    // A global-scope property is FORCED eager: even though we don't pass
    // lazy_eval, it freezes at def time against the current environment.
    let mut t = PropertyTables::new();
    def_value(&mut t, "base", "${5}");
    let top = t.top_scope();
    let macro_body = t.push_scope(top, false);
    def(
        &mut t,
        macro_body,
        PropertyDef {
            scope: Some("global"),
            ..PropertyDef::value("frozen", "${base*10}")
        },
    );
    // Redefine base AFTER: frozen must NOT change (eager forced by scope=global).
    def_value(&mut t, "base", "${1000}");
    assert_eq!(get(&mut t, "frozen"), XacroValue::int(50));
}

// ---------------------------------------------------------------------------
// default= only-if-absent.
// ---------------------------------------------------------------------------

#[test]
fn default_only_if_absent() {
    // default.xacro (oracle): probe_5_88
    //   x=${5}; x default=${999} (IGNORED, x present); y default=${88} (set).
    let mut t = PropertyTables::new();
    def_value(&mut t, "x", "${5}");
    let top = t.top_scope();
    def(
        &mut t,
        top,
        PropertyDef {
            name: "x",
            default: Some("${999}"),
            ..PropertyDef::default()
        },
    );
    def(
        &mut t,
        top,
        PropertyDef {
            name: "y",
            default: Some("${88}"),
            ..PropertyDef::default()
        },
    );
    assert_eq!(get(&mut t, "x"), XacroValue::int(5)); // unchanged
    assert_eq!(get(&mut t, "y"), XacroValue::int(88)); // defaulted
}

// ---------------------------------------------------------------------------
// remove=.
// ---------------------------------------------------------------------------

#[test]
fn remove_then_default_repopulates() {
    // remove.xacro (oracle): probe_77
    //   x=${5}; x remove="true"; x default=${77}  -> 77 (removed, then defaulted)
    let mut t = PropertyTables::new();
    def_value(&mut t, "x", "${5}");
    let top = t.top_scope();
    def(
        &mut t,
        top,
        PropertyDef {
            name: "x",
            remove: Some("true"),
            ..PropertyDef::default()
        },
    );
    assert!(!t.contains(top, "x"), "x must be gone after remove");
    def(
        &mut t,
        top,
        PropertyDef {
            name: "x",
            default: Some("${77}"),
            ..PropertyDef::default()
        },
    );
    assert_eq!(get(&mut t, "x"), XacroValue::int(77));
}

// ---------------------------------------------------------------------------
// is_valid_name: reject python keywords + `__` prefix.
// ---------------------------------------------------------------------------

#[test]
fn is_valid_name_rejects_keyword() {
    // kw.xacro (oracle stderr):
    //   error: Property names must be valid python identifiers: class
    assert!(!is_valid_name("class"));
    assert!(!is_valid_name("for"));
    assert!(!is_valid_name("lambda"));
    // True/False/None parse as Constant, not Name -> rejected too.
    assert!(!is_valid_name("True"));
    assert!(!is_valid_name("None"));
    // Ordinary identifiers are valid.
    assert!(is_valid_name("base_length"));
    assert!(is_valid_name("_private"));
    assert!(is_valid_name("wheel1"));
    // Non-identifiers rejected.
    assert!(!is_valid_name("has space"));
    assert!(!is_valid_name("1leading"));
    assert!(!is_valid_name(""));

    // And the definition path raises the canonical message on a keyword name.
    let mut t = PropertyTables::new();
    let top = t.top_scope();
    let err = t
        .set_property(top, &PropertyDef::value("class", "${5}"))
        .expect_err("keyword name must be rejected");
    assert_eq!(
        err,
        EvalError::Runtime(
            "Property names must be valid python identifiers: class".to_owned()
        )
    );
}

#[test]
fn definition_rejects_double_underscore_prefix() {
    // dunder.xacro (oracle stderr):
    //   error: Property names must not start with double underscore:__x
    let mut t = PropertyTables::new();
    let top = t.top_scope();
    let err = t
        .set_property(top, &PropertyDef::value("__x", "${5}"))
        .expect_err("__ name must be rejected");
    assert_eq!(
        err,
        EvalError::Runtime(
            "Property names must not start with double underscore:__x".to_owned()
        )
    );
}

// ---------------------------------------------------------------------------
// _eval_literal-at-def typing: a literal (non-${...}) value is typed at def.
// ---------------------------------------------------------------------------

#[test]
fn literal_value_typed_at_def() {
    // A bare value (no ${...}) is _eval_literal-coerced:
    //   "42" -> Int, "2.5" -> Float, "true" -> Bool, "1_000" -> Str (PEP515),
    //   "plain" -> Str.
    let mut t = PropertyTables::new();
    def_value(&mut t, "i", "42");
    def_value(&mut t, "f", "2.5");
    def_value(&mut t, "b", "true");
    def_value(&mut t, "u", "1_000");
    def_value(&mut t, "s", "plain");
    assert_eq!(get(&mut t, "i"), XacroValue::int(42));
    assert_eq!(get(&mut t, "f"), XacroValue::Float(2.5));
    assert_eq!(get(&mut t, "b"), XacroValue::Bool(true));
    assert_eq!(get(&mut t, "u"), XacroValue::Str("1_000".to_owned()));
    assert_eq!(get(&mut t, "s"), XacroValue::Str("plain".to_owned()));
}

#[test]
fn single_quoted_literal_lazy_vs_eager_typing() {
    // The DOUBLE-`_eval_literal` subtlety, cross-checked vs canonical xacro
    // (quoted.xacro / quoted2.xacro / quoted3.xacro oracle runs):
    //
    //   LAZY  value="'42'"  -> on access, `_eval_literal(eval_text("42"))`
    //                          re-coerces -> INT 42.   (oracle: q == 42 is True)
    //   EAGER value="'42'" lazy_eval="false" -> a SINGLE `_eval_literal` at def
    //                          strips quotes -> the STRING "42".  (oracle: STR)
    //
    // i.e. a lazy single-quoted number ends up an int, an eager one stays a str.
    let mut t = PropertyTables::new();
    def_value(&mut t, "q_lazy", "'42'");
    def_eager(&mut t, "q_eager", "'42'");
    assert_eq!(get(&mut t, "q_lazy"), XacroValue::int(42));
    assert_eq!(get(&mut t, "q_eager"), XacroValue::Str("42".to_owned()));

    // A lazy single-quoted NON-number stays a string either way.
    def_value(&mut t, "s_lazy", "'hello'");
    assert_eq!(get(&mut t, "s_lazy"), XacroValue::Str("hello".to_owned()));
}

// ---------------------------------------------------------------------------
// eval_text typed-single rule: single ${...} typed, mixed -> str-join.
// ---------------------------------------------------------------------------

#[test]
fn eval_text_typed_single_vs_str_join() {
    // mixed.xacro (oracle): probe_1x2  (a=1, b=2, "${a}x${b}" -> "1x2")
    // floatfmt.xacro (oracle): probe_3.0_0.3333333333333333
    let mut t = PropertyTables::new();
    def_value(&mut t, "a", "${1}");
    def_value(&mut t, "b", "${2}");
    let top = t.top_scope();

    // Single ${...} -> TYPED (an Int, not the string "1").
    assert_eq!(eval_text_at(&mut t, top, "${a}"), XacroValue::int(1));

    // Mixed text -> str-joined per Python str().
    assert_eq!(
        eval_text_at(&mut t, top, "${a}x${b}"),
        XacroValue::Str("1x2".to_owned())
    );

    // Pure literal text (no ${...}) -> the string itself.
    assert_eq!(
        eval_text_at(&mut t, top, "hello"),
        XacroValue::Str("hello".to_owned())
    );

    // A single ${...} that is a dict survives un-stringified (typed-single).
    def_value(&mut t, "d", "${dict(x=1.0)}");
    let mut expected = IndexMap::new();
    expected.insert("x".to_owned(), XacroValue::Float(1.0));
    assert_eq!(eval_text_at(&mut t, top, "${d}"), XacroValue::Dict(expected));
}

#[test]
fn eval_text_dollar_dollar_escape() {
    // `$${x}` de-escapes one `$` and leaves `${x}` as LITERAL text (no eval).
    let mut t = PropertyTables::new();
    def_value(&mut t, "x", "${5}");
    let top = t.top_scope();
    assert_eq!(
        eval_text_at(&mut t, top, "$${x}"),
        XacroValue::Str("${x}".to_owned())
    );
    // A real ${x} still evaluates.
    assert_eq!(eval_text_at(&mut t, top, "${x}"), XacroValue::int(5));
}

// ---------------------------------------------------------------------------
// redefining-global-symbol warning (cross-scope).
// ---------------------------------------------------------------------------

#[test]
fn redefining_global_symbol_warns() {
    // Setting `scope=global` re-establishes a key in the top scope. The
    // "redefining global symbol" warning fires when a key already present in the
    // ROOT is set again. The root holds nothing user-defined by default, so a
    // top-level redefine does NOT warn (matching the oracle's silent probe_6).
    let mut t = PropertyTables::new();
    def_value(&mut t, "x", "${5}");
    def_value(&mut t, "x", "${6}"); // top-level redefine -> NO warning
    assert_eq!(get(&mut t, "x"), XacroValue::int(6));
    assert!(
        t.warnings().is_empty(),
        "top-level redefine must not warn, got {:?}",
        t.warnings()
    );
}

// ---------------------------------------------------------------------------
// registered-function seam reachable from an embedded ${...}.
// ---------------------------------------------------------------------------

#[test]
fn embedded_expr_can_call_registered_function() {
    // A registered Rust fn (the load_yaml / radians seam) is callable from a
    // ${...} that also reads a property, via eval_text_in.
    let mut t = PropertyTables::new();
    t.register_fn_i64("triple", |x| x * 3);
    def_value(&mut t, "n", "${4}");
    let top = t.top_scope();
    // ${triple(n)} -> 12 (property `n`=4 passed to the registered fn).
    assert_eq!(eval_text_at(&mut t, top, "${triple(n)}"), XacroValue::int(12));
}

// ---------------------------------------------------------------------------
// live-globals fidelity: an unreferenced lazy/circular property is NOT forced.
// ---------------------------------------------------------------------------

#[test]
fn unreferenced_circular_property_is_not_forced() {
    // Resolving an expression that does NOT touch a circular property must NOT
    // force it (canonical passes the live Table, so only referenced names get
    // _resolve_d). `c` is circular but `good` evaluates fine.
    let mut t = PropertyTables::new();
    def_value(&mut t, "good", "${1 + 1}");
    def_value(&mut t, "c", "${c}"); // self-circular, but unreferenced below
    let top = t.top_scope();
    assert_eq!(eval_text_at(&mut t, top, "${good}"), XacroValue::int(2));
    // Touching `c` directly still errors (the circular guard fires).
    assert!(t.get(top, "c").is_err());
}

// ---------------------------------------------------------------------------
// get_boolean_value: NON-STRING typed conditions use Python truthiness, not a
// stringified form. Regression tests for the divergence confirmed against the
// live /opt/ros/jazzy/bin/xacro oracle.
// ---------------------------------------------------------------------------

#[test]
fn remove_with_nonintegral_float_condition_is_truthy() {
    // remove_float.xacro (oracle): remove="${1.5}" -> bool(1.5)=True -> x REMOVED
    //   <xacro:property name="x" value="${10}"/>
    //   <xacro:property name="x" remove="${1.5}"/>   (oracle then: name x undefined)
    // The OLD port stringified to "1.5" and errored "not a boolean expression"
    // AND left x in place, a double divergence.
    let mut t = PropertyTables::new();
    def_value(&mut t, "x", "${10}");
    let top = t.top_scope();
    def(
        &mut t,
        top,
        PropertyDef {
            name: "x",
            remove: Some("${1.5}"),
            ..PropertyDef::default()
        },
    );
    assert!(!t.contains(top, "x"), "x must be removed (bool(1.5)=True)");
}

#[test]
fn remove_with_list_condition_truthiness() {
    // remove_list.xacro (oracle): remove="${[1]}" -> bool([1])=True -> REMOVED.
    // remove_empty.xacro (oracle): remove="${[]}" -> bool([])=False -> KEPT.
    let mut t = PropertyTables::new();
    def_value(&mut t, "x", "${10}");
    def_value(&mut t, "y", "${10}");
    let top = t.top_scope();
    def(
        &mut t,
        top,
        PropertyDef {
            name: "x",
            remove: Some("${[1]}"),
            ..PropertyDef::default()
        },
    );
    def(
        &mut t,
        top,
        PropertyDef {
            name: "y",
            remove: Some("${[]}"),
            ..PropertyDef::default()
        },
    );
    assert!(!t.contains(top, "x"), "x removed (bool([1])=True)");
    assert_eq!(get(&mut t, "y"), XacroValue::int(10)); // bool([])=False -> kept
}

#[test]
fn remove_with_none_condition_is_falsy() {
    // remove_none.xacro (oracle): remove="${None}" -> bool(None)=False -> KEPT.
    let mut t = PropertyTables::new();
    def_value(&mut t, "x", "${10}");
    let top = t.top_scope();
    def(
        &mut t,
        top,
        PropertyDef {
            name: "x",
            remove: Some("${None}"),
            ..PropertyDef::default()
        },
    );
    assert_eq!(get(&mut t, "x"), XacroValue::int(10)); // bool(None)=False -> kept
}

#[test]
fn lazy_eval_zero_float_forces_eager() {
    // lazy_float.xacro (oracle): lazy_eval="${0.0}" -> bool(0.0)=False=EAGER ->
    // `d` is DEFINED and frozen at def time = 20 (a later base=100 does NOT change
    // it). The OLD port ERRORED on "0.0" and the property was SILENTLY LOST.
    let mut t = PropertyTables::new();
    def_value(&mut t, "base", "${10}");
    let top = t.top_scope();
    def(
        &mut t,
        top,
        PropertyDef {
            lazy_eval: Some("${0.0}"),
            ..PropertyDef::value("d", "${base*2}")
        },
    );
    def_value(&mut t, "base", "${100}");
    // Defined (not lost) AND frozen eager at 20.
    assert_eq!(get(&mut t, "d"), XacroValue::int(20));
}

#[test]
fn lazy_eval_truthy_float_stays_lazy() {
    // lazy_float2.xacro (oracle): lazy_eval="${1.5}" -> bool(1.5)=True=LAZY ->
    // late binding, uses the REDEFINED base=100 -> 200.
    let mut t = PropertyTables::new();
    def_value(&mut t, "base", "${10}");
    let top = t.top_scope();
    def(
        &mut t,
        top,
        PropertyDef {
            lazy_eval: Some("${1.5}"),
            ..PropertyDef::value("d", "${base*2}")
        },
    );
    def_value(&mut t, "base", "${100}");
    assert_eq!(get(&mut t, "d"), XacroValue::int(200));
}

#[test]
fn boolean_error_embeds_raw_condition_text() {
    // boolerr.xacro (oracle stderr):
    //   Xacro conditional "${'yes'}" evaluated to "yes", which is not a boolean
    //   expression.
    // The condition slot is the ORIGINAL raw attribute text `${'yes'}`; the value
    // slot is the evaluated `yes`. The OLD port put the evaluated value in BOTH.
    let mut t = PropertyTables::new();
    def_value(&mut t, "x", "${10}");
    let top = t.top_scope();
    let err = t
        .set_property(
            top,
            &PropertyDef {
                name: "x",
                remove: Some("${'yes'}"),
                ..PropertyDef::default()
            },
        )
        .expect_err("non-boolean condition must error");
    assert_eq!(
        err,
        EvalError::Runtime(
            "Xacro conditional \"${'yes'}\" evaluated to \"yes\", \
             which is not a boolean expression."
                .to_owned()
        )
    );
}

// ---------------------------------------------------------------------------
// live-globals fidelity: a circular property referenced ONLY in a short-circuited
// / dead branch is NOT forced (canonical resolves a name only when the bytecode
// actually loads it). Confirmed against the oracle.
// ---------------------------------------------------------------------------

#[test]
fn dead_branch_circular_is_not_forced() {
    // deadbranch.xacro (oracle): probe_7
    //   a=${b}, b=${a} (circular), ok=${7}
    //   ${ok if True else a}  -> 7 (the `else a` branch is dead -> a never forced)
    let mut t = PropertyTables::new();
    def_value(&mut t, "a", "${b}");
    def_value(&mut t, "b", "${a}");
    def_value(&mut t, "ok", "${7}");
    let top = t.top_scope();
    assert_eq!(
        eval_text_at(&mut t, top, "${ok if True else a}"),
        XacroValue::int(7)
    );
}

#[test]
fn dead_branch_self_circular_is_not_forced() {
    // deadbranch2.xacro (oracle): probe_1
    //   bad=${bad} (self-circular); ${1 if True else bad} -> 1 (bad never loaded).
    let mut t = PropertyTables::new();
    def_value(&mut t, "bad", "${bad}");
    let top = t.top_scope();
    assert_eq!(
        eval_text_at(&mut t, top, "${1 if True else bad}"),
        XacroValue::int(1)
    );
}

#[test]
fn or_short_circuit_skips_circular() {
    // orshort.xacro (oracle): probe_7
    //   ok=${7}, c=${c} (self-circular); ${ok or c} -> 7 (`or` short-circuits, c
    //   never loaded).
    let mut t = PropertyTables::new();
    def_value(&mut t, "ok", "${7}");
    def_value(&mut t, "c", "${c}");
    let top = t.top_scope();
    assert_eq!(eval_text_at(&mut t, top, "${ok or c}"), XacroValue::int(7));
}

#[test]
fn live_branch_circular_resurfaces_original_error() {
    // The OTHER side of the deferral: when the circular name IS in the LIVE
    // branch, the EXACT circular error must re-surface (not a bare NameError).
    //   a=${b}, b=${a}; ${a if False else 0} -> the `a` (the IF-true operand) is
    //   loaded first (condition False so the true-operand is NOT loaded; careful:
    //   in `X if C else Y`, X is only evaluated if C is truthy). Use a directly-
    //   live reference instead:  ${a + 0}.
    let mut t = PropertyTables::new();
    def_value(&mut t, "a", "${b}");
    def_value(&mut t, "b", "${a}");
    let top = t.top_scope();
    let err = t
        .eval_text_in(top, "${a + 0}")
        .expect_err("a is live -> circular error must surface");
    let msg = match err {
        EvalError::Runtime(m) => m,
        other => panic!("expected Runtime, got {other:?}"),
    };
    assert_eq!(
        msg,
        "circular variable definition: a -> b -> a\n\
         Consider disabling lazy evaluation via lazy_eval=\"false\""
    );
}

// ---------------------------------------------------------------------------
// is_valid_name: Unicode identifiers Python (and the oracle) accept are valid.
// ---------------------------------------------------------------------------

#[test]
fn is_valid_name_accepts_unicode_identifiers() {
    // unicode.xacro / unicode2.xacro (oracle): probe_42 / probe_5 with names
    // `café` / `über`: Python's ast.parse accepts any Unicode identifier, so the
    // ASCII-only port FATAL-rejected a legal canonical xacro file.
    assert!(is_valid_name("café"));
    assert!(is_valid_name("über"));
    assert!(is_valid_name("résumé"));
    assert!(is_valid_name("λ"));
    assert!(is_valid_name("你好"));
    // ASCII identifiers still valid, keywords / non-idents still rejected.
    assert!(is_valid_name("base_length"));
    assert!(!is_valid_name("class"));
    assert!(!is_valid_name("x²")); // superscript-2 is not XID_Continue (oracle rejects)
    assert!(!is_valid_name("5x"));
    assert!(!is_valid_name(""));

    // And the definition path actually DEFINES a unicode-named property.
    let mut t = PropertyTables::new();
    def_value(&mut t, "café", "${42}");
    assert_eq!(get(&mut t, "café"), XacroValue::int(42));
}

// ---------------------------------------------------------------------------
// helper: eval_text against the public API.
// ---------------------------------------------------------------------------

fn eval_text_at(t: &mut PropertyTables, scope: ScopeId, text: &str) -> XacroValue {
    t.eval_text_in(scope, text)
        .unwrap_or_else(|e| panic!("eval_text `{text}` failed: {e}"))
}
