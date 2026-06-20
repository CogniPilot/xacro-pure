//! MACRO expansion cross-check.
//!
//! Two layers (mirroring `tests/pipeline.rs`):
//!   * **Self-contained tests** (always run): expand a SYNTHETIC xacro through
//!     `xacro_pure::process_document_with` (with a hermetic in-test
//!     [`FnIncludeReader`] + fake `$(find)`) and assert the expanded structure.
//!     These exercise value/keyword params, defaults, `^`/`^|` forwarding,
//!     `*block`/`**content` block params, `xacro:insert_block`, namespaced macro
//!     resolution, and `xacro:call` dynamic dispatch.
//!   * **Oracle parity tests** (opt-in, `XACRO_ORACLE=1` + canonical `xacro`):
//!     expand the SAME source through `/opt/ros/jazzy/bin/xacro` and diff the
//!     semantic field stream against the port. This ENFORCES the parity claim.
//!
//! The diff is at the SEMANTIC level (normalized element/attribute tree with
//! numeric tokens canonicalized), not byte-for-byte: the faithful pretty-printer is
//! a separate concern, so the two serializers differ in whitespace/quote-style.

mod common;

use common::{assert_semantic_parity, port_expand, semantic_fields};

#[test]
fn value_and_keyword_params() {
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:macro name="leg" params="side reflect:=1 color:='red'">
    <link name="${side}_leg"><c>${color}</c><v>${reflect * 2}</v></link>
  </xacro:macro>
  <xacro:leg side="left" reflect="-1"/>
  <xacro:leg side="right"/>
</robot>"#;
    let out = port_expand(src);
    let fields = semantic_fields(&out);
    assert!(fields.contains(&("link".to_owned(), "name=left_leg".to_owned())));
    assert!(fields.contains(&("link".to_owned(), "name=right_leg".to_owned())));
    // left: reflect=-1 -> v=-2, color default 'red'; right: reflect=1 -> v=2.
    assert!(out.contains("-2"));
    assert!(out.contains(">2<"));
    assert_eq!(out.matches(">red<").count(), 2);
    assert!(!out.contains("xacro:"));
    assert_semantic_parity(src);
}

#[test]
fn default_expression_param() {
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:property name="base" value="3"/>
  <xacro:macro name="m" params="x:=${base + 1}">
    <link name="L"><v>${x}</v></link>
  </xacro:macro>
  <xacro:m/>
</robot>"#;
    let out = port_expand(src);
    assert!(out.contains(">4<"));
    assert_semantic_parity(src);
}

#[test]
fn forward_caret_pulls_from_caller() {
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:property name="radius" value="0.5"/>
  <xacro:property name="length" value="2.0"/>
  <xacro:macro name="cyl" params="radius:=^ length:=^|1.0 name">
    <link name="${name}"><r>${radius}</r><l>${length}</l></link>
  </xacro:macro>
  <xacro:cyl name="a"/>
</robot>"#;
    let out = port_expand(src);
    assert!(out.contains(">0.5<"));
    assert!(out.contains(">2.0<"));
    assert_semantic_parity(src);
}

#[test]
fn forward_caret_bar_falls_back_to_default() {
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:macro name="cyl" params="length:=^|9.9 name">
    <link name="${name}"><l>${length}</l></link>
  </xacro:macro>
  <xacro:cyl name="a"/>
</robot>"#;
    let out = port_expand(src);
    assert!(out.contains(">9.9<"));
    assert_semantic_parity(src);
}

#[test]
fn star_block_and_double_star_content() {
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:macro name="frame" params="name *origin **props">
    <link name="${name}">
      <xacro:insert_block name="origin"/>
      <extra><xacro:insert_block name="props"/></extra>
    </link>
  </xacro:macro>
  <xacro:frame name="L1">
    <origin xyz="1 2 3"/>
    <props_wrap>
      <mass value="5"/>
      <foo bar="baz"/>
    </props_wrap>
  </xacro:frame>
</robot>"#;
    let out = port_expand(src);
    // *origin inserts the <origin> element itself; **props inserts the wrapper's
    // CHILDREN (mass + foo) into <extra>.
    assert!(out.contains(r#"<origin xyz="1 2 3""#));
    assert!(out.contains(r#"<mass value="5""#));
    assert!(out.contains(r#"<foo bar="baz""#));
    assert!(!out.contains("props_wrap"));
    assert_semantic_parity(src);
}

#[test]
fn insert_block_reused_multiple_times() {
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:macro name="dup" params="*body">
    <a><xacro:insert_block name="body"/></a>
    <b><xacro:insert_block name="body"/></b>
  </xacro:macro>
  <xacro:dup><tag k="v"/></xacro:dup>
</robot>"#;
    let out = port_expand(src);
    assert_eq!(out.matches(r#"<tag k="v""#).count(), 2);
    assert_semantic_parity(src);
}

#[test]
fn nested_macro_calls() {
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:macro name="inner" params="n"><leaf id="${n}"/></xacro:macro>
  <xacro:macro name="outer" params="base">
    <group><xacro:inner n="${base}"/><xacro:inner n="${base + 1}"/></group>
  </xacro:macro>
  <xacro:outer base="10"/>
</robot>"#;
    let out = port_expand(src);
    assert!(out.contains(r#"<leaf id="10""#));
    assert!(out.contains(r#"<leaf id="11""#));
    assert_semantic_parity(src);
}

#[test]
fn dynamic_call_dispatch() {
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:macro name="thing" params="n"><link name="thing_${n}"/></xacro:macro>
  <xacro:property name="which" value="thing"/>
  <xacro:call macro="${which}" n="7"/>
</robot>"#;
    let out = port_expand(src);
    assert!(out.contains(r#"<link name="thing_7""#));
    assert_semantic_parity(src);
}

#[test]
fn macro_conditional_body() {
    // xacro:if inside a macro body (the SO-ARM ros2_control pattern).
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:macro name="ctrl" params="mode">
    <hw>
      <xacro:if value="${mode == 'real'}"><plugin>real</plugin></xacro:if>
      <xacro:if value="${mode == 'mock'}"><plugin>mock</plugin></xacro:if>
    </hw>
  </xacro:macro>
  <xacro:ctrl mode="mock"/>
</robot>"#;
    let out = port_expand(src);
    assert!(out.contains(">mock<"));
    assert!(!out.contains(">real<"));
    assert_semantic_parity(src);
}

#[test]
fn macro_defined_in_body_is_confined() {
    // A `<xacro:macro>` GRABBED while expanding another macro's body lives in a
    // throwaway child macro scope (`scoped_macros = Table(scoped_macros)`) and
    // disappears after the call. Calling `inner` at the top level afterwards must
    // FAIL with "unknown macro"; canonical confines body-defined macros. (The old
    // port leaked `inner` into the outer/global scope and wrongly succeeded.)
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:macro name="outer" params="">
    <xacro:macro name="inner" params=""><inner/></xacro:macro>
    <outer/>
  </xacro:macro>
  <xacro:outer/>
  <xacro:inner/>
</robot>"#;
    let err = common::port_expand_result(src).unwrap_err();
    assert!(
        format!("{err}").contains("unknown macro"),
        "expected body-defined macro to be confined, got: {err}"
    );
    // Oracle parity for the ERROR path: canonical also errors here.
    if common::oracle_errors(src) {
        // both error -> parity on the error case
    } else {
        panic!("oracle did not error on body-defined-macro leak (or oracle disabled is fine)");
    }
}

#[test]
fn macro_in_body_can_be_called_within_same_body() {
    // A body-defined macro IS usable within the SAME body (the throwaway child
    // macro scope is live during body expansion). Here `outer` defines `inner` and
    // immediately calls it.
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:macro name="outer" params="">
    <xacro:macro name="inner" params="k"><leaf v="${k}"/></xacro:macro>
    <wrap><xacro:inner k="9"/></wrap>
  </xacro:macro>
  <xacro:outer/>
</robot>"#;
    let out = port_expand(src);
    assert!(out.contains(r#"<leaf v="9""#));
    assert_semantic_parity(src);
}

#[test]
fn unknown_macro_is_error() {
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:nope/>
</robot>"#;
    let err = common::port_expand_result(src).unwrap_err();
    assert!(format!("{err}").contains("unknown macro"));
}

#[test]
fn invalid_parameter_is_error() {
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:macro name="m" params="a"><x/></xacro:macro>
  <xacro:m a="1" b="2"/>
</robot>"#;
    let err = common::port_expand_result(src).unwrap_err();
    assert!(format!("{err}").contains("Invalid parameter"));
}

#[test]
fn undefined_required_parameter_is_error() {
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:macro name="m" params="a b"><x/></xacro:macro>
  <xacro:m a="1"/>
</robot>"#;
    let err = common::port_expand_result(src).unwrap_err();
    assert!(format!("{err}").contains("Undefined parameters"));
}

#[test]
fn scope_parent_from_macro_body_reaches_caller() {
    // A property defined with `scope="parent"` inside a (plain) macro body must
    // land in the CALLER's scope, so it is visible after the call. This guards the
    // body-scope parenting change: a plain macro's body scope is still parented on
    // the caller's scope (`prop_def_scope == caller_prop` for a direct hit), so
    // `scope="parent"` retargets to the caller as before.
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:macro name="setter" params="">
    <xacro:property name="leaked" value="LEAKED" scope="parent"/>
  </xacro:macro>
  <xacro:setter/>
  <link name="${leaked}"/>
</robot>"#;
    let out = port_expand(src);
    assert!(out.contains(r#"<link name="LEAKED""#), "scope=parent failed: {out}");
    assert_semantic_parity(src);
}
