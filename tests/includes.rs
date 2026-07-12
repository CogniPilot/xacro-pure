//! INCLUDE cross-check.
//!
//! Self-contained tests (always run) over a HERMETIC virtual-FS include reader,
//! plus opt-in canonical-`xacro` oracle parity (`XACRO_ORACLE=1`). Exercises:
//! relative include, multi-file include, glob include, `ns=` namespaced include,
//! `optional=` swallowing a missing file, and the missing-required-file error.

mod common;

use common::{
    assert_semantic_parity, assert_semantic_parity_with_files, port_expand,
    port_expand_result_with_files, port_expand_with_files, semantic_fields,
};

#[test]
fn relative_include_splices_children() {
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:include filename="parts.xacro"/>
  <link name="local"/>
</robot>"#;
    let parts = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <link name="from_include_a"/>
  <link name="from_include_b"/>
</robot>"#;
    let out = port_expand_with_files(main, &[("parts.xacro", parts)]);
    let names: Vec<String> = semantic_fields(&out)
        .into_iter()
        .filter(|(t, _)| t == "link")
        .map(|(_, a)| a)
        .collect();
    assert_eq!(
        names,
        vec![
            "name=from_include_a",
            "name=from_include_b",
            "name=local"
        ]
    );
    assert!(!out.contains("xacro:include"));
    assert_semantic_parity_with_files(main, &[("parts.xacro", parts)]);
}

#[test]
fn included_macro_is_callable() {
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:include filename="lib.xacro"/>
  <xacro:widget n="42"/>
</robot>"#;
    let lib = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="widget" params="n"><link name="w_${n}"/></xacro:macro>
</robot>"#;
    let out = port_expand_with_files(main, &[("lib.xacro", lib)]);
    assert!(out.contains(r#"<link name="w_42""#));
    assert_semantic_parity_with_files(main, &[("lib.xacro", lib)]);
}

#[test]
fn included_property_is_visible() {
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:include filename="props.xacro"/>
  <link name="l_${size}"/>
</robot>"#;
    let props = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="size" value="99"/>
</robot>"#;
    let out = port_expand_with_files(main, &[("props.xacro", props)]);
    assert!(out.contains(r#"<link name="l_99""#));
    assert_semantic_parity_with_files(main, &[("props.xacro", props)]);
}

#[test]
fn glob_include_sorted() {
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:include filename="*.part.xacro"/>
</robot>"#;
    let a = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro"><link name="part_a"/></robot>"#;
    let b = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro"><link name="part_b"/></robot>"#;
    let files = [("a.part.xacro", a), ("b.part.xacro", b)];
    let out = port_expand_with_files(main, &files);
    let names: Vec<String> = semantic_fields(&out)
        .into_iter()
        .filter(|(t, _)| t == "link")
        .map(|(_, attr)| attr)
        .collect();
    assert_eq!(names, vec!["name=part_a", "name=part_b"]);
    assert_semantic_parity_with_files(main, &files);
}

#[test]
fn namespaced_include() {
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:include filename="lib.xacro" ns="mylib"/>
  <xacro:mylib.widget n="1"/>
</robot>"#;
    let lib = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="widget" params="n"><link name="w_${n}"/></xacro:macro>
</robot>"#;
    let out = port_expand_with_files(main, &[("lib.xacro", lib)]);
    assert!(out.contains(r#"<link name="w_1""#));
    assert_semantic_parity_with_files(main, &[("lib.xacro", lib)]);
}

#[test]
fn namespaced_macro_body_sees_namespace_property() {
    // A namespaced macro whose BODY references a property defined in the SAME
    // included file. Canonical scopes the body's symbol table under the macro's
    // DEFINING (namespace) symbol scope, so `${secret}` resolves to the namespace
    // property. (The old port parented the body on the CALLER scope and failed
    // with `NameError: name 'secret' is not defined`.)
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:include filename="lib.xacro" ns="mylib"/>
  <xacro:mylib.emit/>
</robot>"#;
    let lib = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="secret" value="FROM_NS"/>
  <xacro:macro name="emit" params=""><emit v="${secret}"/></xacro:macro>
</robot>"#;
    let out = port_expand_with_files(main, &[("lib.xacro", lib)]);
    assert!(
        out.contains(r#"<emit v="FROM_NS""#),
        "namespaced macro body did not see ns property: {out}"
    );
    assert_semantic_parity_with_files(main, &[("lib.xacro", lib)]);
}

#[test]
fn namespaced_property_dotted_access() {
    // `${ns.prop}` dotted access after a `ns=`-include. Canonical binds
    // `symbols[ns] = NameSpace(...)` so `mylib.size` resolves via attribute access.
    // (The old port ignored the ns NAME and failed with `name 'mylib' is not
    // defined`.)
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:include filename="libsize.xacro" ns="mylib"/>
  <link name="l_${mylib.size}"/>
</robot>"#;
    let lib = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="size" value="99"/>
</robot>"#;
    let out = port_expand_with_files(main, &[("libsize.xacro", lib)]);
    assert!(out.contains(r#"<link name="l_99""#), "ns.prop dotted access failed: {out}");
    assert_semantic_parity_with_files(main, &[("libsize.xacro", lib)]);
}

#[test]
fn namespaced_property_dotted_access_parent_fallback() {
    // `${ns.prop}` where `prop` is NOT in the namespace but in the INCLUDING file:
    // canonical's `NameSpace(parent=symbols)` walks the parent chain, so `L.outer`
    // resolves to the outer property. Verified vs the oracle.
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:property name="outer" value="OUTER"/>
  <xacro:include filename="libfb.xacro" ns="L"/>
  <link name="a_${L.own}"/>
  <link name="b_${L.outer}"/>
</robot>"#;
    let lib = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="own" value="OWN"/>
</robot>"#;
    let out = port_expand_with_files(main, &[("libfb.xacro", lib)]);
    assert!(out.contains(r#"<link name="a_OWN""#), "ns own prop failed: {out}");
    assert!(out.contains(r#"<link name="b_OUTER""#), "ns parent fallback failed: {out}");
    assert_semantic_parity_with_files(main, &[("libfb.xacro", lib)]);
}

#[test]
fn namespaced_macro_calls_sibling_reading_namespace_property() {
    // A namespaced macro `a` calls a SIBLING `b` (also in the namespace); `b`'s
    // body reads a namespace property. Exercises both the macro-table side (sibling
    // resolution within the namespace) and the symbol side (body sees ns prop).
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:include filename="libsib.xacro" ns="L"/>
  <xacro:L.a/>
</robot>"#;
    let lib = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="nsprop" value="SIBPROP"/>
  <xacro:macro name="b" params=""><b v="${nsprop}"/></xacro:macro>
  <xacro:macro name="a" params=""><a/><xacro:b/></xacro:macro>
</robot>"#;
    let out = port_expand_with_files(main, &[("libsib.xacro", lib)]);
    assert!(out.contains(r#"<b v="SIBPROP""#), "ns sibling call failed: {out}");
    assert_semantic_parity_with_files(main, &[("libsib.xacro", lib)]);
}

#[test]
fn optional_missing_include_is_swallowed() {
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:include filename="missing.xacro" optional="true"/>
  <link name="survived"/>
</robot>"#;
    let out = port_expand_with_files(main, &[]);
    assert!(out.contains(r#"<link name="survived""#));
    assert_semantic_parity_with_files(main, &[]);
}

#[test]
fn missing_required_include_is_error() {
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:include filename="missing.xacro"/>
</robot>"#;
    let err = port_expand_result_with_files(main, &[]).unwrap_err();
    assert!(format!("{err}").contains("missing.xacro"));
}

#[test]
fn nested_includes_resolve_relative() {
    // main includes inc/a.xacro, which itself includes b.xacro (relative to inc/).
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:include filename="inc/a.xacro"/>
</robot>"#;
    let a = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="b.xacro"/>
  <link name="from_a"/>
</robot>"#;
    let b = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro"><link name="from_b"/></robot>"#;
    let files = [("inc/a.xacro", a), ("inc/b.xacro", b)];
    let out = port_expand_with_files(main, &files);
    assert!(out.contains(r#"<link name="from_b""#));
    assert!(out.contains(r#"<link name="from_a""#));
    assert_semantic_parity_with_files(main, &files);
}

#[test]
fn filename_from_arg_expression() {
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:arg name="which" default="parts.xacro"/>
  <xacro:include filename="$(arg which)"/>
</robot>"#;
    let parts = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro"><link name="argincl"/></robot>"#;
    let out = port_expand_with_files(main, &[("parts.xacro", parts)]);
    assert!(out.contains(r#"<link name="argincl""#));
    assert_semantic_parity_with_files(main, &[("parts.xacro", parts)]);
}

#[test]
fn include_lifts_xmlns_to_parent() {
    // canonical `import_xml_namespaces(elt.parentNode, include.attributes)`: an
    // included root declaring `xmlns:custom` lifts that declaration onto the
    // INCLUDING element (here <robot>), with `<custom:thing>` underneath. The
    // xacro namespace is NOT lifted (already stripped). Asserts the parent's
    // namespace MAP, and oracle parity (root-namespace-map identical).
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:include filename="libns.xacro"/>
</robot>"#;
    let lib = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" xmlns:custom="http://custom">
  <custom:thing val="1"/>
</robot>"#;
    let files = [("libns.xacro", lib)];
    let out = port_expand_with_files(main, &files);
    let root_ns = common::root_namespace_map(&out);
    assert_eq!(
        root_ns.get("custom").map(String::as_str),
        Some("http://custom"),
        "xmlns:custom not lifted to parent: {out}"
    );
    // xacro namespace must NOT be present on the output root.
    assert!(
        !root_ns.contains_key("xacro"),
        "xacro namespace leaked onto output root: {out}"
    );
    // the <custom:thing> element survives with its attribute.
    assert!(out.contains(r#"val="1""#));
    // Oracle parity: the output root's lifted namespace map matches canonical's.
    common::assert_root_namespaces_parity_with_files(main, &files);
    // And the semantic field stream still matches.
    assert_semantic_parity_with_files(main, &files);
}

#[test]
fn macro_body_lifts_xmlns_to_parent() {
    // canonical `import_xml_namespaces(node.parentNode, body.attributes)`: a macro
    // whose BODY ROOT declares `xmlns:gz` lifts it onto the call's parent on
    // expansion. (xmltree stores the body root's xmlns on the macro element; the
    // expansion's first child is the body content.)
    let src = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:macro name="m" params="">
    <gz:thing xmlns:gz="http://gz" k="v"/>
  </xacro:macro>
  <xacro:m/>
</robot>"#;
    let out = port_expand(src);
    // the gz-prefixed element survives (semantic parity below enforces structure).
    assert!(out.contains(r#"k="v""#), "macro body element lost: {out}");
    assert_semantic_parity(src);
}

#[test]
fn self_include_via_dotdot_is_cycle_error() {
    // A file that includes itself through a byte-different but lexically
    // equivalent path (`sub/../self.xacro`) must be caught as a cycle. Without
    // lexical normalization of the comparison key, the raw strings never match
    // and the include recurses until the stack overflows. The cycle is detected
    // at the first self-reference, well before the depth cap, so this proves the
    // lexical normalization, not the depth backstop.
    let main = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="t">
  <xacro:include filename="sub/../self.xacro"/>
</robot>"#;
    let selfdoc = r#"<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="sub/../self.xacro"/>
</robot>"#;
    let err = port_expand_result_with_files(main, &[("sub/../self.xacro", selfdoc)]).unwrap_err();
    assert!(
        format!("{err}").contains("circular"),
        "expected a circular-include error, got: {err}"
    );
}
