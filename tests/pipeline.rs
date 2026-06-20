//! The DOCUMENT-PIPELINE cross-check.
//!
//! Two layers:
//!   * **Self-contained pipeline tests** (always run): expand a SYNTHETIC xacro
//!     through `xacro_pure::process_document` and assert the set of expanded
//!     `<link name="...">` probes (plus that all `xacro:*` markup is gone). These
//!     exercise `<xacro:property>` + `<xacro:if>`/`<xacro:unless>` + `${...}` +
//!     `$(arg ...)` + `$(eval ...)` + `load_yaml`, with an injected fake
//!     [`FnPackageResolver`] + an in-test YAML map (so they are hermetic + wasm-
//!     shaped: no filesystem, no ROS environment).
//!   * **Oracle parity tests** (opt-in, `XACRO_ORACLE=1` + canonical `xacro`
//!     present): expand the SAME source through canonical `/opt/ros/jazzy/bin/
//!     xacro` and diff the `<link>` probe set against the port. This ENFORCES the
//!     parity claim rather than asserting it in prose, mirroring `oracle.rs`.
//!
//! The diff is at the SEMANTIC level (the ordered list of `name="..."` link
//! attributes), not byte-for-byte: the faithful pretty-printer is a separate
//! concern, so the two serializers differ in whitespace/quote-style. The link-name
//! set is the property/conditional/substitution ground truth this pipeline is
//! responsible for.

use std::collections::HashMap;
use std::process::Command;
use std::sync::Mutex;

use xacro_pure::{process_document, FnPackageResolver, PackageResolver};

/// Serialize oracle subprocess spawns. `cargo test` runs test fns on many
/// threads; spawning many `xacro` subprocesses concurrently in some sandboxes
/// trips an `ECHILD` ("No child processes") race in the reaper. The expansion
/// itself is correct under serialization, so we gate the spawn behind a mutex
/// (only affects the opt-in oracle path; the port-only assertions stay parallel).
static ORACLE_LOCK: Mutex<()> = Mutex::new(());

/// The canonical oracle binary. Override with `XACRO_BIN` if installed elsewhere.
fn oracle_bin() -> String {
    std::env::var("XACRO_BIN").unwrap_or_else(|_| "/opt/ros/jazzy/bin/xacro".to_owned())
}

/// True iff the opt-in oracle cross-check should run (env-gated + binary present).
fn oracle_enabled() -> bool {
    if std::env::var("XACRO_ORACLE").as_deref() != Ok("1") {
        return false;
    }
    std::path::Path::new(&oracle_bin()).exists()
}

/// A deterministic fake `$(find)` resolver: `pkg -> /share/pkg`.
fn fake_resolver() -> Box<dyn PackageResolver> {
    Box::new(FnPackageResolver(|p: &str| Ok(format!("/share/{p}"))))
}

/// Extract the ordered list of `<link name="VALUE">` attribute values from an
/// expanded XML string. Robust to both serializers' quoting/whitespace.
fn link_names(xml: &str) -> Vec<String> {
    let mut names = Vec::new();
    let needle = "<link";
    let mut rest = xml;
    while let Some(i) = rest.find(needle) {
        rest = &rest[i + needle.len()..];
        // Within this tag, find name="..." (up to the tag close).
        let tag_end = rest.find('>').unwrap_or(rest.len());
        let tag = &rest[..tag_end];
        if let Some(ni) = tag.find("name=\"") {
            let after = &tag[ni + "name=\"".len()..];
            if let Some(q) = after.find('"') {
                names.push(after[..q].to_owned());
            }
        }
        rest = &rest[tag_end..];
    }
    names
}

/// Expand `src` with the port, using `mappings` for `$(arg)`, a fake `$(find)`,
/// and an in-test YAML map for `load_yaml`.
fn port_expand(src: &str, mappings: &[(&str, &str)], yaml: &[(&str, &str)]) -> String {
    let map: HashMap<String, String> = mappings
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    let yaml_map: HashMap<String, String> = yaml
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    process_document(src, map, fake_resolver(), move |f: &str| {
        yaml_map
            .get(f)
            .cloned()
            .ok_or_else(|| format!("no such yaml file '{f}'"))
    })
    .unwrap_or_else(|e| panic!("port process_document failed: {e}\nsource:\n{src}"))
}

/// Run the canonical oracle on `src` (writing any `yaml` files alongside it) with
/// the given `mappings` (as `name:=value` CLI args), returning the expanded XML.
fn oracle_expand(src: &str, mappings: &[(&str, &str)], yaml: &[(&str, &str)]) -> String {
    let dir = std::env::temp_dir().join(format!(
        "xacro-pure-pipeline-{}-{:x}",
        std::process::id(),
        fxhash(src)
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("doc.xacro");
    std::fs::write(&path, src).expect("write xacro");
    for (name, content) in yaml {
        std::fs::write(dir.join(name), content).expect("write yaml");
    }

    let mut cmd = Command::new(oracle_bin());
    cmd.arg(&path);
    for (k, v) in mappings {
        cmd.arg(format!("{k}:={v}"));
    }
    let out = {
        // Serialize the spawn (see ORACLE_LOCK). A poisoned lock still yields the
        // guard, which is fine here.
        let _guard = ORACLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        cmd.output().expect("spawn canonical xacro")
    };
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        out.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A tiny stable string hash for a unique temp dir.
fn fxhash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Assert the port's expanded link set equals `expected`, AND (when enabled)
/// equals the oracle's. `yaml` is referenced by `load_yaml(name)`; for the
/// oracle the file is written next to the xacro, so reference it by BASENAME.
fn assert_pipeline(
    src: &str,
    mappings: &[(&str, &str)],
    yaml: &[(&str, &str)],
    expected: &[&str],
) {
    let port_xml = port_expand(src, mappings, yaml);
    let port_links = link_names(&port_xml);
    let expected: Vec<String> = expected.iter().map(|s| (*s).to_owned()).collect();
    assert_eq!(
        port_links, expected,
        "port link set mismatch\nexpanded:\n{port_xml}"
    );
    // The xacro namespace + markup must be fully gone.
    assert!(
        !port_xml.contains("xacro:") && !port_xml.contains("xmlns:xacro"),
        "xacro markup leaked into output:\n{port_xml}"
    );

    if oracle_enabled() {
        let oracle_xml = oracle_expand(src, mappings, yaml);
        let oracle_links = link_names(&oracle_xml);
        assert_eq!(
            port_links, oracle_links,
            "port vs oracle link mismatch\nport:\n{port_xml}\noracle:\n{oracle_xml}"
        );
    } else {
        eprintln!("oracle parity skipped (set XACRO_ORACLE=1 with canonical xacro present)");
    }
}

// ---------------------------------------------------------------------------
// Synthetic pipeline cases.
// ---------------------------------------------------------------------------

const PROLOGUE: &str =
    "<?xml version=\"1.0\"?>\n<robot xmlns:xacro=\"http://www.ros.org/wiki/xacro\" name=\"t\">\n";
const EPILOGUE: &str = "</robot>\n";

/// Wrap `body` in a minimal robot with the xacro namespace.
fn doc(body: &str) -> String {
    format!("{PROLOGUE}{body}{EPILOGUE}")
}

#[test]
fn properties_and_late_binding() {
    let src = doc(
        "  <xacro:property name=\"base\" value=\"${10}\"/>\n\
         \x20 <xacro:property name=\"derived\" value=\"${base*2}\"/>\n\
         \x20 <xacro:property name=\"base\" value=\"${100}\"/>\n\
         \x20 <link name=\"d_${derived}\"/>\n",
    );
    // derived is lazy -> late binding -> base=100 -> 200.
    assert_pipeline(&src, &[], &[], &["d_200"]);
}

#[test]
fn if_unless_conditionals() {
    let src = doc(
        "  <xacro:property name=\"flag\" value=\"true\"/>\n\
         \x20 <xacro:if value=\"${flag}\"><link name=\"on\"/></xacro:if>\n\
         \x20 <xacro:unless value=\"${flag}\"><link name=\"off\"/></xacro:unless>\n\
         \x20 <xacro:if value=\"${1 > 2}\"><link name=\"never\"/></xacro:if>\n\
         \x20 <xacro:unless value=\"${1 > 2}\"><link name=\"always\"/></xacro:unless>\n",
    );
    assert_pipeline(&src, &[], &[], &["on", "always"]);
}

#[test]
fn arg_default_and_override() {
    let src = doc(
        "  <xacro:arg name=\"robot\" default=\"so100\"/>\n\
         \x20 <link name=\"a_$(arg robot)\"/>\n",
    );
    // default.
    assert_pipeline(&src, &[], &[], &["a_so100"]);
    // overridden by a mapping.
    assert_pipeline(&src, &[("robot", "so101")], &[], &["a_so101"]);
}

#[test]
fn arg_inside_expression_resolves_first() {
    // The OpenArm pattern: $(arg ...) inside a ${...} string literal resolves to
    // TEXT before the surrounding python expression evaluates.
    let src = doc(
        "  <xacro:arg name=\"robot\" default=\"so100\"/>\n\
         \x20 <link name=\"b_${'pre_' + '$(arg robot)'}\"/>\n",
    );
    assert_pipeline(&src, &[], &[], &["b_pre_so100"]);
}

#[test]
fn arg_inside_bare_expression_arithmetic() {
    // A `$(arg n)` as a bare sub-token of a `${...}` arithmetic expression: it
    // resolves to text (3) FIRST, then the surrounding python arithmetic runs
    // (3 * 4 = 12). Pins the handle_expr inner-eval_text-before-safe_eval rule.
    let src = doc(
        "  <xacro:arg name=\"n\" default=\"3\"/>\n\
         \x20 <link name=\"x_${$(arg n) * 4}\"/>\n",
    );
    assert_pipeline(&src, &[], &[], &["x_12"]);
}

#[test]
fn dollar_dollar_escape_in_attribute() {
    // $${x} is the LITERAL text ${x} (no expansion).
    let src = doc("  <link name=\"lit_$${x}\"/>\n");
    assert_pipeline(&src, &[], &[], &["lit_${x}"]);
}

#[test]
fn eval_substitution_whole_attribute() {
    let src = doc("  <link name=\"$(eval 2 * 5 + 1)\"/>\n");
    assert_pipeline(&src, &[], &[], &["11"]);
}

#[test]
fn load_yaml_subscript_and_units() {
    // load_yaml + nested subscript + the !radians/!degrees unit constructors.
    let yaml = "limits:\n  lower: -1.5\nangle: !radians 0.5\ndeg: !degrees 90\n";
    let src = doc(
        "  <xacro:property name=\"cfg\" value=\"${load_yaml('params.yaml')}\"/>\n\
         \x20 <link name=\"lim_${cfg['limits']['lower']}\"/>\n\
         \x20 <link name=\"ang_${cfg['angle']}\"/>\n\
         \x20 <link name=\"deg_${cfg['deg']}\"/>\n",
    );
    assert_pipeline(
        &src,
        &[],
        &[("params.yaml", yaml)],
        &["lim_-1.5", "ang_0.5", "deg_1.5707963267948966"],
    );
}

#[test]
fn nested_property_and_arithmetic() {
    let src = doc(
        "  <xacro:property name=\"w\" value=\"${1+2}\"/>\n\
         \x20 <xacro:property name=\"big\" value=\"${w*10}\"/>\n\
         \x20 <link name=\"p_${big}\"/>\n\
         \x20 <link name=\"c_${1 + 2 * 3}\"/>\n",
    );
    assert_pipeline(&src, &[], &[], &["p_30", "c_7"]);
}

// ---------------------------------------------------------------------------
// Regression fixes: float str() parity, math.*, tuples, xacro namespace, lexer
// errors, comment removal.
// ---------------------------------------------------------------------------

#[test]
fn float_str_scientific_notation() {
    // Python str(float) switches to scientific form when the decimal exponent is
    // < -4 or >= 16. Re-derived against the live oracle (see also the diff probe
    // in this file). `${1e15}` stays fixed (`1000000000000000.0`), `${1e16}`
    // flips to `1e+16`; small magnitudes mirror (`${1e-7}` -> `1e-07`).
    let src = doc(
        "  <link name=\"a${1e16}\"/>\n\
         \x20 <link name=\"b${1e-7}\"/>\n\
         \x20 <link name=\"c${0.00009}\"/>\n\
         \x20 <link name=\"d${1.5e300}\"/>\n\
         \x20 <link name=\"e${1e15}\"/>\n\
         \x20 <link name=\"f${3.0}\"/>\n\
         \x20 <link name=\"g${[1e16,1e-7]}\"/>\n\
         \x20 <link name=\"h${123456789012345680.0}\"/>\n",
    );
    assert_pipeline(
        &src,
        &[],
        &[],
        &[
            "a1e+16",
            "b1e-07",
            "c9e-05",
            "d1.5e+300",
            "e1000000000000000.0",
            "f3.0",
            "g[1e+16, 1e-07]",
            "h1.2345678901234568e+17",
        ],
    );
}

#[test]
fn math_symbols_bare_and_namespace() {
    // pi/radians/sin/sqrt/floor/atan2/tau/degrees are exposed both bare and under
    // a `math` namespace, matching canonical create_global_symbols.
    let src = doc(
        "  <link name=\"a${pi}\"/>\n\
         \x20 <link name=\"b${radians(90)}\"/>\n\
         \x20 <link name=\"c${sin(0)}\"/>\n\
         \x20 <link name=\"d${sqrt(4)}\"/>\n\
         \x20 <link name=\"e${floor(3.7)}\"/>\n\
         \x20 <link name=\"f${atan2(1,1)}\"/>\n\
         \x20 <link name=\"g${math.pi}\"/>\n\
         \x20 <link name=\"h${tau}\"/>\n\
         \x20 <link name=\"i${degrees(pi)}\"/>\n\
         \x20 <link name=\"j${pi/2}\"/>\n",
    );
    assert_pipeline(
        &src,
        &[],
        &[],
        &[
            "a3.141592653589793",
            "b1.5707963267948966",
            "c0.0",
            "d2.0",
            "e3",
            "f0.7853981633974483",
            "g3.141592653589793",
            "h6.283185307179586",
            "i180.0",
            "j1.5707963267948966",
        ],
    );
}

#[test]
fn tuple_results_str() {
    // A ${...} that yields a tuple str()s like Python: parenthesized, 1-tuple
    // keeps the trailing comma, empty tuple is `()`.
    // Note: a tuple of strings str()s with single-quotes (`('x', 'y')`); inside an
    // XML attribute the serializer entity-escapes `'` as `&apos;` (a pretty-printer
    // concern, not the value), so we probe numeric/nested tuples here and
    // pin the str-repr of a string-tuple directly below.
    let src = doc(
        "  <link name=\"a${(1, 2.0)}\"/>\n\
         \x20 <link name=\"b${(1,)}\"/>\n\
         \x20 <link name=\"c${()}\"/>\n\
         \x20 <link name=\"d${(1, (2, 3))}\"/>\n",
    );
    assert_pipeline(&src, &[], &[], &["a(1, 2.0)", "b(1,)", "c()", "d(1, (2, 3))"]);
}

#[test]
fn xacro_namespace_api() {
    // The canonical xacro.load_yaml / xacro.arg forms (plus the bare load_yaml
    // deprecated alias) all resolve. PyYAML's float resolver requires a signed
    // exponent, so `mass: 2.5e16` stays the STRING `2.5e16` (NOT a float), while
    // `angle: 1.0e-7` is a float -> `1e-07`.
    let yaml = "angle: 1.0e-7\nmass: 2.5e16\n";
    let src = doc(
        "  <xacro:arg name=\"aa\" default=\"hi\"/>\n\
         \x20 <link name=\"a${xacro.load_yaml('p.yaml')['angle']}\"/>\n\
         \x20 <link name=\"b${load_yaml('p.yaml')['mass']}\"/>\n\
         \x20 <link name=\"c${xacro.arg('aa')}\"/>\n",
    );
    assert_pipeline(
        &src,
        &[],
        &[("p.yaml", yaml)],
        &["a1e-07", "b2.5e16", "chi"],
    );
}

#[test]
fn eval_substitution_bare_arg_and_math() {
    // $(eval ...) resolves bare arg names (auto-typed) and math symbols, matching
    // canonical's _DictWrapper + _eval_dict.
    let src = doc(
        "  <xacro:arg name=\"count\" default=\"3\"/>\n\
         \x20 <xacro:arg name=\"w\" default=\"1.5\"/>\n\
         \x20 <xacro:arg name=\"flag\" default=\"true\"/>\n\
         \x20 <link name=\"a$(eval count + 10)\"/>\n\
         \x20 <link name=\"b$(eval w * 2)\"/>\n\
         \x20 <link name=\"c$(eval flag)\"/>\n\
         \x20 <link name=\"d$(eval pi)\"/>\n",
    );
    assert_pipeline(
        &src,
        &[],
        &[],
        &["a13", "b3.0", "cTrue", "d3.141592653589793"],
    );
}

#[test]
fn comments_before_control_nodes_removed() {
    // remove_previous_comments: a comment immediately before xacro:property /
    // xacro:if / xacro:arg is dropped; a comment before a plain element is kept.
    let src = doc(
        "  <!-- propdoc --><xacro:property name=\"r\" value=\"5\"/>\n\
         \x20 <!-- guard --><xacro:if value=\"true\"><link name=\"kept\"/></xacro:if>\n\
         \x20 <!-- argdoc --><xacro:arg name=\"z\" default=\"9\"/>\n\
         \x20 <!-- keepme --><link name=\"plain\"/>\n\
         \x20 <link name=\"uses${r}\"/>\n",
    );
    let xml = port_expand(&src, &[], &[]);
    assert_eq!(link_names(&xml), vec!["kept", "plain", "uses5"]);
    // The three control-node comments are gone; the plain-element comment stays.
    assert!(!xml.contains("propdoc"), "propdoc comment leaked:\n{xml}");
    assert!(!xml.contains("guard"), "guard comment leaked:\n{xml}");
    assert!(!xml.contains("argdoc"), "argdoc comment leaked:\n{xml}");
    assert!(xml.contains("keepme"), "plain comment wrongly removed:\n{xml}");
}

#[test]
fn lexer_rejects_malformed_dollar() {
    // Canonical QuickLexer raises `invalid expression: <rest>` when no token
    // matches: an unterminated ${/$( or a $$+ run with no following brace. The
    // port must ERROR (not silently emit literal text).
    for bad in [
        "<link name=\"foo$(bar\"/>",
        "<link name=\"foo${bar\"/>",
        "<link name=\"$$\"/>",
        "<link name=\"$$x\"/>",
        "<link name=\"lit$$\"/>",
        "<link name=\"a$$$\"/>",
    ] {
        let src = doc(&format!("  {bad}\n"));
        let map: HashMap<String, String> = HashMap::new();
        let r = process_document(&src, map, fake_resolver(), |_: &str| {
            Err("no yaml".to_owned())
        });
        assert!(
            r.is_err(),
            "expected lexer error for [{bad}] but got Ok:\n{:?}",
            r
        );
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("invalid expression"),
            "expected 'invalid expression' for [{bad}], got: {msg}"
        );
    }
}

#[test]
fn lexer_accepts_wellformed_dollar() {
    // Well-formed `$` sequences both serializers accept: a lone/trailing $, a $
    // before a non-brace char, a $$-escape, and consecutive $-runs.
    let src = doc(
        "  <link name=\"cost$\"/>\n\
         \x20 <link name=\"p$5\"/>\n\
         \x20 <link name=\"a$b$c\"/>\n\
         \x20 <link name=\"lit$${x}\"/>\n\
         \x20 <link name=\"a$$${y}b\"/>\n",
    );
    assert_pipeline(
        &src,
        &[],
        &[],
        &["cost$", "p$5", "a$b$c", "lit${x}", "a$${y}b"],
    );
}
