//! Opt-in ORACLE cross-check: actually shell out to the canonical `xacro`
//! (`/opt/ros/jazzy/bin/xacro`) on a minimal `<xacro:property>` + `<link>` probe
//! snippet and diff the expansion against this crate's [`PropertyTables`] result.
//!
//! The parity claim must be ENFORCED, not
//! asserted in a prose comment. The happy-path tests in `properties.rs` pin
//! hand-transcribed constants (correct today, but no guard against transcription
//! error / oracle drift). This harness re-derives each expectation from the live
//! oracle at test time and compares.
//!
//! GATING: these tests are skipped unless `XACRO_ORACLE=1` is set AND the
//! canonical binary is present (so a CI box without ROS is not broken). Run with:
//!
//! ```sh
//! XACRO_ORACLE=1 cargo test --test oracle -- --nocapture
//! ```
//!
//! The diff is intentionally narrow (the property-model subset covered here):
//! a single `<link name="probe_${...}"/>` whose expanded `name=` attribute is the
//! oracle's ground truth, compared to `eval_text`-ing the same probe over a
//! `PropertyTables` built from the same property defs. The full document lexer is
//! out of scope here, so we compare the STRING form of the probe (what xacro serializes
//! into the XML), via `XacroValue::to_python_str` on the port side.

use std::process::Command;
use std::sync::Mutex;

use xacro_pure::{PropertyDef, PropertyTables};

/// Serialize oracle subprocess spawns (see the note in `tests/pipeline.rs`):
/// concurrent `xacro` spawns can trip an `ECHILD` reaper race in some sandboxes.
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

/// Run the canonical oracle on `xacro_src` and return the expanded `probe_...`
/// link name attribute (the text after `probe_` in `<link name="probe_XXX"/>`),
/// or `None` if the oracle errored (so we can assert error parity too).
fn oracle_probe(xacro_src: &str) -> Result<String, String> {
    let dir = std::env::temp_dir().join(format!("xacro-oracle-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("probe-{:x}.xacro", fxhash(xacro_src)));
    std::fs::write(&path, xacro_src).expect("write temp xacro");

    let out = {
        let _guard = ORACLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        Command::new(oracle_bin())
            .arg(&path)
            .output()
            .expect("spawn canonical xacro")
    };
    let _ = std::fs::remove_file(&path);

    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_owned());
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Find `name="probe_..."` and return the part after `probe_`.
    for line in stdout.lines() {
        if let Some(idx) = line.find("name=\"probe_") {
            let rest = &line[idx + "name=\"probe_".len()..];
            if let Some(end) = rest.find('"') {
                return Ok(rest[..end].to_owned());
            }
        }
    }
    Err(format!("no probe_ link in oracle output:\n{stdout}"))
}

/// A tiny stable string hash for a unique temp filename (avoid a dep).
fn fxhash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Build a `<robot>` snippet from a list of `<xacro:property .../>` lines plus a
/// `<link name="probe_${probe}"/>`.
fn snippet(props: &[&str], probe: &str) -> String {
    let mut s = String::from(
        "<?xml version=\"1.0\"?>\n<robot xmlns:xacro=\"http://www.ros.org/wiki/xacro\" name=\"t\">\n",
    );
    for p in props {
        s.push_str("  ");
        s.push_str(p);
        s.push('\n');
    }
    s.push_str(&format!("  <link name=\"probe_{probe}\"/>\n"));
    s.push_str("</robot>\n");
    s
}

/// Assert that the port's `eval_text(probe)` over `defs` stringifies to the SAME
/// thing the oracle expands `probe_${probe}` to, for the given snippet.
fn assert_oracle_parity(defs: &[PropertyDef], probe_expr: &str, prop_lines: &[&str]) {
    if !oracle_enabled() {
        eprintln!("skipping oracle parity (set XACRO_ORACLE=1 with canonical xacro present)");
        return;
    }
    // Port side.
    let mut t = PropertyTables::new();
    let top = t.top_scope();
    for d in defs {
        t.set_property(top, d).expect("port set_property");
    }
    let port = t
        .eval_text_in(top, probe_expr)
        .expect("port eval_text")
        .to_python_str();

    // Oracle side.
    let src = snippet(prop_lines, &xml_escape(probe_expr));
    let oracle = oracle_probe(&src).expect("oracle expansion");

    assert_eq!(
        port, oracle,
        "port vs oracle mismatch for probe `{probe_expr}`\nsnippet:\n{src}"
    );
}

/// Minimal XML attribute escaping for the probe expression embedded in `name=`.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ---------------------------------------------------------------------------
// Parity cases: the SAME inputs the properties.rs regression tests pin, now
// re-derived from the live oracle.
// ---------------------------------------------------------------------------

#[test]
fn oracle_late_binding() {
    assert_oracle_parity(
        &[
            PropertyDef::value("base", "${10}"),
            PropertyDef::value("derived", "${base*2}"),
            PropertyDef::value("base", "${100}"),
        ],
        "${derived}",
        &[
            "<xacro:property name=\"base\" value=\"${10}\"/>",
            "<xacro:property name=\"derived\" value=\"${base*2}\"/>",
            "<xacro:property name=\"base\" value=\"${100}\"/>",
        ],
    );
}

#[test]
fn oracle_lazy_eval_zero_float_eager() {
    assert_oracle_parity(
        &[
            PropertyDef::value("base", "${10}"),
            PropertyDef {
                lazy_eval: Some("${0.0}"),
                ..PropertyDef::value("d", "${base*2}")
            },
            PropertyDef::value("base", "${100}"),
        ],
        "${d}",
        &[
            "<xacro:property name=\"base\" value=\"${10}\"/>",
            "<xacro:property name=\"d\" value=\"${base*2}\" lazy_eval=\"${0.0}\"/>",
            "<xacro:property name=\"base\" value=\"${100}\"/>",
        ],
    );
}

#[test]
fn oracle_dead_branch_circular() {
    assert_oracle_parity(
        &[
            PropertyDef::value("a", "${b}"),
            PropertyDef::value("b", "${a}"),
            PropertyDef::value("ok", "${7}"),
        ],
        "${ok if True else a}",
        &[
            "<xacro:property name=\"a\" value=\"${b}\"/>",
            "<xacro:property name=\"b\" value=\"${a}\"/>",
            "<xacro:property name=\"ok\" value=\"${7}\"/>",
        ],
    );
}

#[test]
fn oracle_unicode_name() {
    assert_oracle_parity(
        &[PropertyDef::value("café", "${42}")],
        "${café}",
        &["<xacro:property name=\"café\" value=\"${42}\"/>"],
    );
}

#[test]
fn oracle_eager_self_reference_dict() {
    assert_oracle_parity(
        &[
            PropertyDef {
                lazy_eval: Some("false"),
                ..PropertyDef::value("d", "${dict(x=1.0)}")
            },
            PropertyDef {
                lazy_eval: Some("false"),
                ..PropertyDef::value("d", "${dict(x=d['x']+100.0)}")
            },
        ],
        "${d['x']}",
        &[
            "<xacro:property name=\"d\" value=\"${dict(x=1.0)}\" lazy_eval=\"false\"/>",
            "<xacro:property name=\"d\" value=\"${dict(x=d['x']+100.0)}\" lazy_eval=\"false\"/>",
        ],
    );
}

#[test]
fn oracle_remove_float_truthy() {
    // remove="${1.5}" is truthy -> x removed -> oracle ERRORS (name x undefined).
    // The port likewise removes x, so reading it is unbound. Assert error parity.
    if !oracle_enabled() {
        eprintln!("skipping oracle parity (set XACRO_ORACLE=1)");
        return;
    }
    let src = snippet(
        &[
            "<xacro:property name=\"x\" value=\"${10}\"/>",
            "<xacro:property name=\"x\" remove=\"${1.5}\"/>",
        ],
        "${x}",
    );
    let oracle = oracle_probe(&src);
    assert!(
        oracle.is_err(),
        "oracle should error (x removed then undefined), got {oracle:?}"
    );

    let mut t = PropertyTables::new();
    let top = t.top_scope();
    t.set_property(top, &PropertyDef::value("x", "${10}")).unwrap();
    t.set_property(
        top,
        &PropertyDef {
            name: "x",
            remove: Some("${1.5}"),
            ..PropertyDef::default()
        },
    )
    .unwrap();
    // Both sides drop x: oracle errors on the now-undefined `x`, port reports it
    // unbound. Mutually-confirmed remove-truthiness parity.
    assert!(!t.contains(top, "x"), "port should also have removed x");
}
