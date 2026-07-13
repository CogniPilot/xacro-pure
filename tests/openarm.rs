//! THE DECISIVE GATE: the PROGRAMMATIC OpenArm v2.0 robot expands END-TO-END
//! through `xacro-pure` in PURE RUST and matches canonical `xacro`.
//!
//! `openarm_v20.urdf.xacro` is the robot every existing Rust xacro crate
//! (`xacro-rs`, `xurdf`) FAILS on: it drives the expansion entirely from data:
//! eager self-referential `lazy_eval="false"` properties, `dict()`/string-concat/
//! `load_yaml` inside ternaries, deeply RECURSIVE macros, and (the enabling
//! gap) DOTTED `load_yaml` access (`cfg.geometry.scale.x`). If this matches
//! canonical, the port has achieved what no crate could.
//!
//! This test:
//!   1. expands the file through `xacro_pure` (native FS include reader + a
//!      corpus-mapped `$(find)` resolver + a real `load_yaml` file reader), and
//!   2. when the oracle is enabled, expands the SAME file through canonical
//!      `/opt/ros/jazzy/bin/xacro` (via a throwaway ament prefix) and asserts the
//!      two expansions are SEMANTICALLY identical, and additionally reports
//!      BYTE-parity.
//!
//! GATING: skipped unless the corpus is present. The oracle parity half is
//! additionally gated on `XACRO_ORACLE=1` + the canonical binary.

use std::collections::HashMap;
use std::process::Command;
use std::sync::Mutex;

use xacro_pure::{process_document_with, FnPackageResolver, FsIncludeReader, PackageResolver};

static ORACLE_LOCK: Mutex<()> = Mutex::new(());

/// The OpenArm corpus package root (the dir containing `openarm_description/`).
fn corpus_pkg_parent() -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let p = format!("{home}/git/hcdf-conversion-examples");
    if std::path::Path::new(&format!(
        "{p}/openarm_description/assets/robot/openarm_v2.0/urdf/openarm_v20.urdf.xacro"
    ))
    .exists()
    {
        Some(p)
    } else {
        None
    }
}

/// The OpenArm v20 entry xacro path.
fn openarm_entry(pkg_parent: &str) -> String {
    format!(
        "{pkg_parent}/openarm_description/assets/robot/openarm_v2.0/urdf/openarm_v20.urdf.xacro"
    )
}

/// Expand OpenArm through the port. `$(find pkg)` maps to `<pkg_parent>/<pkg>`
/// (the corpus layout), matching what the canonical ament prefix resolves to.
/// `load_yaml(path)` reads the file directly off disk (paths are absolute after
/// `$(find)` expansion).
fn port_expand_openarm(pkg_parent: &str) -> String {
    let entry = openarm_entry(pkg_parent);
    let src = std::fs::read_to_string(&entry).expect("read OpenArm entry");
    let parent = pkg_parent.to_owned();
    let resolver: Box<dyn PackageResolver> = Box::new(FnPackageResolver(move |p: &str| {
        Ok(format!("{parent}/{p}"))
    }));
    let reader = FsIncludeReader;
    let dir = std::path::Path::new(&entry)
        .parent()
        .map(|p| p.to_owned())
        .expect("entry has a parent dir");
    process_document_with(
        &src,
        Some(&entry),
        HashMap::new(),
        resolver,
        move |f: &str| {
            // Canonical load_yaml resolves a relative path against the current
            // filestack; OpenArm passes absolute paths (post-$(find)), but be
            // robust to a relative one by anchoring at the entry dir.
            let path = if std::path::Path::new(f).is_absolute() {
                std::path::PathBuf::from(f)
            } else {
                dir.join(f)
            };
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))
        },
        &reader,
    )
    .unwrap_or_else(|e| panic!("port failed to expand OpenArm: {e}"))
}

/// Expand OpenArm through canonical `xacro` via a throwaway ament prefix, or `None`
/// if the oracle is disabled.
fn oracle_expand_openarm(pkg_parent: &str) -> Option<String> {
    if std::env::var("XACRO_ORACLE").as_deref() != Ok("1") {
        return None;
    }
    let bin = std::env::var("XACRO_BIN").unwrap_or_else(|_| "/opt/ros/jazzy/bin/xacro".to_owned());
    if !std::path::Path::new(&bin).exists() {
        return None;
    }
    let prefix = std::env::temp_dir().join(format!("xacro-openarm-ament-{}", std::process::id()));
    let share = prefix.join("share");
    let marker_dir = share.join("ament_index/resource_index/packages");
    std::fs::create_dir_all(&marker_dir).expect("mkdir ament marker");
    std::fs::write(marker_dir.join("openarm_description"), "").expect("write marker");
    let link = share.join("openarm_description");
    let target = format!("{pkg_parent}/openarm_description");
    let _ = std::fs::remove_file(&link);
    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &link).expect("symlink pkg");

    let entry = openarm_entry(pkg_parent);
    let out = {
        let _guard = ORACLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut attempt = 0;
        loop {
            let r = Command::new(&bin)
                .arg(&entry)
                .env("AMENT_PREFIX_PATH", &prefix)
                .output();
            match r {
                Ok(o) => break o,
                Err(_) if attempt < 10 => {
                    attempt += 1;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                Err(e) => panic!("spawn canonical xacro: {e}"),
            }
        }
    };
    let _ = std::fs::remove_dir_all(&prefix);
    if !out.status.success() {
        panic!(
            "canonical xacro failed on OpenArm:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// A normalized SEMANTIC field stream: each element's tag + sorted normalized
/// attrs (+ a `tag#text` entry for non-empty text), in document order.
fn semantic_fields(xml: &str) -> Vec<(String, String)> {
    let root = xmltree::Element::parse(xml.as_bytes()).expect("parse expansion");
    let mut out = Vec::new();
    walk(&root, &mut out);
    out
}

fn walk(elt: &xmltree::Element, out: &mut Vec<(String, String)>) {
    let mut attrs: Vec<(String, String)> = elt
        .attributes
        .iter()
        .map(|(k, v)| (k.clone(), normalize_value(v)))
        .collect();
    attrs.sort();
    out.push((
        elt.name.clone(),
        attrs
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" "),
    ));
    if let Some(text) = elt.get_text() {
        let t = text.trim();
        if !t.is_empty() {
            out.push((format!("{}#text", elt.name), normalize_value(t)));
        }
    }
    for child in &elt.children {
        if let xmltree::XMLNode::Element(e) = child {
            walk(e, out);
        }
    }
}

/// Canonicalize numeric tokens so the two serializers' float reprs compare equal.
fn normalize_value(v: &str) -> String {
    v.split_whitespace()
        .map(|t| match t.parse::<f64>() {
            Ok(f) => format!("~{:.9e}", f),
            Err(_) => t.to_owned(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Extract the ordered `<TAG name="...">` attribute values for a given `tag`.
fn named(xml: &str, tag: &str) -> Vec<String> {
    semantic_fields(xml)
        .into_iter()
        .filter_map(|(t, attrs)| {
            if t != tag {
                return None;
            }
            attrs
                .split(' ')
                .find_map(|kv| kv.strip_prefix("name=").map(str::to_owned))
        })
        .collect()
}

/// Count occurrences of `<TAG ` in the expansion.
fn tag_count(xml: &str, tag: &str) -> usize {
    semantic_fields(xml)
        .into_iter()
        .filter(|(t, _)| t == tag)
        .count()
}

#[test]
fn openarm_expands_pure_rust() {
    let Some(pkg_parent) = corpus_pkg_parent() else {
        eprintln!("OpenArm corpus not present; skipping gate");
        return;
    };
    let port = port_expand_openarm(&pkg_parent);

    // No xacro:* markup survives.
    assert!(!port.contains("xacro:"), "all xacro markup expanded");
    // It produced a real robot: links + joints + meshes.
    assert!(tag_count(&port, "link") > 5, "has links");
    assert!(tag_count(&port, "joint") > 5, "has joints");
    assert!(tag_count(&port, "mesh") > 5, "has meshes");
    // The world/root frame is present.
    let links = named(&port, "link");
    assert!(links.iter().any(|l| l == "world"), "root frame present");
}

#[test]
fn openarm_matches_canonical_semantically() {
    let Some(pkg_parent) = corpus_pkg_parent() else {
        eprintln!("OpenArm corpus not present; skipping gate");
        return;
    };
    let Some(canonical) = oracle_expand_openarm(&pkg_parent) else {
        eprintln!("oracle disabled; skipping OpenArm parity (run with XACRO_ORACLE=1)");
        return;
    };
    let port = port_expand_openarm(&pkg_parent);

    let cf = semantic_fields(&canonical);
    let pf = semantic_fields(&port);

    assert_eq!(
        cf.len(),
        pf.len(),
        "field count: canonical {} vs port {}",
        cf.len(),
        pf.len()
    );
    // Find the first divergence for a focused error.
    for (i, (c, p)) in cf.iter().zip(pf.iter()).enumerate() {
        assert_eq!(
            c, p,
            "OpenArm semantic mismatch at field {i}:\ncanonical: {c:?}\nport:      {p:?}"
        );
    }
}

#[test]
fn openarm_byte_parity_with_canonical() {
    let Some(pkg_parent) = corpus_pkg_parent() else {
        eprintln!("OpenArm corpus not present; skipping gate");
        return;
    };
    let Some(canonical) = oracle_expand_openarm(&pkg_parent) else {
        eprintln!("oracle disabled; skipping OpenArm byte parity (run with XACRO_ORACLE=1)");
        return;
    };
    let port = port_expand_openarm(&pkg_parent);
    if port != canonical {
        // Report how close, with the first differing line.
        let cl: Vec<&str> = canonical.lines().collect();
        let pl: Vec<&str> = port.lines().collect();
        let first = cl
            .iter()
            .zip(pl.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(cl.len().min(pl.len()));
        panic!(
            "OpenArm NOT byte-identical: canonical {} lines, port {} lines; first diff at line {}:\ncanonical: {:?}\nport:      {:?}",
            cl.len(),
            pl.len(),
            first + 1,
            cl.get(first),
            pl.get(first),
        );
    }
}
