//! THE GATE: SO-ARM101 expands END-TO-END through `xacro-pure` and matches
//! canonical `xacro`.
//!
//! `so_arm101.urdf.xacro` is the declarative robot the whole port targets: it
//! combines `$(arg)`/`$(find)` substitution, two `xacro:include`s (one of which
//! supplies a macro, the other a ros2_control macro), namespaced macro CALLS
//! (`xacro:so_arm101`, `xacro:ros2_control`), `xacro:if` conditionals inside a
//! macro body, and `${...}` property/expression evaluation throughout.
//!
//! This test:
//!   1. expands the file through `xacro_pure` (native FS include reader + a
//!      corpus-mapped `$(find)` resolver), and
//!   2. when the oracle is enabled, expands the SAME file through canonical
//!      `/opt/ros/jazzy/bin/xacro` (via a throwaway ament prefix) and asserts the
//!      two expansions are SEMANTICALLY identical (normalized element/attribute
//!      tree, numeric tokens canonicalized so the two serializers' float reprs
//!      compare equal).
//!
//! GATING: skipped unless the corpus is present. The oracle parity half is
//! additionally gated on `XACRO_ORACLE=1` + the canonical binary. The port-only
//! structural assertions (link/joint/mesh counts + names) run whenever the corpus
//! is present, so the GATE has teeth even without ROS installed.

use std::collections::HashMap;
use std::process::Command;
use std::sync::Mutex;

use xacro_pure::{process_document_with, FnPackageResolver, FsIncludeReader, PackageResolver};

static ORACLE_LOCK: Mutex<()> = Mutex::new(());

/// The SO-ARM corpus package root (the dir containing `so_arm101_description/`).
fn corpus_pkg_parent() -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let p = format!("{home}/git/hcdf-conversion-examples/ros2_so_arm");
    if std::path::Path::new(&format!("{p}/so_arm101_description/urdf/so_arm101.urdf.xacro")).exists()
    {
        Some(p)
    } else {
        None
    }
}

/// The SO-ARM entry xacro path.
fn soarm_entry(pkg_parent: &str) -> String {
    format!("{pkg_parent}/so_arm101_description/urdf/so_arm101.urdf.xacro")
}

/// Expand SO-ARM through the port. `$(find pkg)` maps to `<pkg_parent>/<pkg>`
/// (the corpus layout), matching what the canonical ament prefix resolves to.
fn port_expand_soarm(pkg_parent: &str) -> String {
    let entry = soarm_entry(pkg_parent);
    let src = std::fs::read_to_string(&entry).expect("read SO-ARM entry");
    let parent = pkg_parent.to_owned();
    let resolver: Box<dyn PackageResolver> =
        Box::new(FnPackageResolver(move |p: &str| Ok(format!("{parent}/{p}"))));
    let reader = FsIncludeReader;
    process_document_with(
        &src,
        Some(&entry),
        HashMap::new(),
        resolver,
        |_f: &str| Err("load_yaml not used by SO-ARM".to_owned()),
        &reader,
    )
    .unwrap_or_else(|e| panic!("port failed to expand SO-ARM: {e}"))
}

/// Expand SO-ARM through canonical `xacro` via a throwaway ament prefix, or `None`
/// if the oracle is disabled.
fn oracle_expand_soarm(pkg_parent: &str) -> Option<String> {
    if std::env::var("XACRO_ORACLE").as_deref() != Ok("1") {
        return None;
    }
    let bin = std::env::var("XACRO_BIN").unwrap_or_else(|_| "/opt/ros/jazzy/bin/xacro".to_owned());
    if !std::path::Path::new(&bin).exists() {
        return None;
    }
    // Build a throwaway ament prefix exposing so_arm101_description.
    let prefix = std::env::temp_dir().join(format!("xacro-soarm-ament-{}", std::process::id()));
    let share = prefix.join("share");
    let marker_dir = share.join("ament_index/resource_index/packages");
    std::fs::create_dir_all(&marker_dir).expect("mkdir ament marker");
    std::fs::write(marker_dir.join("so_arm101_description"), "").expect("write marker");
    let link = share.join("so_arm101_description");
    let target = format!("{pkg_parent}/so_arm101_description");
    let _ = std::fs::remove_file(&link);
    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &link).expect("symlink pkg");

    let entry = soarm_entry(pkg_parent);
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
            "canonical xacro failed on SO-ARM:\n{}",
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

/// Count occurrences of `<TAG ` in the expansion (a structural tally).
fn tag_count(xml: &str, tag: &str) -> usize {
    semantic_fields(xml).into_iter().filter(|(t, _)| t == tag).count()
}

#[test]
fn soarm_expands_with_expected_structure() {
    let Some(pkg_parent) = corpus_pkg_parent() else {
        eprintln!("SO-ARM corpus not present; skipping gate");
        return;
    };
    let port = port_expand_soarm(&pkg_parent);

    // Robot-level links: the 9 SO-ARM links (world + 8 body links).
    let links = named(&port, "link");
    assert_eq!(
        links,
        vec![
            "world",
            "base_link",
            "shoulder_link",
            "upper_arm_link",
            "lower_arm_link",
            "wrist_link",
            "gripper_link",
            "gripper_frame_link",
            "jaw_link",
        ],
        "SO-ARM robot-level links"
    );

    // Robot-level joints carrying a type: 8 (1 base-fixed + 6 revolute + 1 fixed
    // gripper-frame). (`<joint>` also appears inside transmissions/ros2_control;
    // those carry no `type`, so we filter on the type-bearing ones.)
    let typed_joints: Vec<(String, String)> = semantic_fields(&port)
        .into_iter()
        .filter(|(t, attrs)| t == "joint" && attrs.contains("type="))
        .collect();
    assert_eq!(typed_joints.len(), 8, "type-bearing joints");

    // Meshes: 34 mesh elements (17 distinct visual/collision pairs).
    assert_eq!(tag_count(&port, "mesh"), 34, "mesh elements");

    // Transmissions: 6.
    assert_eq!(tag_count(&port, "transmission"), 6, "transmissions");

    // ros2_control: present, with mock_components selected (default hardware type).
    assert!(port.contains("mock_components/GenericSystem"));
    assert!(!port.contains("FeetechHardwareInterface")); // the 'real' branch dropped
    assert!(!port.contains("MujocoSystemInterface")); // the 'mujoco' branch dropped

    // Material color comes from the $(arg color) default "0.2 0.7 0.85 1.0".
    assert!(port.contains("0.2 0.7 0.85 1.0"));

    // No xacro:* markup survives.
    assert!(!port.contains("xacro:"), "all xacro markup expanded");
}

#[test]
fn soarm_matches_canonical_semantically() {
    let Some(pkg_parent) = corpus_pkg_parent() else {
        eprintln!("SO-ARM corpus not present; skipping gate");
        return;
    };
    let Some(canonical) = oracle_expand_soarm(&pkg_parent) else {
        eprintln!("oracle disabled; skipping SO-ARM parity (run with XACRO_ORACLE=1)");
        return;
    };
    let port = port_expand_soarm(&pkg_parent);

    let cf = semantic_fields(&canonical);
    let pf = semantic_fields(&port);

    // Field-count parity (the 226-field semantic bar; SO-ARM has ~393).
    assert_eq!(
        cf.len(),
        pf.len(),
        "field count: canonical {} vs port {}",
        cf.len(),
        pf.len()
    );
    // Full ordered semantic equality.
    assert_eq!(
        cf, pf,
        "SO-ARM semantic mismatch\n--- canonical (first 4000) ---\n{}\n--- port (first 4000) ---\n{}",
        &canonical[..canonical.len().min(4000)],
        &port[..port.len().min(4000)]
    );
}

#[test]
fn soarm_byte_parity_with_canonical() {
    let Some(pkg_parent) = corpus_pkg_parent() else {
        eprintln!("SO-ARM corpus not present; skipping gate");
        return;
    };
    let Some(canonical) = oracle_expand_soarm(&pkg_parent) else {
        eprintln!("oracle disabled; skipping SO-ARM byte parity (run with XACRO_ORACLE=1)");
        return;
    };
    let port = port_expand_soarm(&pkg_parent);
    if port != canonical {
        let cl: Vec<&str> = canonical.lines().collect();
        let pl: Vec<&str> = port.lines().collect();
        let first = cl
            .iter()
            .zip(pl.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(cl.len().min(pl.len()));
        panic!(
            "SO-ARM NOT byte-identical: canonical {} lines, port {} lines; first diff at line {}:\ncanonical: {:?}\nport:      {:?}",
            cl.len(),
            pl.len(),
            first + 1,
            cl.get(first),
            pl.get(first),
        );
    }
}
