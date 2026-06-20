//! Shared test helpers for the macro/include cross-checks: a hermetic port
//! expander, a normalized SEMANTIC field extractor, and an opt-in canonical-xacro
//! oracle parity check.

#![allow(dead_code)] // each test crate uses a subset of these helpers

use std::collections::HashMap;
use std::process::Command;
use std::sync::Mutex;

use xacro_pure::{
    process_document_with, FnIncludeReader, FnPackageResolver, PackageResolver, ProcessError,
};

/// Serialize oracle subprocess spawns (cargo runs tests on many threads; some
/// sandboxes trip an `ECHILD` reaper race when spawning many `xacro` concurrently).
static ORACLE_LOCK: Mutex<()> = Mutex::new(());

/// The canonical oracle binary. Override with `XACRO_BIN` if installed elsewhere.
fn oracle_bin() -> String {
    std::env::var("XACRO_BIN").unwrap_or_else(|_| "/opt/ros/jazzy/bin/xacro".to_owned())
}

/// True iff the opt-in oracle cross-check should run (env-gated + binary present).
pub fn oracle_enabled() -> bool {
    if std::env::var("XACRO_ORACLE").as_deref() != Ok("1") {
        return false;
    }
    std::path::Path::new(&oracle_bin()).exists()
}

/// A deterministic fake `$(find)` resolver: `pkg -> /share/pkg`.
fn fake_resolver() -> Box<dyn PackageResolver> {
    Box::new(FnPackageResolver(|p: &str| Ok(format!("/share/{p}"))))
}

/// Expand `src` with the port through a HERMETIC include reader backed by
/// `extra_files` (a map of path -> content). No filesystem, no ROS; wasm-shaped.
pub fn port_expand_with_files(src: &str, extra_files: &[(&str, &str)]) -> String {
    port_expand_result_with_files(src, extra_files)
        .unwrap_or_else(|e| panic!("port process_document failed: {e}\nsource:\n{src}"))
}

/// Expand `src` with the port (no included files needed).
pub fn port_expand(src: &str) -> String {
    port_expand_with_files(src, &[])
}

/// Fallible port expansion with no included files (for error-path assertions).
pub fn port_expand_result(src: &str) -> Result<String, ProcessError> {
    port_expand_result_with_files(src, &[])
}

/// Fallible port expansion with an in-memory virtual FS for includes.
pub fn port_expand_result_with_files(
    src: &str,
    extra_files: &[(&str, &str)],
) -> Result<String, ProcessError> {
    let files: HashMap<String, String> = extra_files
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    let keys: Vec<String> = files.keys().cloned().collect();
    let reader = FnIncludeReader {
        read_fn: move |f: &str| {
            // The port (current_file=None) resolves a relative include to
            // `./name`; the virtual-FS keys are bare names, so strip a leading
            // `./` before lookup.
            let key = f.strip_prefix("./").unwrap_or(f);
            files
                .get(key)
                .cloned()
                .ok_or_else(|| format!("no such file: {f}"))
        },
        glob_fn: move |spec: &str| {
            let spec = spec.strip_prefix("./").unwrap_or(spec);
            // A tiny glob over the virtual-FS keys; re-prefix matches with `./`
            // so the resolved paths feed back through `read_fn` consistently.
            let mut out: Vec<String> = keys
                .iter()
                .filter(|k| glob_match(spec, k))
                .map(|k| format!("./{k}"))
                .collect();
            out.sort();
            out
        },
    };
    process_document_with(
        src,
        None,
        HashMap::new(),
        fake_resolver(),
        |_f: &str| Err("no yaml in test".to_owned()),
        &reader,
    )
}

/// A minimal glob matcher for the virtual-FS reader (`*` = any run within a path
/// component-agnostic match; `?` = one char). Sufficient for the include tests.
fn glob_match(pattern: &str, name: &str) -> bool {
    fn rec(p: &[u8], n: &[u8]) -> bool {
        match p.first() {
            None => n.is_empty(),
            Some(b'*') => rec(&p[1..], n) || (!n.is_empty() && rec(p, &n[1..])),
            Some(b'?') => !n.is_empty() && rec(&p[1..], &n[1..]),
            Some(&c) => !n.is_empty() && c == n[0] && rec(&p[1..], &n[1..]),
        }
    }
    rec(pattern.as_bytes(), name.as_bytes())
}

/// Extract a flat, NORMALIZED list of `(tag, attr-string)` fields from an XML
/// string: each element contributes its tag plus its sorted `k=v` attribute pairs
/// (numeric values canonicalized so the two serializers' float reprs compare
/// equal). This is the SEMANTIC ground truth, order-preserving over the document.
pub fn semantic_fields(xml: &str) -> Vec<(String, String)> {
    let root = match xmltree_parse(xml) {
        Some(r) => r,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    walk(&root, &mut out);
    out
}

/// A node in our minimal parse: tag + attrs + children + text. We reuse
/// `xmltree` (already a dep) for parsing.
fn xmltree_parse(xml: &str) -> Option<xmltree::Element> {
    xmltree::Element::parse(xml.as_bytes()).ok()
}

fn walk(elt: &xmltree::Element, out: &mut Vec<(String, String)>) {
    let mut attrs: Vec<(String, String)> = elt
        .attributes
        .iter()
        .map(|(k, v)| (k.clone(), normalize_value(v)))
        .collect();
    attrs.sort();
    let attr_str = attrs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ");
    out.push((elt.name.clone(), attr_str));
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

/// Canonicalize a (possibly whitespace-separated, possibly numeric) value so that
/// the two serializers' differing float reprs (e.g. `0.5` vs `0.5`, `1e-16` vs
/// `1.0e-16`) compare equal. Non-numeric tokens pass through verbatim.
fn normalize_value(v: &str) -> String {
    let toks: Vec<&str> = v.split_whitespace().collect();
    if toks.is_empty() {
        return v.trim().to_owned();
    }
    toks.iter()
        .map(|t| match t.parse::<f64>() {
            Ok(f) => format!("~{:.9e}", f),
            Err(_) => (*t).to_owned(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Run the canonical oracle on `src` (writing any `extra_files` alongside it,
/// relative to the temp dir) and return the expanded XML, or `None` if the oracle
/// is disabled.
pub fn oracle_expand_with_files(src: &str, extra_files: &[(&str, &str)]) -> Option<String> {
    if !oracle_enabled() {
        return None;
    }
    let dir = std::env::temp_dir().join(format!(
        "xacro-pure-p3-{}-{:x}",
        std::process::id(),
        fxhash(src)
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("doc.xacro");
    std::fs::write(&path, src).expect("write xacro");
    for (name, content) in extra_files {
        let p = dir.join(name);
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&p, content).expect("write extra file");
    }
    // Serialize the ENTIRE spawn+wait under the global lock: cargo runs tests on
    // many threads, and concurrent `wait()` on child processes trips an `ECHILD`
    // ("No child processes") reaper race in some sandboxes. Holding the lock for
    // the whole `output()` (spawn AND wait) keeps a single oracle subprocess
    // outstanding at a time. A few retries absorb any residual transient.
    let out = {
        let _guard = ORACLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut attempt = 0;
        loop {
            match Command::new(oracle_bin()).arg(&path).output() {
                Ok(o) => break o,
                Err(_) if attempt < 10 => {
                    attempt += 1;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                Err(e) => panic!("spawn canonical xacro failed: {e}"),
            }
        }
    };
    let _ = std::fs::remove_dir_all(&dir);
    if !out.status.success() {
        panic!(
            "canonical xacro failed on:\n{src}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Assert the port's expansion of `src` is SEMANTICALLY identical to the canonical
/// oracle's (when the oracle is enabled; otherwise a no-op). The port is given no
/// included files; for include tests use [`assert_semantic_parity_with_files`].
pub fn assert_semantic_parity(src: &str) {
    assert_semantic_parity_with_files(src, &[]);
}

/// Whether the canonical oracle ERRORS (non-zero exit) on `src`. Returns `true`
/// when the oracle is disabled (so an error-path parity assertion that wraps this
/// is a no-op when the oracle is unavailable). Used to confirm the port and the
/// oracle agree on the ERROR cases (e.g. a confined body-defined macro called
/// outside its body).
pub fn oracle_errors(src: &str) -> bool {
    oracle_errors_with_files(src, &[])
}

/// [`oracle_errors`] with an in-memory/on-disk set of included files.
pub fn oracle_errors_with_files(src: &str, extra_files: &[(&str, &str)]) -> bool {
    if !oracle_enabled() {
        return true; // disabled -> treat as "agrees" so the wrapper is a no-op
    }
    let dir = std::env::temp_dir().join(format!(
        "xacro-pure-p3err-{}-{:x}",
        std::process::id(),
        fxhash(src)
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("doc.xacro");
    std::fs::write(&path, src).expect("write xacro");
    for (name, content) in extra_files {
        let p = dir.join(name);
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&p, content).expect("write extra file");
    }
    let out = {
        let _guard = ORACLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut attempt = 0;
        loop {
            match Command::new(oracle_bin()).arg(&path).output() {
                Ok(o) => break o,
                Err(_) if attempt < 10 => {
                    attempt += 1;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                Err(e) => panic!("spawn canonical xacro failed: {e}"),
            }
        }
    };
    let _ = std::fs::remove_dir_all(&dir);
    !out.status.success()
}

/// Assert semantic parity with an in-memory/on-disk set of included files.
pub fn assert_semantic_parity_with_files(src: &str, extra_files: &[(&str, &str)]) {
    let Some(canonical) = oracle_expand_with_files(src, extra_files) else {
        return; // oracle disabled
    };
    let port = port_expand_with_files(src, extra_files);
    let cf = semantic_fields(&canonical);
    let pf = semantic_fields(&port);
    assert_eq!(
        cf, pf,
        "semantic mismatch\n--- canonical ---\n{canonical}\n--- port ---\n{port}"
    );
}

/// Extract the ROOT element's prefixed-namespace map (`prefix -> uri`) from an XML
/// string, excluding the reserved `xml`/`xmlns` prefixes and the default namespace.
/// Used to assert that `xmlns:*` declarations are lifted to the parent
/// (`import_xml_namespaces`). `xmltree` keeps these in `Element.namespaces`.
pub fn root_namespace_map(xml: &str) -> std::collections::BTreeMap<String, String> {
    let mut out = std::collections::BTreeMap::new();
    let root = match xmltree::Element::parse(xml.as_bytes()) {
        Ok(r) => r,
        Err(_) => return out,
    };
    if let Some(ns) = root.namespaces.as_ref() {
        for (prefix, uri) in &ns.0 {
            if prefix.is_empty() || prefix == "xml" || prefix == "xmlns" {
                continue;
            }
            out.insert(prefix.clone(), uri.clone());
        }
    }
    out
}

/// Assert the OUTPUT ROOT's lifted `xmlns:*` namespace map matches the canonical
/// oracle's (when enabled; otherwise a no-op). Both the port and the oracle strip
/// `xmlns:xacro`, so the comparison is over the genuinely lifted declarations.
pub fn assert_root_namespaces_parity_with_files(src: &str, extra_files: &[(&str, &str)]) {
    let Some(canonical) = oracle_expand_with_files(src, extra_files) else {
        return; // oracle disabled
    };
    let port = port_expand_with_files(src, extra_files);
    let cf = root_namespace_map(&canonical);
    let pf = root_namespace_map(&port);
    assert_eq!(
        cf, pf,
        "root namespace-map mismatch\n--- canonical ---\n{canonical}\n--- port ---\n{port}"
    );
}

/// A tiny deterministic hash for temp-file names (FNV-1a).
fn fxhash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
