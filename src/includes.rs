//! INCLUDES: `xacro:include` (relative resolution, globbing, namespaces,
//! optional, the injected reader seam).
//!
//! A faithful port of canonical xacro's include machinery (`xacro/__init__.py`):
//! `process_include`, `get_include_files`, `abs_filename_spec`,
//! `import_xml_namespaces`, and the `filestack` bookkeeping.
//!
//! ## In-order processing (matching canonical `ros2`)
//! The `ros2` branch of canonical xacro does NOT run a separate include pre-pass:
//! `xacro:include` is handled INLINE inside `eval_all` (in-order processing), so
//! an include's contents are evaluated in document order alongside macros and
//! properties. We follow the canonical code exactly; the include branch in
//! [`crate::dom::eval_all`] recurses `eval_all` over the included document's root
//! and splices its children in place of the `<xacro:include>` element.
//!
//! ## The injected reader seam (wasm-capable)
//! Canonical reads an included file with `open()` (native FS). To keep the core
//! wasm-capable, the file bytes are obtained through an INJECTED reader closure
//! (the same seam pattern `load_yaml` uses): native callers supply a
//! `std::fs`-backed reader; a wasm caller supplies a virtual-FS map. The reader
//! takes the RESOLVED absolute path and returns the file source (or an error,
//! which `optional=` may swallow).
//!
//! ## Relative resolution + the filestack
//! A non-absolute `filename` is resolved against the dirname of the CURRENT file
//! (`filestack[-1]`), via [`abs_filename_spec`]. The filestack is pushed before
//! reading an included file and popped after, so nested includes resolve relative
//! to their own location and `$(dirname)` sees the right file. The same stack
//! powers cycle detection.
//!
//! ## Globbing
//! A `filename` containing a glob metachar (`*`, `?`, or `[`) is expanded to the
//! sorted list of matching files (canonical `glob.glob`, `sorted`). To stay
//! dependency-light and wasm-clean, globbing is delegated to the injected
//! [`IncludeReader::glob`] (native: a real directory walk; wasm/tests: the
//! virtual map's matching keys). A non-glob spec is taken verbatim.

use std::path::{Path, PathBuf};

/// The injected file-access seam for `xacro:include`. Native callers back this
/// with `std::fs`; wasm/test callers back it with a virtual map. Kept object-safe
/// (a `dyn IncludeReader`) so the pipeline holds one without a type parameter.
pub trait IncludeReader {
    /// Read the file at the RESOLVED absolute path `path`, returning its source.
    /// An `Err` is treated as an I/O error (swallowed by `optional=true`).
    fn read(&self, path: &str) -> Result<String, String>;

    /// Expand a glob `spec` (already made absolute) into the SORTED list of
    /// matching paths. The default returns the spec verbatim as a single-element
    /// list (no globbing); a reader that supports globbing overrides this.
    fn glob(&self, spec: &str) -> Vec<String> {
        vec![spec.to_owned()]
    }
}

/// A [`IncludeReader`] backed by closures, so a test/wasm caller can write one
/// inline. `read` is required; `glob` falls back to the trait default (verbatim)
/// unless a `glob_fn` is supplied.
pub struct FnIncludeReader<R, G>
where
    R: Fn(&str) -> Result<String, String>,
    G: Fn(&str) -> Vec<String>,
{
    /// The read closure.
    pub read_fn: R,
    /// The glob-expansion closure.
    pub glob_fn: G,
}

impl<R, G> IncludeReader for FnIncludeReader<R, G>
where
    R: Fn(&str) -> Result<String, String>,
    G: Fn(&str) -> Vec<String>,
{
    fn read(&self, path: &str) -> Result<String, String> {
        (self.read_fn)(path)
    }
    fn glob(&self, spec: &str) -> Vec<String> {
        (self.glob_fn)(spec)
    }
}

/// The native `std::fs`-backed include reader: reads files from disk and globs
/// via a directory walk (no external glob crate, a small wasm-clean walker that
/// is simply never linked on wasm, where this reader is not used).
#[cfg(not(target_arch = "wasm32"))]
pub struct FsIncludeReader;

#[cfg(not(target_arch = "wasm32"))]
impl IncludeReader for FsIncludeReader {
    fn read(&self, path: &str) -> Result<String, String> {
        std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))
    }

    fn glob(&self, spec: &str) -> Vec<String> {
        glob_fs(spec)
    }
}

/// `abs_filename_spec(filename_spec)`: if `filename_spec` is not already
/// absolute, prepend the dirname of the current file (`filestack[-1]`). A `None`
/// current file means basedir `.` (canonical's `parent_filename or '.'`).
pub fn abs_filename_spec(filename_spec: &str, current_file: Option<&str>) -> String {
    let p = Path::new(filename_spec);
    if p.is_absolute() {
        return filename_spec.to_owned();
    }
    let basedir = match current_file {
        Some(f) if !f.is_empty() => {
            let parent = Path::new(f).parent();
            match parent {
                Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
                _ => PathBuf::from("."),
            }
        }
        _ => PathBuf::from("."),
    };
    basedir.join(filename_spec).to_string_lossy().into_owned()
}

/// Whether a filename spec contains a glob metacharacter (`*`, `?`, `[`). Mirrors
/// canonical's `re.search('[*[?]+', filename_spec)`.
pub fn is_glob(spec: &str) -> bool {
    spec.contains('*') || spec.contains('?') || spec.contains('[')
}

/// `get_include_files`: turn a (already eval_text'd) filename spec into the list
/// of files to include. Resolves relative to `current_file`, then globs if it
/// contains a metachar (sorted), else returns the single resolved path.
pub fn get_include_files(
    filename_spec: &str,
    current_file: Option<&str>,
    reader: &dyn IncludeReader,
) -> Vec<String> {
    let abs = abs_filename_spec(filename_spec, current_file);
    if is_glob(&abs) {
        let mut files = reader.glob(&abs);
        files.sort();
        files
    } else {
        vec![abs]
    }
}

/// A minimal native glob for the `*`/`?`/`[...]` metacharacters over the real
/// filesystem, sufficient for `xacro:include` specs (which glob a single
/// directory level in practice, e.g. `urdf/*.xacro`). Avoids an external glob
/// crate (keeping the dep set minimal); compiled out on wasm.
#[cfg(not(target_arch = "wasm32"))]
fn glob_fs(spec: &str) -> Vec<String> {
    // Split the spec into directory components; walk literal components and match
    // a component containing a metachar against the directory entries.
    let path = Path::new(spec);
    let mut bases: Vec<PathBuf> = vec![if path.is_absolute() {
        PathBuf::from("/")
    } else {
        PathBuf::from(".")
    }];
    let mut first = true;
    for comp in path.components() {
        use std::path::Component;
        let part = match comp {
            Component::RootDir => continue,
            Component::CurDir => continue,
            Component::Normal(s) => s.to_string_lossy().into_owned(),
            Component::ParentDir => "..".to_owned(),
            Component::Prefix(p) => p.as_os_str().to_string_lossy().into_owned(),
        };
        let mut next: Vec<PathBuf> = Vec::new();
        if part.contains('*') || part.contains('?') || part.contains('[') {
            for base in &bases {
                if let Ok(entries) = std::fs::read_dir(base) {
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().into_owned();
                        if fnmatch(&part, &name) {
                            next.push(entry.path());
                        }
                    }
                }
            }
        } else {
            for base in &bases {
                next.push(base.join(&part));
            }
        }
        bases = next;
        first = false;
        if bases.is_empty() {
            break;
        }
    }
    if first {
        return Vec::new();
    }
    bases
        .into_iter()
        .map(|p| {
            // Normalize a leading `./` away to match canonical glob output shape.
            let s = p.to_string_lossy().into_owned();
            s.strip_prefix("./").map(str::to_owned).unwrap_or(s)
        })
        .collect()
}

/// A minimal `fnmatch` for the glob metacharacters `*` (any run), `?` (one char),
/// `[...]` (a char class). Used only by the native [`glob_fs`].
#[cfg(not(target_arch = "wasm32"))]
fn fnmatch(pattern: &str, name: &str) -> bool {
    fn rec(p: &[char], n: &[char]) -> bool {
        if p.is_empty() {
            return n.is_empty();
        }
        match p[0] {
            '*' => {
                // match zero or more chars
                rec(&p[1..], n) || (!n.is_empty() && rec(p, &n[1..]))
            }
            '?' => !n.is_empty() && rec(&p[1..], &n[1..]),
            '[' => {
                // a char class up to the closing `]`.
                if n.is_empty() {
                    return false;
                }
                if let Some(close) = p.iter().position(|&c| c == ']') {
                    let class = &p[1..close];
                    if char_in_class(class, n[0]) {
                        rec(&p[close + 1..], &n[1..])
                    } else {
                        false
                    }
                } else {
                    // literal `[`
                    !n.is_empty() && p[0] == n[0] && rec(&p[1..], &n[1..])
                }
            }
            c => !n.is_empty() && c == n[0] && rec(&p[1..], &n[1..]),
        }
    }
    let pc: Vec<char> = pattern.chars().collect();
    let nc: Vec<char> = name.chars().collect();
    rec(&pc, &nc)
}

/// Whether `c` is in the bracket class `class` (supporting `a-z` ranges and a
/// leading `!`/`^` negation). Used only by the native [`fnmatch`].
#[cfg(not(target_arch = "wasm32"))]
fn char_in_class(class: &[char], c: char) -> bool {
    let (negate, class) = match class.first() {
        Some('!') | Some('^') => (true, &class[1..]),
        _ => (false, class),
    };
    let mut i = 0;
    let mut found = false;
    while i < class.len() {
        if i + 2 < class.len() && class[i + 1] == '-' {
            if class[i] <= c && c <= class[i + 2] {
                found = true;
            }
            i += 3;
        } else {
            if class[i] == c {
                found = true;
            }
            i += 1;
        }
    }
    found != negate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abs_relative_to_current() {
        assert_eq!(
            abs_filename_spec("inc.xacro", Some("/a/b/main.xacro")),
            "/a/b/inc.xacro"
        );
    }

    #[test]
    fn abs_already_absolute() {
        assert_eq!(
            abs_filename_spec("/x/y/inc.xacro", Some("/a/b/main.xacro")),
            "/x/y/inc.xacro"
        );
    }

    #[test]
    fn abs_no_current_file() {
        assert_eq!(abs_filename_spec("inc.xacro", None), "./inc.xacro");
    }

    #[test]
    fn glob_detection() {
        assert!(is_glob("urdf/*.xacro"));
        assert!(is_glob("a?b"));
        assert!(is_glob("a[0-9]b"));
        assert!(!is_glob("plain.xacro"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn fnmatch_basic() {
        assert!(fnmatch("*.xacro", "a.xacro"));
        assert!(fnmatch("a?b", "axb"));
        assert!(!fnmatch("a?b", "ab"));
        assert!(fnmatch("a[0-9]b", "a5b"));
        assert!(!fnmatch("a[0-9]b", "axb"));
        assert!(!fnmatch("*.xacro", "a.urdf"));
    }
}
