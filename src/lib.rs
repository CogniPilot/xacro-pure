//! # xacro-pure: a faithful, pure-Rust port of canonical ROS `xacro`
//!
//! Now COMPLETE and the **production engine behind
//! `hcdformat::expand_xacro`** (feature `xacro`): a plain path dependency, no
//! subprocess fallback, no `xacro-rs`/`pyisheval` fork, no Python anywhere.
//! It expands both the declarative (SO-ARM) and programmatic (OpenArm) xacro
//! classes BYTE-IDENTICALLY to `/opt/ros/*/bin/xacro`; the corpus tests in
//! `tests/` compare against the live canonical oracle.
//!
//! The pieces, each a direct port of its canonical counterpart:
//!
//! * [`safe_eval`]: canonical `xacro/__init__.py::safe_eval`, backed by
//!   `rustpython-vm` (a CPython-grade interpreter in pure Rust) so `${...}`
//!   expressions get **full Python `eval()` parity**. Dunder names rejected,
//!   builtins sandboxed to the canonical allowlist, `math.*`/`xacro.*` +
//!   namespace + Rust-function injection. See [`eval`].
//! * [`eval_literal`]: canonical `Table._eval_literal`, the def-time type
//!   coercion (int -> float -> bool, single-quote strip, PEP515 guard). See
//!   [`literal`].
//! * [`PropertyTables`]: canonical xacro's `Table`/`NameSpace` system: per-scope
//!   property tables with a per-property lazy/eager flag and lazy
//!   resolution-on-first-access (late binding, eager early-binding, eager
//!   self-reference, scopes, circular detection with the exact canonical error).
//!   See [`table`].
//! * The full `eval_text` lexer: `${...}` with the typed-single-result rule (a
//!   single `${...}` returns the TYPED value; mixed text is str-joined), `$(...)`
//!   substitution args (`arg`/`find`/`env`/`optenv`/`dirname`/`cwd`/`eval`), and
//!   the `$$` escapes. See [`eval_text`](crate::eval_text).
//! * Macros + includes: params/defaults, `^`/`^|` forwarding, `*block`/
//!   `**content`, recursion; `ns=`/`optional=`/glob includes read through an
//!   injected reader (wasm-clean). See [`macros`] and [`includes`].
//! * `load_yaml`: unit constructors (`!radians`/`!degrees`/...) and dotted
//!   access (`cfg.geometry.scale.x`) via the canonical dict/list wrappers. See
//!   [`load_yaml`].
//! * The canonical serializer: a `fixed_writexml` port so output is
//!   byte-comparable to canonical's pretty-printer. See [`serialize`].
//!
//! The document pipeline has two entry points: [`process_document`] (native,
//! `std::fs`-backed) and [`process_document_with`] (any target incl. wasm;
//! caller-supplied include/YAML readers and `$(find)` package resolver).
//!
//! ## Example
//! ```
//! use xacro_pure::{safe_eval, Namespace, XacroValue};
//!
//! let mut ns = Namespace::new();
//! ns.set("reflect_x", XacroValue::int(-1));
//! let v = safe_eval("(reflect_x if reflect_x == -1 else 1) * 3.0", &ns).unwrap();
//! assert_eq!(v, XacroValue::Float(-3.0));
//! ```

mod bridge;
mod dom;
mod error;
mod eval;
mod eval_text;
mod includes;
mod literal;
mod load_yaml;
mod macros;
mod math_builtins;
mod namespace;
mod serialize;
mod substitution_args;
mod table;
mod value;
mod xacro_builtins;

#[cfg(not(target_arch = "wasm32"))]
pub use dom::process_document;
pub use dom::{process_document_with, ProcessError, DEFAULT_MAX_EXPANSION_DEPTH};
pub use includes::{FnIncludeReader, IncludeReader};
#[cfg(not(target_arch = "wasm32"))]
pub use includes::FsIncludeReader;
pub use error::EvalError;
pub use eval::safe_eval;
pub use literal::eval_literal;
pub use namespace::Namespace;
pub use substitution_args::{AmentPackageResolver, FnPackageResolver, PackageResolver};
pub use table::{is_valid_name, PropertyDef, PropertyTables, ScopeId};
pub use value::XacroValue;
