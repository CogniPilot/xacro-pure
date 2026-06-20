//! Error type for the expression-evaluation seam. Every failure path returns an
//! [`EvalError`] (never a panic), so the eventual `eval_text` caller can wrap it
//! the way canonical xacro re-raises as a `XacroException` with context.

use std::fmt;

/// An error from compiling or evaluating a `${...}` expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    /// The expression referenced a dunder (`__...`) name. Canonical xacro's
    /// `safe_eval` rejects these via `code.co_names` to block sandbox escape;
    /// this is the same security check applied to the parsed name set.
    InvalidName(String),
    /// The expression failed to compile (a syntax error).
    Compile(String),
    /// The expression compiled but raised at runtime (e.g. `NameError`,
    /// `KeyError`, `ZeroDivisionError`). Carries the Python traceback text.
    Runtime(String),
    /// The result was a Python type with no `XacroValue` representation.
    UnsupportedType(String),
    /// An internal inconsistency crossing the Rust/Python bridge.
    Bridge(String),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::InvalidName(names) => {
                write!(f, "Use of invalid name(s): {names}")
            }
            EvalError::Compile(msg) => write!(f, "expression compile error: {msg}"),
            EvalError::Runtime(msg) => write!(f, "expression runtime error: {msg}"),
            EvalError::UnsupportedType(msg) => write!(f, "unsupported result type: {msg}"),
            EvalError::Bridge(msg) => write!(f, "value bridge error: {msg}"),
        }
    }
}

impl std::error::Error for EvalError {}
