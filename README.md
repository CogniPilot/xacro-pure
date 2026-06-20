# xacro-pure

A faithful, pure-Rust [xacro](https://github.com/ros/xacro) processor. Expands ROS xacro
to URDF with no ROS installation, no Python installation, and no subprocess, on every
target Rust supports, including WebAssembly.

## Why

Every prior Rust take on xacro (xacro-rs, xurdf, pyisheval-based evaluators) falls short
of real-world xacro files for the same reason: `${...}` expressions are arbitrary Python.
xacro-pure closes that gap by backing its expression evaluator with
[rustpython-vm](https://github.com/RustPython/RustPython), a CPython-grade interpreter in
pure Rust, so `${...}` gets full Python `eval()` parity: comprehensions, conditional
expressions, big integers, string methods, the lot.

The rest of the pipeline is a piece-by-piece port of canonical xacro:

- the property model: `Table`/`NameSpace` scopes, lazy and eager evaluation, circular
  reference detection with the canonical error text
- the full `eval_text` lexer: `${...}`, `$(...)` substitution args (`find` / `arg` /
  `eval` / `env` / `optenv` / `dirname` / `cwd`), and `$$` escaping, with canonical
  precedence and ordering
- macros and includes: params, defaults, `^` forwarding, block and `**content`
  parameters, namespaced macros, `ns=` / `optional=` / glob includes, all through an
  injected reader so hermetic and in-browser callers control file access
- `load_yaml` with the unit constructors (`!radians`, `!degrees`, ...) and dotted
  attribute access
- a port of xacro's fixed `writexml` serializer, so output is byte-comparable

Both declarative (SO-ARM class) and programmatic (OpenArm class) xacro files expand
byte-identically to `/opt/ros/*/bin/xacro`. The oracle parity tests that prove this are
included; they are gated behind `XACRO_ORACLE=1` with a canonical xacro on PATH, so the
regular test suite never needs ROS.

## Usage

```rust
use std::collections::HashMap;
use xacro_pure::{process_document, AmentPackageResolver};

let source = std::fs::read_to_string("robot.urdf.xacro")?;
let urdf = process_document(
    &source,
    HashMap::from([("prefix".to_string(), "left_".to_string())]), // xacro args
    Box::new(AmentPackageResolver),                               // $(find pkg)
    |path| std::fs::read_to_string(path).map_err(|e| e.to_string()), // load_yaml reader
)?;
```

For WebAssembly or any environment where the library must not touch the filesystem,
use `process_document_with` and supply an `IncludeReader` plus a `PackageResolver`;
every file access flows through your callbacks.

## License

Apache-2.0. Copyright 2026 CogniPilot Foundation.
