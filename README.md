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

## `load_yaml` and PyYAML's SafeLoader

Canonical xacro loads YAML with PyYAML's `safe_load`. `load_yaml` matches it for
every construct that appears in robot description YAML: scalars, mappings, sequences,
the `!!str` / `!!int` / `!!float` / `!!bool` / `!!null` / `!!seq` / `!!map` core-schema
tags, and xacro's own `!radians` / `!degrees` / ... unit constructors. Five standard
tags that SafeLoader constructs are deliberately unsupported, and one implicit type
resolves differently:

- `!!set`, `!!omap`, `!!pairs`, `!!binary`, `!!timestamp` are rejected with an error
  naming the construct. There is no value type for a set, ordered map, pair list,
  byte string, or timestamp; none appear in robot YAML; and byte-identical `str()`
  output for a set is impossible in principle (Python set iteration order is
  hash-randomized per process). A clear rejection beats accepting a substitute type.
- YAML aliases (`*anchor`) are rejected: a `load_yaml` document is a plain value tree.
- A plain (untagged, unquoted) timestamp such as `2026-07-12` stays a string, where
  PyYAML resolves it to a `date` / `datetime`. Under `str()` interpolation the result
  is textually identical; the two differ only if an expression uses the value in a
  type-dependent way. An explicit `!!timestamp` tag is rejected as above.

## License

Apache-2.0. Copyright 2026 CogniPilot Foundation.
