//! The `xacro.*` namespace object exposed into the `${...}` eval scope.
//!
//! Canonical `create_global_symbols` (`xacro/__init__.py:214-217`):
//! ```python
//! expose(load_yaml=load_yaml, abs_filename=abs_filename_spec, dotify=YamlDictWrapper,
//!        ns='xacro', deprecate_msg=...)        # xacro.load_yaml etc. + deprecated bare aliases
//! expose(arg=lambda name: substitution_args_context['arg'][name], ns='xacro')
//! ```
//! i.e. `xacro.load_yaml` / `xacro.abs_filename` / `xacro.dotify` / `xacro.arg`
//! are the modern, canonical forms (the bare `load_yaml(...)` is a deprecated
//! alias kept working). Modern xacros (and the OpenArm / SO-ARM corpus this port
//! targets) call `${xacro.load_yaml('cfg.yaml')}`, so without the `xacro`
//! namespace those files fail with `name 'xacro' is not defined`.
//!
//! The eval VM is `without_stdlib`, so we build `xacro` as a module object whose
//! attributes are native callables, the same shape as the `math` namespace.
//!  * `xacro.load_yaml(file)` uses the injected YAML reader (the document
//!    pipeline supplies a real FS reader / a virtual map), parsing via
//!    [`crate::load_yaml::load_yaml_str`], identical to the bare `load_yaml`.
//!  * `xacro.arg(name)` reads the substitution-args snapshot (canonical's
//!    `substitution_args_context['arg'][name]`), auto-typed.
//!  * `xacro.abs_filename(spec)` / `xacro.dotify(d)` are provided as faithful
//!    no-frills shims (abs_filename returns the spec; dotify returns the dict
//!    unchanged; subscript access already works, and dotted attribute access on a
//!    plain dict is the documented `load_yaml` gap, not a `dotify` regression).

use std::collections::HashMap;

use rustpython_vm::{PyObjectRef, VirtualMachine};

use crate::namespace::YamlReader;
use crate::value::XacroValue;

/// Build the `xacro` namespace module for the current `vm`, capturing the YAML
/// `reader` and an `args` snapshot for `xacro.arg`. Returns the module object to
/// be injected under the name `xacro` in the eval globals.
pub fn build_xacro_module(
    vm: &VirtualMachine,
    reader: Option<&YamlReader>,
    args: &HashMap<String, String>,
) -> PyObjectRef {
    let module = vm.new_module("xacro", vm.ctx.new_dict(), None);

    // xacro.load_yaml(filename)
    if let Some(reader) = reader {
        let reader = reader.clone();
        let load_yaml = vm.new_function(
            "load_yaml",
            move |filename: String, vm: &VirtualMachine| -> rustpython_vm::PyResult {
                let src = reader(&filename)
                    .map_err(|e| vm.new_runtime_error(format!("load_yaml('{filename}'): {e}")))?;
                let value = crate::load_yaml::load_yaml_str(&src)
                    .map_err(|e| vm.new_runtime_error(format!("load_yaml('{filename}'): {e}")))?;
                Ok(crate::bridge::to_py(&value, vm))
            },
        );
        let _ = module.set_attr("load_yaml", load_yaml, vm);
    }

    // xacro.arg(name) -> auto-typed substitution arg.
    {
        let args = args.clone();
        let arg = vm.new_function(
            "arg",
            move |name: String, vm: &VirtualMachine| -> rustpython_vm::PyResult {
                match args.get(&name) {
                    Some(raw) => Ok(crate::bridge::to_py(&convert_value_auto(raw), vm)),
                    None => Err(vm.new_key_error(vm.ctx.new_str(name).into())),
                }
            },
        );
        let _ = module.set_attr("arg", arg, vm);
    }

    // xacro.abs_filename(spec) -> spec (no remapping in the pure/wasm core).
    {
        let abs_filename = vm.new_function("abs_filename", |spec: String| -> String { spec });
        let _ = module.set_attr("abs_filename", abs_filename, vm);
    }

    // xacro.dotify(d) -> d (subscript access already works on the loaded dict).
    {
        let dotify = vm.new_function(
            "dotify",
            |d: PyObjectRef, _vm: &VirtualMachine| -> PyObjectRef { d },
        );
        let _ = module.set_attr("dotify", dotify, vm);
    }

    module.into()
}

/// Canonical `convert_value(value, 'auto')` (duplicated from `substitution_args`
/// to keep this module self-contained): float if it has a `.`, else int, else
/// bool for `true`/`false`, else the raw string.
fn convert_value_auto(value: &str) -> XacroValue {
    if value.contains('.') {
        if let Ok(f) = value.parse::<f64>() {
            return XacroValue::Float(f);
        }
    } else if let Ok(i) = value.parse::<num_bigint::BigInt>() {
        return XacroValue::Int(i);
    }
    match value.to_ascii_lowercase().as_str() {
        "true" => return XacroValue::Bool(true),
        "false" => return XacroValue::Bool(false),
        _ => {}
    }
    XacroValue::Str(value.to_owned())
}
