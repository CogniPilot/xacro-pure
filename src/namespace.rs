//! The evaluation namespace: the property table plus the registered Rust
//! functions that an expression may call.
//!
//! In canonical xacro the `globals` passed to `safe_eval` is the property
//! `Table` augmented with `math.*`, a few allowed builtins, and the `xacro.*` /
//! `python.*` callables (`load_yaml`, `radians`, `degrees`, ...). This namespace
//! models the two pieces that matter for the eval seam:
//!
//!   * **properties**: `name -> XacroValue`, injected as Python variables.
//!   * **functions**: named Rust closures callable from the Python expression.
//!     This is the extensible hook for `xacro.load_yaml` & friends;
//!     demonstrated here by a stub used by the 9th gate case.
//!
//! The full property model (`Table`/`NameSpace`, lazy/eager flags, scopes,
//! circular detection) lives in [`crate::table`], NOT here.

use indexmap::IndexMap;
use rustpython_vm::{PyObjectRef, VirtualMachine};

use crate::value::XacroValue;

/// A registered Rust function callable from a Python expression.
///
/// `build` is invoked once per evaluation, inside the live VM, to produce the
/// callable Python object. Wrapping the construction in a closure (rather than
/// holding a `PyObjectRef`) is what lets a registration outlive any single
/// short-lived interpreter; the actual `PyNativeFunction` is built fresh in
/// whatever VM is doing the eval.
pub struct FunctionReg {
    /// The name the function is bound to in the expression scope.
    pub name: String,
    /// Constructs the callable in the given VM.
    pub build: Box<dyn Fn(&VirtualMachine) -> PyObjectRef>,
}

/// A YAML source reader for the `xacro.load_yaml` / bare `load_yaml` callables.
/// Stored on the namespace (not just as a [`FunctionReg`]) so the eval layer can
/// ALSO build the canonical `xacro` namespace object (`xacro.load_yaml(...)`)
/// from the same reader, alongside the bare deprecated alias. An `Rc` so it can
/// be cheaply cloned into the per-eval native callable.
pub type YamlReader = std::rc::Rc<dyn Fn(&str) -> Result<String, String>>;

/// The namespace handed to [`crate::safe_eval`].
#[derive(Default)]
pub struct Namespace {
    properties: IndexMap<String, XacroValue>,
    functions: Vec<FunctionReg>,
    /// The YAML reader behind `load_yaml` / `xacro.load_yaml`, if installed.
    yaml_reader: Option<YamlReader>,
}

impl Namespace {
    /// An empty namespace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind (or rebind) a property `name` to `value`.
    pub fn set(&mut self, name: impl Into<String>, value: XacroValue) -> &mut Self {
        self.properties.insert(name.into(), value);
        self
    }

    /// Iterate over the bound properties (insertion order).
    pub fn properties(&self) -> impl Iterator<Item = (&String, &XacroValue)> {
        self.properties.iter()
    }

    /// Iterate over the registered functions.
    pub fn functions(&self) -> impl Iterator<Item = &FunctionReg> {
        self.functions.iter()
    }

    /// Install the YAML reader used by `load_yaml` / `xacro.load_yaml`.
    pub fn set_yaml_reader(&mut self, reader: YamlReader) -> &mut Self {
        self.yaml_reader = Some(reader);
        self
    }

    /// The installed YAML reader, if any.
    pub fn yaml_reader(&self) -> Option<&YamlReader> {
        self.yaml_reader.as_ref()
    }

    /// Register a Rust function under `name`, given a builder that constructs the
    /// callable in the active VM. This is the low-level seam; see
    /// [`Namespace::register_fn_i64`] for the common arity-1 numeric shape.
    pub fn register_raw(
        &mut self,
        name: impl Into<String>,
        build: impl Fn(&VirtualMachine) -> PyObjectRef + 'static,
    ) -> &mut Self {
        self.functions.push(FunctionReg {
            name: name.into(),
            build: Box::new(build),
        });
        self
    }

    /// Convenience: register a pure `i64 -> i64` function (the simplest shape
    /// that proves the seam, a stand-in for the eventual `xacro.load_yaml` /
    /// `radians` callables). `vm.new_function` accepts any closure whose
    /// signature `rustpython` can map to Python call conventions.
    pub fn register_fn_i64(
        &mut self,
        name: &'static str,
        f: impl Fn(i64) -> i64 + Copy + 'static,
    ) -> &mut Self {
        self.register_raw(name, move |vm| vm.new_function(name, f).into())
    }
}
