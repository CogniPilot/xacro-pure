//! The `math.*` symbols exposed into every `${...}` eval scope.
//!
//! Canonical xacro's `create_global_symbols` (`xacro/__init__.py:211-212`) does:
//! ```python
//! expose([(k, v) for k, v in math.__dict__.items() if not k.startswith('_')],
//!        ns='math', deprecate_msg='')
//! ```
//! i.e. it injects EVERY public `math` symbol BOTH directly (bare `pi`, `sin`,
//! `radians`, ...) AND under a `math` namespace (`math.pi`, `math.radians`, ...).
//! `pi`/`radians`/`sin`/`cos`/`atan2` are ubiquitous in real robot xacros
//! (`${pi/2}` RPY, `${radians(45)}` joint limits), so without these the pipeline
//! cannot process essentially any SO-ARM / OpenArm file.
//!
//! The eval VM is built `without_stdlib`, so there is no importable `math`
//! module; we reproduce the symbols natively. Each function is a Rust closure
//! registered via `vm.new_function`, accepting Python `int` OR `float` arguments
//! (CPython's math functions coerce ints) through [`ArgIntoFloat`], and returning
//! the same Python type CPython does (`floor`/`ceil`/`trunc`/`factorial`/`gcd`/
//! `lcm`/`comb`/`perm`/`isqrt` -> `int`; `isnan`/`isinf`/`isfinite`/`isclose` ->
//! `bool`; everything else -> `float`). Constants (`pi`/`tau`/`e`/`inf`/`nan`)
//! are injected as floats.
//!
//! Both the bare names and the `math` namespace object reference the SAME native
//! callables, matching canonical's dual exposure.

use rustpython_vm::function::{ArgIntoFloat, OptionalArg};
use rustpython_vm::{PyObjectRef, VirtualMachine};

// NOTE on coverage: `gamma`/`lgamma`/`erf`/`erfc`/`fsum`/`dist`/`frexp`/`modf`/
// `nextafter`/`ulp`/`prod`/`sumprod` (math symbols with no `std`/portable
// pure-Rust equivalent) are intentionally omitted; they do not appear in robot
// xacros. A reference to one therefore raises a faithful `NameError` (the
// disallowed-builtin sentinel / undefined name) rather than silently resolving
// wrong. The symbols installed below are precisely those that ARE exposed.

/// Inject every bare `math` symbol into `set` (a `set(name, obj)` sink), AND build
/// a `math` namespace object with the same members, injecting it under `math`.
///
/// `set` is a closure so the same population logic serves the global-scope
/// injection (`scope.globals.set_item`) used by `eval.rs`.
pub fn install_math<S>(vm: &VirtualMachine, mut set: S)
where
    S: FnMut(&str, PyObjectRef),
{
    // The `math` namespace object: a module whose attributes are the symbols, so
    // `math.pi` / `math.radians(90)` resolve (canonical exposes math under a
    // namespace too). A module supports attribute access without a custom class.
    let math_mod = vm.new_module("math", vm.ctx.new_dict(), None);

    // Helper: bind a symbol both bare (via `set`) and as a `math.` attribute.
    // All call sites pass string literals, so `&'static str` is sufficient and
    // satisfies `set_attr`'s interned-name lifetime requirement.
    let mut bind = |name: &'static str, obj: PyObjectRef| {
        let _ = math_mod.set_attr(name, obj.clone(), vm);
        set(name, obj);
    };

    // ---- constants ----
    bind("pi", vm.ctx.new_float(std::f64::consts::PI).into());
    bind("tau", vm.ctx.new_float(std::f64::consts::TAU).into());
    bind("e", vm.ctx.new_float(std::f64::consts::E).into());
    bind("inf", vm.ctx.new_float(f64::INFINITY).into());
    bind("nan", vm.ctx.new_float(f64::NAN).into());

    // ---- float-returning, single arg ----
    macro_rules! f1 {
        ($name:literal, $f:expr) => {
            bind(
                $name,
                vm.new_function($name, move |x: ArgIntoFloat| -> f64 {
                    let x = f64::from(x);
                    ($f)(x)
                })
                .into(),
            );
        };
    }
    f1!("sin", f64::sin);
    f1!("cos", f64::cos);
    f1!("tan", f64::tan);
    f1!("asin", f64::asin);
    f1!("acos", f64::acos);
    f1!("atan", f64::atan);
    f1!("sinh", f64::sinh);
    f1!("cosh", f64::cosh);
    f1!("tanh", f64::tanh);
    f1!("asinh", f64::asinh);
    f1!("acosh", f64::acosh);
    f1!("atanh", f64::atanh);
    f1!("degrees", f64::to_degrees);
    f1!("radians", f64::to_radians);
    f1!("sqrt", f64::sqrt);
    f1!("cbrt", f64::cbrt);
    f1!("exp", f64::exp);
    f1!("exp2", f64::exp2);
    f1!("expm1", f64::exp_m1);
    f1!("log2", f64::log2);
    f1!("log10", f64::log10);
    f1!("log1p", f64::ln_1p);
    f1!("fabs", f64::abs);

    // ---- float-returning, two args ----
    bind(
        "atan2",
        vm.new_function("atan2", |y: ArgIntoFloat, x: ArgIntoFloat| -> f64 {
            f64::from(y).atan2(f64::from(x))
        })
        .into(),
    );
    bind(
        "hypot",
        vm.new_function("hypot", |x: ArgIntoFloat, y: ArgIntoFloat| -> f64 {
            f64::from(x).hypot(f64::from(y))
        })
        .into(),
    );
    bind(
        "pow",
        vm.new_function("pow", |x: ArgIntoFloat, y: ArgIntoFloat| -> f64 {
            f64::from(x).powf(f64::from(y))
        })
        .into(),
    );
    bind(
        "copysign",
        vm.new_function("copysign", |x: ArgIntoFloat, y: ArgIntoFloat| -> f64 {
            f64::from(x).copysign(f64::from(y))
        })
        .into(),
    );
    bind(
        "fmod",
        vm.new_function("fmod", |x: ArgIntoFloat, y: ArgIntoFloat| -> f64 {
            // C fmod (Python's math.fmod) keeps the sign of x: `x - trunc(x/y)*y`.
            let (x, y) = (f64::from(x), f64::from(y));
            x % y
        })
        .into(),
    );
    bind(
        "remainder",
        vm.new_function("remainder", |x: ArgIntoFloat, y: ArgIntoFloat| -> f64 {
            // IEEE-754 remainder (round-half-to-even quotient), matching
            // math.remainder.
            let (x, y) = (f64::from(x), f64::from(y));
            ieee_remainder(x, y)
        })
        .into(),
    );
    bind(
        "ldexp",
        vm.new_function("ldexp", |m: ArgIntoFloat, e: i32| -> f64 {
            f64::from(m) * 2f64.powi(e)
        })
        .into(),
    );

    // ---- log with optional base (math.log(x[, base])) ----
    bind(
        "log",
        vm.new_function("log", |x: ArgIntoFloat, base: OptionalArg<ArgIntoFloat>| -> f64 {
            let x = f64::from(x);
            match base {
                OptionalArg::Present(b) => x.log(f64::from(b)),
                OptionalArg::Missing => x.ln(),
            }
        })
        .into(),
    );

    // ---- int-returning (floor/ceil/trunc/factorial/gcd/lcm/comb/perm/isqrt) ----
    bind(
        "floor",
        vm.new_function("floor", |x: ArgIntoFloat| -> i64 { f64::from(x).floor() as i64 }).into(),
    );
    bind(
        "ceil",
        vm.new_function("ceil", |x: ArgIntoFloat| -> i64 { f64::from(x).ceil() as i64 }).into(),
    );
    bind(
        "trunc",
        vm.new_function("trunc", |x: ArgIntoFloat| -> i64 { f64::from(x).trunc() as i64 }).into(),
    );
    bind(
        "factorial",
        vm.new_function("factorial", |n: i64| -> i64 {
            // Matches math.factorial for the small ranges xacro uses.
            (1..=n.max(0)).product()
        })
        .into(),
    );
    bind(
        "gcd",
        vm.new_function("gcd", |a: i64, b: i64| -> i64 { gcd_i64(a.abs(), b.abs()) }).into(),
    );
    bind(
        "lcm",
        vm.new_function("lcm", |a: i64, b: i64| -> i64 {
            if a == 0 || b == 0 {
                0
            } else {
                (a.abs() / gcd_i64(a.abs(), b.abs())) * b.abs()
            }
        })
        .into(),
    );
    bind(
        "isqrt",
        vm.new_function("isqrt", |n: i64| -> i64 { (n.max(0) as f64).sqrt() as i64 }).into(),
    );
    bind(
        "comb",
        vm.new_function("comb", |n: i64, k: i64| -> i64 { binom(n, k) }).into(),
    );
    bind(
        "perm",
        vm.new_function("perm", |n: i64, k: OptionalArg<i64>| -> i64 {
            let k = match k {
                OptionalArg::Present(k) => k,
                OptionalArg::Missing => n,
            };
            if k < 0 || k > n {
                0
            } else {
                ((n - k + 1)..=n).product()
            }
        })
        .into(),
    );

    // ---- bool-returning predicates ----
    bind(
        "isnan",
        vm.new_function("isnan", |x: ArgIntoFloat| -> bool { f64::from(x).is_nan() }).into(),
    );
    bind(
        "isinf",
        vm.new_function("isinf", |x: ArgIntoFloat| -> bool { f64::from(x).is_infinite() }).into(),
    );
    bind(
        "isfinite",
        vm.new_function("isfinite", |x: ArgIntoFloat| -> bool { f64::from(x).is_finite() }).into(),
    );
    bind(
        "isclose",
        vm.new_function(
            "isclose",
            |a: ArgIntoFloat,
             b: ArgIntoFloat,
             rel_tol: OptionalArg<ArgIntoFloat>,
             abs_tol: OptionalArg<ArgIntoFloat>|
             -> bool {
                let (a, b) = (f64::from(a), f64::from(b));
                let rel = rel_tol.map_or(1e-9, f64::from);
                let abs = abs_tol.map_or(0.0, f64::from);
                if a == b {
                    return true;
                }
                if a.is_infinite() || b.is_infinite() {
                    return false;
                }
                (a - b).abs() <= (rel * a.abs().max(b.abs())).max(abs)
            },
        )
        .into(),
    );

    // Finally expose the `math` namespace object itself.
    set("math", math_mod.into());
}

/// Euclid's GCD on non-negative `i64`s.
fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// `n choose k` (math.comb): 0 outside `0 <= k <= n`.
fn binom(n: i64, k: i64) -> i64 {
    if k < 0 || k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: i64 = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// IEEE-754 remainder (math.remainder): `x - round_half_even(x / y) * y`.
fn ieee_remainder(x: f64, y: f64) -> f64 {
    if y == 0.0 || x.is_infinite() {
        return f64::NAN;
    }
    if y.is_infinite() {
        return x;
    }
    let q = (x / y).round_ties_even();
    x - q * y
}
