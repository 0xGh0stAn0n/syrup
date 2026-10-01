//! Turns operation names such as `find_faces_in_top_half` into typed plans
//! that can be generated, compiled and loaded as native code.

pub mod abi {
    include!("abi_prelude.rs");
}

pub mod catalog;
pub mod error;
pub mod intent;
pub mod interp;
pub mod plan;

pub use error::{ErrorKind, Result, Stage, SyrupError};
pub use intent::{Intent, NormRect, OrderKey, PixelRect, Ratio, RegionSpec};
