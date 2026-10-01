//! The one native entry point the Python package uses. Operations are not
//! bound one by one: Python passes a name and pixels, and gets JSON back.

use std::path::PathBuf;
use std::sync::OnceLock;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;
use serde::Deserialize;
use syrup_runtime::intent::region_named;
use syrup_runtime::{
    ErrorKind, ImageInput, Intent, NormRect, Operation, OrderKey, OwnedImage, PixelRect, Ratio,
    RegionSpec, RunParams, Runtime, Stage, SyrupError, catalog,
};

create_exception!(_native, NativeError, PyException);

fn raise(e: SyrupError) -> PyErr {
    NativeError::new_err(serde_json::to_string(&e).expect("errors serialize"))
}

fn runtime() -> PyResult<&'static Runtime> {
    static RUNTIME: OnceLock<Result<Runtime, SyrupError>> = OnceLock::new();
    RUNTIME
        .get_or_init(Runtime::from_env)
        .as_ref()
        .map_err(|e| raise(e.clone()))
}

#[pyclass(frozen, module = "syrup._native")]
struct NativeOperation(Operation);

#[pymethods]
impl NativeOperation {
    #[getter]
    fn name(&self) -> &str {
        self.0.name()
    }

    #[getter]
    fn intent(&self) -> String {
        self.0.intent().to_string()
    }

    #[getter]
    fn plan_hash(&self) -> &str {
        self.0.plan_hash()
    }

    #[getter]
    fn artifact_key(&self) -> &str {
        self.0.artifact_key()
    }

    #[getter]
    fn needs_region(&self) -> bool {
        self.0.plan().needs_caller_region()
    }

    fn plan(&self) -> String {
        serde_json::to_string(self.0.plan()).expect("plans serialize")
    }

    fn source(&self) -> String {
        self.0.source()
    }

    fn explain(&self) -> String {
        self.0.explain()
    }

    fn prepare(&self, py: Python<'_>) -> PyResult<String> {
        let prepared = py.detach(|| self.0.prepare()).map_err(raise)?;
        Ok(serde_json::json!({
            "status": prepared.status,
            "library": prepared.library,
            "elapsed_ms": prepared.elapsed.as_secs_f64() * 1000.0,
            "manifest": prepared.manifest,
        })
        .to_string())
    }

    #[pyo3(signature = (pixels, width, height, channels, min_confidence=None, max_results=None, region=None))]
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        py: Python<'_>,
        pixels: PyBackedBytes,
        width: u32,
        height: u32,
        channels: u32,
        min_confidence: Option<f32>,
        max_results: Option<u32>,
        region: Option<(u32, u32, u32, u32)>,
    ) -> PyResult<String> {
        let params = RunParams {
            min_confidence,
            max_results,
            region: region.map(|(x, y, w, h)| PixelRect { x, y, w, h }),
        };
        let result = py
            .detach(|| {
                let image = ImageInput::new(&pixels, width, height, channels)?;
                self.0.run(&image, &params)
            })
            .map_err(raise)?;
        Ok(serde_json::to_string(&result).expect("results serialize"))
    }
}

#[pyfunction]
fn resolve(name: &str) -> PyResult<NativeOperation> {
    runtime()?.resolve(name).map(NativeOperation).map_err(raise)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    find: String,
    region: Option<RegionArg>,
    order: Option<String>,
    limit: Option<u32>,
    min_area_pct: Option<u32>,
    max_area_pct: Option<u32>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RegionArg {
    Named(String),
    Fractions([f64; 4]),
}

fn intent_from(spec: Spec) -> Result<Intent, SyrupError> {
    let malformed = |reason: String| SyrupError::new(Stage::Resolve, ErrorKind::Malformed, reason);
    let target = catalog::target_named(&spec.find).ok_or_else(|| {
        SyrupError::new(
            Stage::Resolve,
            ErrorKind::Unsupported,
            format!("nothing in the catalog finds {:?}", spec.find),
        )
        .with_hint(format!("known targets: {}", catalog::known_targets()))
    })?;
    let region = match spec.region {
        None => None,
        Some(RegionArg::Named(name)) => Some(
            region_named(&name)
                .ok_or_else(|| malformed(format!("{name:?} is not a region name")))?,
        ),
        Some(RegionArg::Fractions(f)) => {
            let ratio = |v: f64| {
                Ratio::from_f64(v)
                    .ok_or_else(|| malformed(format!("region fraction {v} is not in [0, 1]")))
            };
            let rect = NormRect::new(ratio(f[0])?, ratio(f[1])?, ratio(f[2])?, ratio(f[3])?)
                .ok_or_else(|| malformed(format!("region {f:?} is empty or leaves the image")))?;
            Some(RegionSpec::Fixed { rect })
        }
    };
    let order = match spec.order {
        None => OrderKey::ConfidenceDesc,
        Some(word) => OrderKey::parse(&word)
            .ok_or_else(|| malformed(format!("{word:?} is not an ordering")))?,
    };
    Ok(Intent {
        target,
        region,
        order,
        limit: spec.limit,
        min_area_pct: spec.min_area_pct,
        max_area_pct: spec.max_area_pct,
    })
}

#[pyfunction]
fn define(name: &str, spec: &str) -> PyResult<NativeOperation> {
    let spec: Spec = serde_json::from_str(spec).map_err(|e| {
        raise(
            SyrupError::new(Stage::Resolve, ErrorKind::Malformed, e.to_string())
                .for_operation(name),
        )
    })?;
    let intent = intent_from(spec).map_err(|e| raise(e.for_operation(name)))?;
    runtime()?
        .define(name, intent)
        .map(NativeOperation)
        .map_err(raise)
}

#[pyfunction]
fn decode_image(py: Python<'_>, path: PathBuf) -> PyResult<(Py<PyBytes>, u32, u32, u32)> {
    let image = py.detach(|| OwnedImage::open(&path)).map_err(raise)?;
    let data = PyBytes::new(py, &image.data).unbind();
    Ok((data, image.width, image.height, image.channels))
}

#[pyfunction]
fn cache_dir() -> PyResult<PathBuf> {
    Ok(runtime()?.store().root().to_path_buf())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NativeError", m.py().get_type::<NativeError>())?;
    m.add_class::<NativeOperation>()?;
    m.add_function(wrap_pyfunction!(resolve, m)?)?;
    m.add_function(wrap_pyfunction!(define, m)?)?;
    m.add_function(wrap_pyfunction!(decode_image, m)?)?;
    m.add_function(wrap_pyfunction!(cache_dir, m)?)?;
    Ok(())
}
