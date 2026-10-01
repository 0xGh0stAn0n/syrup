use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use crate::abi::*;
use crate::cache::{self, Manifest, Store};
use crate::catalog;
use crate::codegen;
use crate::compiler::{HOST_TARGET, Rustc};
use crate::contract::*;
use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::host::{ExecHost, well_formed};
use crate::intent::{self, Intent, PixelRect};
use crate::loader::Module;
use crate::plan::Plan;
use crate::providers::{ModelSource, Providers};
use crate::validate;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// May generate and compile.
    Development,
    /// Only loads artifacts that were prepared ahead of time.
    Frozen,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub cache_dir: PathBuf,
    pub mode: Mode,
    pub rustc: Option<PathBuf>,
    pub compile_timeout: Duration,
    pub face_model: ModelSource,
}

impl Config {
    /// Reads `SYRUP_CACHE_DIR`, `SYRUP_MODE` (dev|frozen), `SYRUP_RUSTC`,
    /// `SYRUP_COMPILE_TIMEOUT_SECS` and `SYRUP_FACE_MODEL`.
    pub fn from_env() -> Result<Config> {
        let var = |name: &str| env::var_os(name).filter(|v| !v.is_empty());
        let bad = |name: &str, why: &str| {
            SyrupError::new(
                Stage::Input,
                ErrorKind::BadParameter,
                format!("{name} {why}"),
            )
        };
        let mode = match var("SYRUP_MODE").as_ref().and_then(|v| v.to_str()) {
            None | Some("dev" | "development") => Mode::Development,
            Some("frozen") => Mode::Frozen,
            Some(_) => return Err(bad("SYRUP_MODE", "must be dev or frozen")),
        };
        let compile_timeout = match var("SYRUP_COMPILE_TIMEOUT_SECS") {
            None => Duration::from_secs(120),
            Some(v) => v
                .to_str()
                .and_then(|v| v.parse().ok())
                .map(Duration::from_secs)
                .ok_or_else(|| {
                    bad(
                        "SYRUP_COMPILE_TIMEOUT_SECS",
                        "must be a whole number of seconds",
                    )
                })?,
        };
        Ok(Config {
            cache_dir: var("SYRUP_CACHE_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(default_cache_dir),
            mode,
            rustc: var("SYRUP_RUSTC").map(PathBuf::from),
            compile_timeout,
            face_model: var("SYRUP_FACE_MODEL")
                .map_or(ModelSource::Bundled, |p| ModelSource::File(p.into())),
        })
    }
}

fn default_cache_dir() -> PathBuf {
    let base = if let Some(dir) = env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(dir)
    } else if let Some(dir) = env::var_os("LOCALAPPDATA") {
        PathBuf::from(dir)
    } else if let Some(home) = env::home_dir() {
        if cfg!(target_os = "macos") {
            home.join("Library").join("Caches")
        } else {
            home.join(".cache")
        }
    } else {
        env::temp_dir()
    };
    base.join("syrup")
}

struct Loaded {
    module: Module,
    manifest: Manifest,
}

struct Inner {
    config: Config,
    store: Store,
    rustc: OnceLock<Result<Rustc>>,
    providers: Providers,
    loaded: Mutex<HashMap<String, Arc<Loaded>>>,
    building: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    defined: RwLock<HashMap<String, Intent>>,
}

#[derive(Clone)]
pub struct Runtime(Arc<Inner>);

impl Runtime {
    pub fn new(config: Config) -> Runtime {
        Runtime(Arc::new(Inner {
            store: Store::new(config.cache_dir.clone()),
            providers: Providers::new(config.face_model.clone()),
            config,
            rustc: OnceLock::new(),
            loaded: Mutex::default(),
            building: Mutex::default(),
            defined: RwLock::default(),
        }))
    }

    pub fn from_env() -> Result<Runtime> {
        Ok(Runtime::new(Config::from_env()?))
    }

    pub fn config(&self) -> &Config {
        &self.0.config
    }

    pub fn store(&self) -> &Store {
        &self.0.store
    }

    /// Names declared with [`Runtime::define`] win over the grammar.
    pub fn resolve(&self, name: &str) -> Result<Operation> {
        let defined = self
            .0
            .defined
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .copied();
        let intent = match defined {
            Some(intent) => intent,
            None => intent::parse(name)?,
        };
        self.operation(name, intent)
    }

    /// Gives `name` an explicit meaning, for names outside the grammar.
    pub fn define(&self, name: &str, intent: Intent) -> Result<Operation> {
        let conflict = |reason: String| {
            Err(SyrupError::new(Stage::Resolve, ErrorKind::Conflicting, reason).for_operation(name))
        };
        let identifier = name.bytes().next().is_some_and(|b| b.is_ascii_lowercase())
            && name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
        if !identifier {
            return Err(SyrupError::new(
                Stage::Resolve,
                ErrorKind::Malformed,
                "defined names are lowercase identifiers",
            )
            .for_operation(name));
        }
        intent.check().map_err(|e| e.for_operation(name))?;
        if let Ok(parsed) = intent::parse(name)
            && parsed != intent
        {
            return conflict(format!("{name} already means: {parsed}"));
        }
        let mut defined = self.0.defined.write().unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = defined.get(name)
            && *existing != intent
        {
            return conflict(format!("{name} is already defined as: {existing}"));
        }
        defined.insert(name.to_string(), intent);
        drop(defined);
        self.operation(name, intent)
    }

    fn operation(&self, name: &str, intent: Intent) -> Result<Operation> {
        let plan = Plan::build(&intent).map_err(|e| e.for_operation(name))?;
        let plan_hash = plan.hash();
        Ok(Operation {
            runtime: self.clone(),
            name: name.to_string(),
            key: cache::artifact_key(&plan_hash),
            intent,
            plan,
            plan_hash,
        })
    }
}

#[derive(Clone)]
pub struct Operation {
    runtime: Runtime,
    name: String,
    intent: Intent,
    plan: Plan,
    plan_hash: String,
    key: String,
}

#[derive(Debug, Clone)]
pub struct Prepared {
    pub status: ArtifactStatus,
    pub manifest: Manifest,
    pub library: PathBuf,
    pub elapsed: Duration,
}

impl Operation {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn intent(&self) -> &Intent {
        &self.intent
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    pub fn plan_hash(&self) -> &str {
        &self.plan_hash
    }

    pub fn artifact_key(&self) -> &str {
        &self.key
    }

    pub fn source(&self) -> String {
        codegen::generate(&self.plan, &self.intent, &self.plan_hash)
    }

    pub fn explain(&self) -> String {
        let artifact = match self.runtime.0.store.open(&self.key) {
            Ok(Some((_, library))) => format!("built ({})", library.display()),
            Ok(None) => "not built yet".to_string(),
            Err(e) => format!("damaged: {}", e.reason),
        };
        let steps: String = self
            .plan
            .describe()
            .lines()
            .map(|l| format!("    {l}\n"))
            .collect();
        format!(
            "{}\n  means: {}\n  plan {}:\n{steps}  artifact {}: {artifact}\n",
            self.name, self.intent, self.plan_hash, self.key
        )
    }

    /// Makes the compiled module available: from this process, from disk,
    /// or by generating, compiling and validating it.
    pub fn prepare(&self) -> Result<Prepared> {
        let started = Instant::now();
        let (loaded, status) = self
            .load_or_build()
            .map_err(|e| e.for_operation(&self.name))?;
        Ok(Prepared {
            status,
            manifest: loaded.manifest.clone(),
            library: loaded.module.path.clone(),
            elapsed: started.elapsed(),
        })
    }

    fn cached(&self) -> Option<Arc<Loaded>> {
        self.runtime
            .0
            .loaded
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&self.key)
            .cloned()
    }

    fn load_or_build(&self) -> Result<(Arc<Loaded>, ArtifactStatus)> {
        let inner = &self.runtime.0;
        if let Some(loaded) = self.cached() {
            return Ok((loaded, ArtifactStatus::InMemory));
        }
        let gate = inner
            .building
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(self.key.clone())
            .or_default()
            .clone();
        let _gate = gate.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(loaded) = self.cached() {
            return Ok((loaded, ArtifactStatus::InMemory));
        }
        let frozen = inner.config.mode == Mode::Frozen;
        // A frozen cache may be mounted read-only, so frozen mode never writes.
        let _lock = if frozen {
            None
        } else {
            Some(inner.store.lock(&self.key)?)
        };
        let (loaded, status) = match inner.store.open(&self.key) {
            Ok(Some((manifest, library))) => {
                let module = Module::load(&library, &self.plan_hash)?;
                (Loaded { module, manifest }, ArtifactStatus::LoadedFromDisk)
            }
            Ok(None) if frozen => {
                return Err(SyrupError::new(
                    Stage::Load,
                    ErrorKind::NotPrepared,
                    format!("no prepared artifact for {} and frozen mode never compiles", self.name),
                )
                .with_hint("run `syrup prepare` with the same cache directory where a compiler is available")
                .with_detail("artifact", self.key.clone()));
            }
            Err(e) if e.kind == ErrorKind::Integrity && !frozen => {
                inner.store.quarantine(&self.key)?;
                (self.build()?, ArtifactStatus::Compiled)
            }
            Err(e) => return Err(e),
            Ok(None) => (self.build()?, ArtifactStatus::Compiled),
        };
        let loaded = Arc::new(loaded);
        inner
            .loaded
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(self.key.clone(), loaded.clone());
        Ok((loaded, status))
    }

    fn build(&self) -> Result<Loaded> {
        let inner = &self.runtime.0;
        let rustc = inner
            .rustc
            .get_or_init(|| Rustc::find(inner.config.rustc.as_deref()))
            .clone()?;
        let staging = inner.store.staging(&self.key)?;
        match self.build_in(&staging, &rustc) {
            Ok(manifest) => {
                let dir = inner.store.publish(&staging, &self.key)?;
                let module = Module::load(&dir.join(&manifest.library), &self.plan_hash)?;
                Ok(Loaded { module, manifest })
            }
            Err(e) => {
                let kept = inner.store.keep_failure(&staging, &self.key);
                Err(e.with_detail("kept", kept.display().to_string()))
            }
        }
    }

    fn build_in(&self, dir: &Path, rustc: &Rustc) -> Result<Manifest> {
        let started = Instant::now();
        let write = |name: &str, bytes: &[u8]| {
            let path = dir.join(name);
            fs::write(&path, bytes)
                .map_err(|e| SyrupError::io(Stage::Generate, "cannot write", &path, e))
        };
        let source = self.source();
        write("op.rs", source.as_bytes())?;
        write(
            "plan.json",
            &serde_json::to_vec_pretty(&self.plan).expect("plans serialize"),
        )?;
        codegen::check_policy(&source)?;

        let crate_name = format!("syrup_op_{}", &self.key[..16]);
        let library = format!(
            "{}{crate_name}{}",
            env::consts::DLL_PREFIX,
            env::consts::DLL_SUFFIX
        );
        rustc.compile(
            &dir.join("op.rs"),
            &dir.join(&library),
            &crate_name,
            self.runtime.0.config.compile_timeout,
        )?;

        // Unloaded again before publishing, so the directory can be renamed.
        let cases = validate::check(
            &Module::load(&dir.join(&library), &self.plan_hash)?,
            &self.plan,
        )?;

        let hash_of = |name: &str| {
            let path = dir.join(name);
            cache::sha256_file(&path)
                .map_err(|e| SyrupError::io(Stage::Compile, "cannot read", &path, e))
        };
        let manifest = Manifest {
            key: self.key.clone(),
            intent: self.intent.to_string(),
            plan_hash: self.plan_hash.clone(),
            generator: format!(
                "syrup-runtime {} codegen {}",
                env!("CARGO_PKG_VERSION"),
                codegen::CODEGEN_VERSION
            ),
            abi_version: SYRUP_ABI_VERSION,
            target: HOST_TARGET.to_string(),
            rustc: rustc.version.clone(),
            source_sha256: hash_of("op.rs")?,
            library_sha256: hash_of(&library)?,
            library,
            built_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            build_ms: started.elapsed().as_millis() as u64,
            validation_cases: cases,
        };
        write(
            "manifest.json",
            &serde_json::to_vec_pretty(&manifest).expect("manifests serialize"),
        )?;
        Ok(manifest)
    }

    pub fn run(&self, image: &ImageInput<'_>, params: &RunParams) -> Result<FindResult> {
        self.run_inner(image, params)
            .map_err(|e| e.for_operation(&self.name))
    }

    fn run_inner(&self, image: &ImageInput<'_>, params: &RunParams) -> Result<FindResult> {
        let entry = catalog::entry(self.intent.target);
        let bad = |reason: String| {
            Err(SyrupError::new(
                Stage::Input,
                ErrorKind::BadParameter,
                reason,
            ))
        };
        let min_confidence = params
            .min_confidence
            .unwrap_or(entry.default_min_confidence);
        if !(0.0..=1.0).contains(&min_confidence) {
            return bad(format!(
                "min_confidence must be between 0 and 1, got {min_confidence}"
            ));
        }
        if params.max_results == Some(0) {
            return bad("max_results must be at least 1".into());
        }
        let region = match (self.plan.needs_caller_region(), params.region) {
            (true, None) => return bad("this operation needs a region (x, y, w, h)".into()),
            (false, Some(_)) => {
                return bad("this operation takes no region; use an _in_region operation".into());
            }
            (false, None) => None,
            (true, Some(r)) => {
                if r.w == 0 || r.h == 0 || r.x >= image.width() || r.y >= image.height() {
                    return bad(format!(
                        "region {r:?} does not overlap the {}x{} image",
                        image.width(),
                        image.height()
                    ));
                }
                Some(PixelRect {
                    x: r.x,
                    y: r.y,
                    w: r.w.min(image.width() - r.x),
                    h: r.h.min(image.height() - r.y),
                })
            }
        };

        let prepared_at = Instant::now();
        let (loaded, status) = self.load_or_build()?;
        let prepare_ms = prepared_at.elapsed().as_secs_f64() * 1000.0;

        let abi_params = SyrupParams {
            min_confidence,
            max_results: params.max_results.unwrap_or(0),
            has_region: region.is_some() as u32,
            region_x: region.map_or(0, |r| r.x),
            region_y: region.map_or(0, |r| r.y),
            region_w: region.map_or(0, |r| r.w),
            region_h: region.map_or(0, |r| r.h),
        };
        let mut host = ExecHost::new(*image, &self.runtime.0.providers);
        let table = host.table();
        let executed_at = Instant::now();
        // SAFETY: `table` points at `host`, which stays in place until the
        // call returns, and `image` outlives it.
        let code = unsafe { loaded.module.run(&table, &image.as_abi(), &abi_params) };
        let execute_ms = executed_at.elapsed().as_secs_f64() * 1000.0;
        // A provider failure stands even if the module ignored it.
        if let Some(e) = host.error.take() {
            return Err(e);
        }
        if code != SYRUP_OK {
            return Err(module_error(code));
        }

        let limit = [self.intent.limit, params.max_results]
            .into_iter()
            .flatten()
            .min();
        // x + w can round a hair past the edge it was clipped to.
        let (w, h) = (image.width() as f32 + 1e-3, image.height() as f32 + 1e-3);
        let inside = |d: &SyrupDetection| {
            let n = d.n_keypoints as usize;
            well_formed(d)
                && d.w > 0.0
                && d.h > 0.0
                && d.x >= 0.0
                && d.y >= 0.0
                && d.x + d.w <= w
                && d.y + d.h <= h
                && d.score >= min_confidence
                && d.payload as usize <= host.texts.len()
                && d.keypoints[..2 * n]
                    .chunks(2)
                    .all(|k| (0.0..=w).contains(&k[0]) && (0.0..=h).contains(&k[1]))
        };
        if let Some(bad) = host.out.iter().find(|d| !inside(d)) {
            return Err(SyrupError::new(
                Stage::Execute,
                ErrorKind::ContractViolation,
                format!("the module emitted a result that breaks the contract: {bad:?}"),
            ));
        }
        if limit.is_some_and(|limit| host.out.len() > limit as usize) {
            return Err(SyrupError::new(
                Stage::Execute,
                ErrorKind::ContractViolation,
                format!(
                    "the module emitted {} results, more than its limit",
                    host.out.len()
                ),
            ));
        }

        let items = host
            .out
            .iter()
            .map(|d| Found {
                label: entry.label,
                bbox: BoxF {
                    x: d.x,
                    y: d.y,
                    w: d.w,
                    h: d.h,
                },
                confidence: d.score,
                keypoints: (0..d.n_keypoints as usize)
                    .map(|k| Keypoint {
                        name: entry.keypoints[k],
                        x: d.keypoints[2 * k],
                        y: d.keypoints[2 * k + 1],
                    })
                    .collect(),
                text: (d.payload > 0).then(|| host.texts[d.payload as usize - 1].clone()),
            })
            .collect();
        let manifest = &loaded.manifest;
        Ok(FindResult {
            items,
            provenance: Provenance {
                operation: self.name.clone(),
                intent: self.intent.to_string(),
                plan_hash: self.plan_hash.clone(),
                artifact: ArtifactRef {
                    key: self.key.clone(),
                    path: loaded.module.path.display().to_string(),
                    binary_sha256: manifest.library_sha256.clone(),
                    rustc: manifest.rustc.clone(),
                    target: manifest.target.clone(),
                    generator: manifest.generator.clone(),
                },
                artifact_status: status,
                providers: self
                    .plan
                    .capabilities()
                    .into_iter()
                    .filter_map(|c| self.runtime.0.providers.get(c).ok())
                    .map(|p| p.info())
                    .collect(),
                params: EffectiveParams {
                    min_confidence,
                    max_results: params.max_results,
                    region,
                },
                image: (image.width(), image.height(), image.channels()),
                prepare_ms,
                execute_ms,
            },
        })
    }
}

fn module_error(code: i32) -> SyrupError {
    let (stage, kind, reason) = match code {
        SYRUP_ERR_PANIC => (
            Stage::Execute,
            ErrorKind::Panic,
            "the generated module panicked".to_string(),
        ),
        SYRUP_ERR_ABI => (
            Stage::Load,
            ErrorKind::AbiMismatch,
            "the module rejected the host's ABI".to_string(),
        ),
        SYRUP_ERR_INPUT => (
            Stage::Execute,
            ErrorKind::ModuleFailed,
            "the module rejected its input".to_string(),
        ),
        code => (
            Stage::Execute,
            ErrorKind::ModuleFailed,
            format!("the module failed with status {code}"),
        ),
    };
    SyrupError::new(stage, kind, reason)
}
