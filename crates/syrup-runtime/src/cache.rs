//! Artifact store. Artifacts are built in a private staging directory and
//! published with a single rename, so readers only ever see complete ones.
//! Published artifacts are never modified; damaged ones are moved aside.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::abi::SYRUP_ABI_VERSION;
use crate::codegen::CODEGEN_VERSION;
use crate::compiler::{FLAGS, HOST_TARGET};
use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::plan::hex;

const LAYOUT: &str = "v1";

/// Everything that decides whether a compiled module can be reused. The
/// operation name is deliberately absent, and so is the rustc version: the
/// module only talks C ABI, so any compiler's output stays valid.
pub fn artifact_key(plan_hash: &str) -> String {
    let identity = format!(
        "plan={plan_hash};codegen={CODEGEN_VERSION};abi={SYRUP_ABI_VERSION};target={HOST_TARGET};flags={}",
        FLAGS.join(" ")
    );
    hex(&Sha256::digest(identity))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub key: String,
    pub intent: String,
    pub plan_hash: String,
    pub generator: String,
    pub abi_version: u32,
    pub target: String,
    pub rustc: String,
    pub source_sha256: String,
    pub library: String,
    pub library_sha256: String,
    pub built_at: u64,
    pub build_ms: u64,
    pub validation_cases: u32,
}

pub struct Store {
    root: PathBuf,
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    Ok(hex(&Sha256::digest(fs::read(path)?)))
}

fn now() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
}

fn integrity(reason: String) -> SyrupError {
    SyrupError::new(Stage::Load, ErrorKind::Integrity, reason)
        .with_hint("the damaged artifact is moved aside and rebuilt in development mode; in frozen mode, prepare it again")
}

impl Store {
    pub fn new(root: PathBuf) -> Store {
        Store { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn artifact_dir(&self, key: &str) -> PathBuf {
        self.root.join(LAYOUT).join(key)
    }

    pub fn failed_dir(&self, key: &str) -> PathBuf {
        self.root.join("failed").join(key)
    }

    fn create(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir).map_err(|e| SyrupError::io(Stage::Compile, "cannot create", dir, e))
    }

    /// Exclusive across processes and threads; released when dropped.
    pub fn lock(&self, key: &str) -> Result<File> {
        let dir = self.root.join("locks");
        self.create(&dir)?;
        let path = dir.join(format!("{key}.lock"));
        let file = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(|e| SyrupError::io(Stage::Compile, "cannot open", &path, e))?;
        file.lock()
            .map_err(|e| SyrupError::io(Stage::Compile, "cannot lock", &path, e))?;
        Ok(file)
    }

    pub fn staging(&self, key: &str) -> Result<PathBuf> {
        let dir = self.root.join("staging").join(format!(
            "{key}.{}.{}",
            std::process::id(),
            now().as_nanos()
        ));
        self.create(&dir)?;
        Ok(dir)
    }

    /// The published artifact for `key`, if any, after checking it is intact.
    pub fn open(&self, key: &str) -> Result<Option<(Manifest, PathBuf)>> {
        let dir = self.artifact_dir(key);
        let manifest_path = dir.join("manifest.json");
        let text = match fs::read_to_string(&manifest_path) {
            Ok(text) => text,
            Err(e) if e.kind() == io::ErrorKind::NotFound && !dir.exists() => return Ok(None),
            Err(e) => {
                return Err(integrity(format!(
                    "cannot read {}: {e}",
                    manifest_path.display()
                )));
            }
        };
        let manifest: Manifest = serde_json::from_str(&text).map_err(|e| {
            integrity(format!(
                "{} is not a valid manifest: {e}",
                manifest_path.display()
            ))
        })?;
        if manifest.key != key || manifest.abi_version != SYRUP_ABI_VERSION {
            return Err(integrity(format!(
                "{} does not describe artifact {key}",
                manifest_path.display()
            )));
        }
        let library = dir.join(&manifest.library);
        let actual = sha256_file(&library)
            .map_err(|e| integrity(format!("cannot read {}: {e}", library.display())))?;
        if actual != manifest.library_sha256 {
            return Err(integrity(format!(
                "{} does not match its manifest (sha256 {actual}, expected {})",
                library.display(),
                manifest.library_sha256
            ))
            .with_detail("library", library.display().to_string()));
        }
        Ok(Some((manifest, library)))
    }

    /// Returns the published directory. Losing a race to another builder is
    /// fine: both built the same thing, and the first one wins.
    pub fn publish(&self, staging: &Path, key: &str) -> Result<PathBuf> {
        let target = self.artifact_dir(key);
        self.create(target.parent().expect("artifact dirs have a parent"))?;
        let mut attempt = 0;
        loop {
            match fs::rename(staging, &target) {
                Ok(()) => return Ok(target),
                Err(_) if target.join("manifest.json").exists() => {
                    let _ = fs::remove_dir_all(staging);
                    return Ok(target);
                }
                // Windows can hold a just-unloaded library for a moment.
                Err(_) if attempt < 20 => {
                    attempt += 1;
                    thread::sleep(Duration::from_millis(25));
                }
                Err(e) => return Err(SyrupError::io(Stage::Compile, "cannot publish", &target, e)),
            }
        }
    }

    /// Keeps a failed build's source and logs for inspection.
    pub fn keep_failure(&self, staging: &Path, key: &str) -> PathBuf {
        let target = self.failed_dir(key);
        let _ = fs::remove_dir_all(&target);
        let _ = fs::create_dir_all(target.parent().expect("failed dirs have a parent"));
        if fs::rename(staging, &target).is_err() {
            let _ = fs::remove_dir_all(staging);
        }
        target
    }

    pub fn quarantine(&self, key: &str) -> Result<()> {
        let target = self
            .root
            .join("failed")
            .join(format!("{key}.damaged.{}", now().as_nanos()));
        self.create(target.parent().expect("failed dirs have a parent"))?;
        let dir = self.artifact_dir(key);
        fs::rename(&dir, &target)
            .map_err(|e| SyrupError::io(Stage::Load, "cannot move aside", &dir, e))
    }

    pub fn list(&self) -> Vec<Manifest> {
        let Ok(entries) = fs::read_dir(self.root.join(LAYOUT)) else {
            return vec![];
        };
        let mut manifests: Vec<Manifest> = entries
            .flatten()
            .filter_map(|entry| fs::read_to_string(entry.path().join("manifest.json")).ok())
            .filter_map(|text| serde_json::from_str(&text).ok())
            .collect();
        manifests.sort_by_key(|m| m.built_at);
        manifests
    }

    pub fn clear(&self) -> Result<()> {
        for dir in [LAYOUT, "staging", "failed", "locks"] {
            let path = self.root.join(dir);
            match fs::remove_dir_all(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(SyrupError::io(Stage::Load, "cannot remove", &path, e)),
            }
        }
        Ok(())
    }
}
