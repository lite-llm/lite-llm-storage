use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::checkpoint::{CheckpointManifest, ShardHashVerifier};
use crate::error::{StorageError, StorageResult};
use crate::types::TierId;

const MANIFEST_FILE: &str = "manifest.llmchk";

#[derive(Debug, Clone)]
pub struct SnapshotRepository {
    root: PathBuf,
}

#[derive(Debug, Clone)]
pub struct RestoredSnapshot {
    pub snapshot_id: String,
    pub manifest: CheckpointManifest,
    pub shard_bytes: BTreeMap<String, Vec<u8>>,
}

impl SnapshotRepository {
    pub fn new(root: impl Into<PathBuf>) -> StorageResult<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn commit_snapshot(
        &self,
        snapshot_id: &str,
        manifest: &CheckpointManifest,
        shard_bytes: &BTreeMap<String, Vec<u8>>,
        verifier: &dyn ShardHashVerifier,
    ) -> StorageResult<()> {
        if snapshot_id.trim().is_empty() {
            return Err(StorageError::InvalidConfig("snapshot_id must not be empty"));
        }

        manifest.validate()?;
        manifest.verify_shards(shard_bytes, verifier)?;

        let lock_path = self.lock_path(snapshot_id);
        let _lock = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&lock_path)
            .map_err(|err| {
                if err.kind() == std::io::ErrorKind::AlreadyExists {
                    StorageError::LockHeld
                } else {
                    StorageError::from(err)
                }
            })?;

        let temp_dir = self.temp_snapshot_dir(snapshot_id);
        let final_dir = self.snapshot_dir(snapshot_id);

        if temp_dir.exists() {
            fs::remove_dir_all(&temp_dir)?;
        }

        fs::create_dir_all(&temp_dir)?;

        let result = self.write_snapshot_contents(&temp_dir, manifest, shard_bytes, verifier)
            .and_then(|_| {
                if final_dir.exists() {
                    fs::remove_dir_all(&final_dir)?;
                }
                fs::rename(&temp_dir, &final_dir)?;
                Ok(())
            });

        if result.is_err() {
            let _ = fs::remove_dir_all(&temp_dir);
        }

        let _ = fs::remove_file(&lock_path);
        result
    }

    fn write_snapshot_contents(
        &self,
        dir: &Path,
        manifest: &CheckpointManifest,
        shard_bytes: &BTreeMap<String, Vec<u8>>,
        verifier: &dyn ShardHashVerifier,
    ) -> StorageResult<()> {
        for shard in &manifest.shards {
            let bytes = shard_bytes
                .get(&shard.path)
                .ok_or_else(|| StorageError::MissingShard(shard.path.clone()))?;

            if bytes.len() as u64 != shard.bytes {
                return Err(StorageError::InvalidManifest(
                    "shard byte length does not match manifest",
                ));
            }

            verifier.verify(
                &shard.path,
                &shard.hash_algorithm,
                &shard.checksum_hex,
                bytes,
            )?;

            let shard_path = path_from_manifest_path(dir, &shard.path);
            if let Some(parent) = shard_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(shard_path, bytes)?;
        }

        let manifest_path = dir.join(MANIFEST_FILE);
        let mut manifest_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(manifest_path)?;
        let canonical = manifest.to_canonical_string()?;
        manifest_file.write_all(canonical.as_bytes())?;
        manifest_file.flush()?;

        Ok(())
    }

    pub fn restore_snapshot(
        &self,
        snapshot_id: &str,
        selected_tiers: Option<&[TierId]>,
        verifier: &dyn ShardHashVerifier,
    ) -> StorageResult<RestoredSnapshot> {
        let snapshot_dir = self.snapshot_dir(snapshot_id);
        if !snapshot_dir.exists() {
            return Err(StorageError::SnapshotNotFound(snapshot_id.to_owned()));
        }

        let manifest_path = snapshot_dir.join(MANIFEST_FILE);
        if !manifest_path.exists() {
            return Err(StorageError::CorruptedSnapshot(
                "manifest file missing in snapshot directory",
            ));
        }

        let canonical = fs::read_to_string(manifest_path)?;
        let mut manifest = CheckpointManifest::from_canonical_string(&canonical)?;

        if let Some(tiers) = selected_tiers {
            manifest = manifest.filter_tiers(tiers);
            manifest.validate()?;
        }

        let mut loaded = BTreeMap::new();

        for shard in &manifest.shards {
            let shard_path = path_from_manifest_path(&snapshot_dir, &shard.path);
            if !shard_path.exists() {
                return Err(StorageError::MissingShard(shard.path.clone()));
            }

            let bytes = fs::read(&shard_path)?;
            if bytes.len() as u64 != shard.bytes {
                return Err(StorageError::CorruptedSnapshot(
                    "shard byte length differs from manifest",
                ));
            }

            verifier.verify(
                &shard.path,
                &shard.hash_algorithm,
                &shard.checksum_hex,
                &bytes,
            )?;

            loaded.insert(shard.path.clone(), bytes);
        }

        Ok(RestoredSnapshot {
            snapshot_id: snapshot_id.to_owned(),
            manifest,
            shard_bytes: loaded,
        })
    }

    pub fn list_snapshots(&self) -> StorageResult<Vec<String>> {
        let mut ids = Vec::new();

        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }

            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".tmp") {
                continue;
            }
            if name.starts_with('.') {
                continue;
            }
            if entry.path().join(MANIFEST_FILE).exists() {
                ids.push(name);
            }
        }

        ids.sort();
        Ok(ids)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn snapshot_dir(&self, snapshot_id: &str) -> PathBuf {
        self.root.join(snapshot_id)
    }

    fn temp_snapshot_dir(&self, snapshot_id: &str) -> PathBuf {
        self.root.join(format!("{snapshot_id}.tmp"))
    }

    fn lock_path(&self, snapshot_id: &str) -> PathBuf {
        self.root.join(format!("{snapshot_id}.lock"))
    }
}

fn path_from_manifest_path(root: &Path, manifest_path: &str) -> PathBuf {
    let mut path = root.to_path_buf();
    for part in manifest_path.split('/') {
        path.push(part);
    }
    path
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::checkpoint::{
        fnv64_hex, CheckpointManifest, Fnv64HashVerifier, OptimStateRef, RouterStateRef,
        ShardDescriptor, ShardType, TierManifestEntry,
    };
    use crate::placement::{PlacementPolicyKind, StorageTier};
    use crate::types::ExpertKey;

    use super::SnapshotRepository;

    fn unique_temp_root() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock must be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("lite-llm-storage-tests-{nanos}"))
    }

    fn sample_manifest() -> (CheckpointManifest, BTreeMap<String, Vec<u8>>) {
        let dense_bytes = b"dense-shard".to_vec();
        let expert_bytes = b"expert-shard".to_vec();

        let manifest = CheckpointManifest {
            model_id: "lite-llm-base".to_owned(),
            epoch: 1,
            step: 100,
            tiers: vec![
                TierManifestEntry {
                    tier_id: 1,
                    name: "hot".to_owned(),
                    size_budget_bytes: 1024 * 1024,
                    placement_policy: PlacementPolicyKind::Prioritized,
                    priority_score: 1,
                },
                TierManifestEntry {
                    tier_id: 2,
                    name: "warm".to_owned(),
                    size_budget_bytes: 4 * 1024 * 1024,
                    placement_policy: PlacementPolicyKind::Lru,
                    priority_score: 2,
                },
            ],
            shards: vec![
                ShardDescriptor {
                    path: "dense/l0/q.bin".to_owned(),
                    shard_type: ShardType::Dense,
                    tensor_name: Some("attention_q".to_owned()),
                    expert_key: None,
                    shape: vec![2, 5],
                    dtype: "bf16".to_owned(),
                    tier_hint: StorageTier::Hot,
                    checksum_hex: fnv64_hex(&dense_bytes),
                    hash_algorithm: "fnv64".to_owned(),
                    version: 1,
                    bytes: dense_bytes.len() as u64,
                },
                ShardDescriptor {
                    path: "experts/t2/g0/e1.bin".to_owned(),
                    shard_type: ShardType::Expert,
                    tensor_name: None,
                    expert_key: Some(ExpertKey::new(2, 0, 1)),
                    shape: vec![2, 6],
                    dtype: "bf16".to_owned(),
                    tier_hint: StorageTier::Warm,
                    checksum_hex: fnv64_hex(&expert_bytes),
                    hash_algorithm: "fnv64".to_owned(),
                    version: 2,
                    bytes: expert_bytes.len() as u64,
                },
            ],
            optim_state: Some(OptimStateRef {
                path: "optim/state.bin".to_owned(),
                checksum_hex: fnv64_hex(b"optim-state"),
                hash_algorithm: "fnv64".to_owned(),
                version: 1,
            }),
            router_state: Some(RouterStateRef {
                path: "router/state.bin".to_owned(),
                checksum_hex: fnv64_hex(b"router-state"),
                hash_algorithm: "fnv64".to_owned(),
                version: 1,
                base_seed: 42,
                layer_seeds: vec![],
            }),
            metadata_version: 1,
        };

        let mut bytes = BTreeMap::new();
        bytes.insert("dense/l0/q.bin".to_owned(), dense_bytes);
        bytes.insert("experts/t2/g0/e1.bin".to_owned(), expert_bytes);

        (manifest, bytes)
    }

    #[test]
    fn restore_roundtrip_is_consistent() {
        let root = unique_temp_root();
        let repo = SnapshotRepository::new(&root).expect("repo should initialize");
        let verifier = Fnv64HashVerifier;

        let (manifest, shards) = sample_manifest();
        repo.commit_snapshot("snap-1", &manifest, &shards, &verifier)
            .expect("snapshot commit should succeed");

        let restored = repo
            .restore_snapshot("snap-1", None, &verifier)
            .expect("snapshot restore should succeed");

        assert_eq!(restored.manifest, manifest);
        assert_eq!(restored.shard_bytes, shards);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn restore_detects_corrupted_shard() {
        let root = unique_temp_root();
        let repo = SnapshotRepository::new(&root).expect("repo should initialize");
        let verifier = Fnv64HashVerifier;

        let (manifest, shards) = sample_manifest();
        repo.commit_snapshot("snap-2", &manifest, &shards, &verifier)
            .expect("snapshot commit should succeed");

        let corrupted = root.join("snap-2").join("experts").join("t2").join("g0").join("e1.bin");
        std::fs::write(corrupted, b"CORRUPTED").expect("must write corruption");

        let restore = repo.restore_snapshot("snap-2", None, &verifier);
        assert!(restore.is_err());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn incomplete_tmp_snapshot_is_not_restorable() {
        let root = unique_temp_root();
        let repo = SnapshotRepository::new(&root).expect("repo should initialize");
        let verifier = Fnv64HashVerifier;

        std::fs::create_dir_all(root.join("snap-3.tmp")).expect("must create temp snapshot dir");

        let restore = repo.restore_snapshot("snap-3", None, &verifier);
        assert!(restore.is_err());

        let _ = std::fs::remove_dir_all(root);
    }
}
