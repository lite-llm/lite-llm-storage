use std::collections::{BTreeMap, BTreeSet};

use crate::error::{StorageError, StorageResult};
use crate::placement::{PlacementPolicyKind, StorageTier};
use crate::types::{ExpertKey, TierId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardType {
    Dense,
    Expert,
}

impl ShardType {
    fn parse(value: &str) -> StorageResult<Self> {
        match value {
            "dense" => Ok(Self::Dense),
            "expert" => Ok(Self::Expert),
            _ => Err(StorageError::ParseError("unknown shard type")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Expert => "expert",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierManifestEntry {
    pub tier_id: TierId,
    pub name: String,
    pub size_budget_bytes: u64,
    pub placement_policy: PlacementPolicyKind,
    pub priority_score: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardDescriptor {
    pub path: String,
    pub shard_type: ShardType,
    pub tensor_name: Option<String>,
    pub expert_key: Option<ExpertKey>,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub tier_hint: StorageTier,
    pub checksum_hex: String,
    pub hash_algorithm: String,
    pub version: u32,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimStateRef {
    pub path: String,
    pub checksum_hex: String,
    pub hash_algorithm: String,
    pub version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterSeedRef {
    pub layer: u32,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterStateRef {
    pub path: String,
    pub checksum_hex: String,
    pub hash_algorithm: String,
    pub version: u32,
    pub base_seed: u64,
    pub layer_seeds: Vec<RouterSeedRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointManifest {
    pub model_id: String,
    pub epoch: u64,
    pub step: u64,
    pub tiers: Vec<TierManifestEntry>,
    pub shards: Vec<ShardDescriptor>,
    pub optim_state: Option<OptimStateRef>,
    pub router_state: Option<RouterStateRef>,
    pub metadata_version: u32,
}

pub trait ShardHashVerifier {
    fn verify(
        &self,
        path: &str,
        algorithm: &str,
        expected_hex: &str,
        bytes: &[u8],
    ) -> StorageResult<()>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Fnv64HashVerifier;

impl ShardHashVerifier for Fnv64HashVerifier {
    fn verify(
        &self,
        path: &str,
        algorithm: &str,
        expected_hex: &str,
        bytes: &[u8],
    ) -> StorageResult<()> {
        if algorithm != "fnv64" {
            return Err(StorageError::InvalidManifest(
                "unsupported hash algorithm for built-in verifier",
            ));
        }

        let actual = fnv64_hex(bytes);
        if actual != expected_hex {
            return Err(StorageError::ChecksumMismatch {
                path: path.to_owned(),
                expected: expected_hex.to_owned(),
                actual,
            });
        }

        Ok(())
    }
}

impl CheckpointManifest {
    pub fn validate(&self) -> StorageResult<()> {
        if self.metadata_version == 0 {
            return Err(StorageError::InvalidManifest(
                "metadata_version must be greater than zero",
            ));
        }
        if self.model_id.trim().is_empty() {
            return Err(StorageError::InvalidManifest("model_id is required"));
        }
        if self.tiers.is_empty() {
            return Err(StorageError::InvalidManifest("tiers must not be empty"));
        }
        if self.shards.is_empty() {
            return Err(StorageError::InvalidManifest("shards must not be empty"));
        }

        let mut tier_ids = BTreeSet::new();
        for tier in &self.tiers {
            if !tier_ids.insert(tier.tier_id) {
                return Err(StorageError::InvalidManifest("duplicate tier_id in tiers"));
            }
            if tier.name.trim().is_empty() {
                return Err(StorageError::InvalidManifest("tier name must not be empty"));
            }
            if tier.size_budget_bytes == 0 {
                return Err(StorageError::InvalidManifest(
                    "tier size budget must be greater than zero",
                ));
            }
        }

        for shard in &self.shards {
            if shard.path.trim().is_empty() {
                return Err(StorageError::InvalidManifest("shard path is required"));
            }
            if shard.shape.is_empty() {
                return Err(StorageError::InvalidManifest(
                    "shard shape must contain at least one dimension",
                ));
            }
            if shard.dtype.trim().is_empty() {
                return Err(StorageError::InvalidManifest("shard dtype is required"));
            }
            if shard.bytes == 0 {
                return Err(StorageError::InvalidManifest(
                    "shard bytes must be greater than zero",
                ));
            }
            if shard.checksum_hex.trim().is_empty() || shard.hash_algorithm.trim().is_empty() {
                return Err(StorageError::InvalidManifest(
                    "shard checksum and hash algorithm are required",
                ));
            }

            match shard.shard_type {
                ShardType::Dense => {
                    if shard.tensor_name.as_deref().unwrap_or("").trim().is_empty() {
                        return Err(StorageError::InvalidManifest(
                            "dense shard requires tensor_name",
                        ));
                    }
                    if shard.expert_key.is_some() {
                        return Err(StorageError::InvalidManifest(
                            "dense shard must not include expert_key",
                        ));
                    }
                }
                ShardType::Expert => {
                    let expert = shard.expert_key.ok_or(StorageError::InvalidManifest(
                        "expert shard requires expert_key",
                    ))?;
                    if !tier_ids.contains(&expert.tier) {
                        return Err(StorageError::TierUnknown(expert.tier));
                    }
                }
            }
        }

        if let Some(optim) = &self.optim_state {
            if optim.path.trim().is_empty()
                || optim.checksum_hex.trim().is_empty()
                || optim.hash_algorithm.trim().is_empty()
            {
                return Err(StorageError::InvalidManifest(
                    "optim_state fields must be populated",
                ));
            }
        }

        if let Some(router) = &self.router_state {
            if router.path.trim().is_empty()
                || router.checksum_hex.trim().is_empty()
                || router.hash_algorithm.trim().is_empty()
            {
                return Err(StorageError::InvalidManifest(
                    "router_state fields must be populated",
                ));
            }
        }

        Ok(())
    }

    pub fn verify_shards(
        &self,
        shard_bytes: &BTreeMap<String, Vec<u8>>,
        verifier: &dyn ShardHashVerifier,
    ) -> StorageResult<()> {
        self.validate()?;

        for shard in &self.shards {
            let bytes = shard_bytes
                .get(&shard.path)
                .ok_or_else(|| StorageError::MissingShard(shard.path.clone()))?;

            verifier.verify(
                &shard.path,
                &shard.hash_algorithm,
                &shard.checksum_hex,
                bytes,
            )?;
        }

        Ok(())
    }

    pub fn filter_tiers(&self, tiers: &[TierId]) -> Self {
        let selected: BTreeSet<TierId> = tiers.iter().copied().collect();

        let filtered_tiers: Vec<TierManifestEntry> = self
            .tiers
            .iter()
            .filter(|tier| selected.contains(&tier.tier_id))
            .cloned()
            .collect();

        let filtered_shards: Vec<ShardDescriptor> = self
            .shards
            .iter()
            .filter(|shard| match shard.shard_type {
                ShardType::Dense => true,
                ShardType::Expert => shard
                    .expert_key
                    .map(|key| selected.contains(&key.tier))
                    .unwrap_or(false),
            })
            .cloned()
            .collect();

        Self {
            model_id: self.model_id.clone(),
            epoch: self.epoch,
            step: self.step,
            tiers: filtered_tiers,
            shards: filtered_shards,
            optim_state: self.optim_state.clone(),
            router_state: self.router_state.clone(),
            metadata_version: self.metadata_version,
        }
    }

    pub fn to_canonical_string(&self) -> StorageResult<String> {
        self.validate()?;

        let mut out = String::new();
        out.push_str(&format!("metadata_version={}\n", self.metadata_version));
        out.push_str(&format!("model_id={}\n", escape(&self.model_id)));
        out.push_str(&format!("epoch={}\n", self.epoch));
        out.push_str(&format!("step={}\n", self.step));

        for tier in &self.tiers {
            out.push_str(&format!(
                "tier|{}|{}|{}|{}|{}\n",
                tier.tier_id,
                escape(&tier.name),
                tier.size_budget_bytes,
                placement_kind_to_str(tier.placement_policy),
                tier.priority_score
            ));
        }

        for shard in &self.shards {
            let tensor_name = shard.tensor_name.clone().unwrap_or_else(|| "-".to_owned());
            let expert_key = shard
                .expert_key
                .map(|key| key.encode())
                .unwrap_or_else(|| "-".to_owned());
            let shape = shard
                .shape
                .iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<String>>()
                .join(",");

            out.push_str(&format!(
                "shard|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n",
                escape(&shard.path),
                shard.shard_type.as_str(),
                escape(&tensor_name),
                expert_key,
                shape,
                escape(&shard.dtype),
                storage_tier_to_str(shard.tier_hint),
                shard.checksum_hex,
                shard.hash_algorithm,
                shard.version,
                shard.bytes
            ));
        }

        if let Some(optim) = &self.optim_state {
            out.push_str(&format!(
                "optim|{}|{}|{}|{}\n",
                escape(&optim.path),
                optim.checksum_hex,
                optim.hash_algorithm,
                optim.version
            ));
        }

        if let Some(router) = &self.router_state {
            out.push_str(&format!(
                "router|{}|{}|{}|{}|{}\n",
                escape(&router.path),
                router.checksum_hex,
                router.hash_algorithm,
                router.version,
                router.base_seed
            ));

            for seed in &router.layer_seeds {
                out.push_str(&format!("router_seed|{}|{}\n", seed.layer, seed.seed));
            }
        }

        out.push_str("end\n");
        Ok(out)
    }

    pub fn from_canonical_string(value: &str) -> StorageResult<Self> {
        let mut model_id = None;
        let mut epoch = None;
        let mut step = None;
        let mut metadata_version = None;
        let mut tiers = Vec::new();
        let mut shards = Vec::new();
        let mut optim_state = None;
        let mut router_state: Option<RouterStateRef> = None;

        for raw in value.lines() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if line == "end" {
                break;
            }

            if let Some(rest) = line.strip_prefix("metadata_version=") {
                metadata_version = Some(
                    rest.parse::<u32>()
                        .map_err(|_| StorageError::ParseError("invalid metadata_version"))?,
                );
                continue;
            }
            if let Some(rest) = line.strip_prefix("model_id=") {
                model_id = Some(unescape(rest));
                continue;
            }
            if let Some(rest) = line.strip_prefix("epoch=") {
                epoch = Some(
                    rest.parse::<u64>()
                        .map_err(|_| StorageError::ParseError("invalid epoch"))?,
                );
                continue;
            }
            if let Some(rest) = line.strip_prefix("step=") {
                step = Some(
                    rest.parse::<u64>()
                        .map_err(|_| StorageError::ParseError("invalid step"))?,
                );
                continue;
            }

            let mut parts = line.split('|');
            let tag = parts
                .next()
                .ok_or(StorageError::ParseError("missing line tag"))?;

            match tag {
                "tier" => {
                    let tier_id = parts
                        .next()
                        .ok_or(StorageError::ParseError("tier_id missing"))?
                        .parse::<u16>()
                        .map_err(|_| StorageError::ParseError("invalid tier_id"))?;
                    let name = unescape(
                        parts
                            .next()
                            .ok_or(StorageError::ParseError("tier name missing"))?,
                    );
                    let size_budget_bytes = parts
                        .next()
                        .ok_or(StorageError::ParseError("tier size missing"))?
                        .parse::<u64>()
                        .map_err(|_| StorageError::ParseError("invalid tier size"))?;
                    let placement_policy = str_to_placement_kind(
                        parts
                            .next()
                            .ok_or(StorageError::ParseError("placement policy missing"))?,
                    )?;
                    let priority_score = parts
                        .next()
                        .ok_or(StorageError::ParseError("tier priority missing"))?
                        .parse::<u32>()
                        .map_err(|_| StorageError::ParseError("invalid tier priority"))?;

                    tiers.push(TierManifestEntry {
                        tier_id,
                        name,
                        size_budget_bytes,
                        placement_policy,
                        priority_score,
                    });
                }
                "shard" => {
                    let path = unescape(
                        parts
                            .next()
                            .ok_or(StorageError::ParseError("shard path missing"))?,
                    );
                    let shard_type = ShardType::parse(
                        parts
                            .next()
                            .ok_or(StorageError::ParseError("shard type missing"))?,
                    )?;
                    let tensor_name_raw = unescape(
                        parts
                            .next()
                            .ok_or(StorageError::ParseError("tensor_name missing"))?,
                    );
                    let expert_key_raw = parts
                        .next()
                        .ok_or(StorageError::ParseError("expert_key missing"))?;
                    let shape_raw = parts
                        .next()
                        .ok_or(StorageError::ParseError("shape missing"))?;
                    let dtype = unescape(
                        parts
                            .next()
                            .ok_or(StorageError::ParseError("dtype missing"))?,
                    );
                    let tier_hint = str_to_storage_tier(
                        parts
                            .next()
                            .ok_or(StorageError::ParseError("tier_hint missing"))?,
                    )?;
                    let checksum_hex = parts
                        .next()
                        .ok_or(StorageError::ParseError("checksum missing"))?
                        .to_owned();
                    let hash_algorithm = parts
                        .next()
                        .ok_or(StorageError::ParseError("hash_algorithm missing"))?
                        .to_owned();
                    let version = parts
                        .next()
                        .ok_or(StorageError::ParseError("version missing"))?
                        .parse::<u32>()
                        .map_err(|_| StorageError::ParseError("invalid shard version"))?;
                    let bytes = parts
                        .next()
                        .ok_or(StorageError::ParseError("bytes missing"))?
                        .parse::<u64>()
                        .map_err(|_| StorageError::ParseError("invalid shard bytes"))?;

                    let shape = if shape_raw.is_empty() {
                        Vec::new()
                    } else {
                        let mut parsed = Vec::new();
                        for dim in shape_raw.split(',') {
                            parsed.push(dim.parse::<usize>().map_err(|_| {
                                StorageError::ParseError("invalid shape dimension")
                            })?);
                        }
                        parsed
                    };

                    let tensor_name = if tensor_name_raw == "-" {
                        None
                    } else {
                        Some(tensor_name_raw)
                    };
                    let expert_key = if expert_key_raw == "-" {
                        None
                    } else {
                        Some(
                            ExpertKey::parse(expert_key_raw)
                                .ok_or(StorageError::ParseError("invalid expert key"))?,
                        )
                    };

                    shards.push(ShardDescriptor {
                        path,
                        shard_type,
                        tensor_name,
                        expert_key,
                        shape,
                        dtype,
                        tier_hint,
                        checksum_hex,
                        hash_algorithm,
                        version,
                        bytes,
                    });
                }
                "optim" => {
                    let path = unescape(
                        parts
                            .next()
                            .ok_or(StorageError::ParseError("optim path missing"))?,
                    );
                    let checksum_hex = parts
                        .next()
                        .ok_or(StorageError::ParseError("optim checksum missing"))?
                        .to_owned();
                    let hash_algorithm = parts
                        .next()
                        .ok_or(StorageError::ParseError("optim hash_algorithm missing"))?
                        .to_owned();
                    let version = parts
                        .next()
                        .ok_or(StorageError::ParseError("optim version missing"))?
                        .parse::<u32>()
                        .map_err(|_| StorageError::ParseError("invalid optim version"))?;
                    optim_state = Some(OptimStateRef {
                        path,
                        checksum_hex,
                        hash_algorithm,
                        version,
                    });
                }
                "router" => {
                    let path = unescape(
                        parts
                            .next()
                            .ok_or(StorageError::ParseError("router path missing"))?,
                    );
                    let checksum_hex = parts
                        .next()
                        .ok_or(StorageError::ParseError("router checksum missing"))?
                        .to_owned();
                    let hash_algorithm = parts
                        .next()
                        .ok_or(StorageError::ParseError("router hash_algorithm missing"))?
                        .to_owned();
                    let version = parts
                        .next()
                        .ok_or(StorageError::ParseError("router version missing"))?
                        .parse::<u32>()
                        .map_err(|_| StorageError::ParseError("invalid router version"))?;
                    let base_seed = parts
                        .next()
                        .ok_or(StorageError::ParseError("router base seed missing"))?
                        .parse::<u64>()
                        .map_err(|_| StorageError::ParseError("invalid router base seed"))?;

                    router_state = Some(RouterStateRef {
                        path,
                        checksum_hex,
                        hash_algorithm,
                        version,
                        base_seed,
                        layer_seeds: Vec::new(),
                    });
                }
                "router_seed" => {
                    let layer = parts
                        .next()
                        .ok_or(StorageError::ParseError("router_seed layer missing"))?
                        .parse::<u32>()
                        .map_err(|_| StorageError::ParseError("invalid router_seed layer"))?;
                    let seed = parts
                        .next()
                        .ok_or(StorageError::ParseError("router_seed value missing"))?
                        .parse::<u64>()
                        .map_err(|_| StorageError::ParseError("invalid router_seed value"))?;

                    let router = router_state
                        .as_mut()
                        .ok_or(StorageError::ParseError("router_seed without router line"))?;
                    router.layer_seeds.push(RouterSeedRef { layer, seed });
                }
                _ => {
                    // Forward compatibility: ignore unknown lines.
                }
            }
        }

        let manifest = Self {
            model_id: model_id.ok_or(StorageError::ParseError("missing model_id"))?,
            epoch: epoch.ok_or(StorageError::ParseError("missing epoch"))?,
            step: step.ok_or(StorageError::ParseError("missing step"))?,
            tiers,
            shards,
            optim_state,
            router_state,
            metadata_version: metadata_version
                .ok_or(StorageError::ParseError("missing metadata_version"))?,
        };

        manifest.validate()?;
        Ok(manifest)
    }
}

pub fn fnv64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn storage_tier_to_str(tier: StorageTier) -> &'static str {
    match tier {
        StorageTier::Hot => "hot",
        StorageTier::Warm => "warm",
        StorageTier::Cold => "cold",
        StorageTier::Archive => "archive",
    }
}

fn str_to_storage_tier(value: &str) -> StorageResult<StorageTier> {
    match value {
        "hot" => Ok(StorageTier::Hot),
        "warm" => Ok(StorageTier::Warm),
        "cold" => Ok(StorageTier::Cold),
        "archive" => Ok(StorageTier::Archive),
        _ => Err(StorageError::ParseError("invalid storage tier")),
    }
}

fn placement_kind_to_str(kind: PlacementPolicyKind) -> &'static str {
    match kind {
        PlacementPolicyKind::Prioritized => "prioritized",
        PlacementPolicyKind::Lru => "lru",
        PlacementPolicyKind::Static => "static",
    }
}

fn str_to_placement_kind(value: &str) -> StorageResult<PlacementPolicyKind> {
    match value {
        "prioritized" => Ok(PlacementPolicyKind::Prioritized),
        "lru" => Ok(PlacementPolicyKind::Lru),
        "static" => Ok(PlacementPolicyKind::Static),
        _ => Err(StorageError::ParseError("invalid placement policy")),
    }
}

fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('|', "\\p")
        .replace('\n', "\\n")
}

fn unescape(value: &str) -> String {
    let mut out = String::new();
    let mut chars = value.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(next) = chars.next() {
                match next {
                    'n' => out.push('\n'),
                    'p' => out.push('|'),
                    '\\' => out.push('\\'),
                    other => {
                        out.push('\\');
                        out.push(other);
                    }
                }
            } else {
                out.push('\\');
            }
        } else {
            out.push(ch);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        fnv64_hex, CheckpointManifest, Fnv64HashVerifier, OptimStateRef, RouterSeedRef,
        RouterStateRef, ShardDescriptor, ShardType, TierManifestEntry,
    };
    use crate::placement::{PlacementPolicyKind, StorageTier};
    use crate::types::ExpertKey;

    fn sample_manifest() -> CheckpointManifest {
        CheckpointManifest {
            model_id: "lite-llm-base".to_owned(),
            epoch: 5,
            step: 10_000,
            tiers: vec![
                TierManifestEntry {
                    tier_id: 1,
                    name: "hot".to_owned(),
                    size_budget_bytes: 40 * 1024 * 1024,
                    placement_policy: PlacementPolicyKind::Prioritized,
                    priority_score: 10,
                },
                TierManifestEntry {
                    tier_id: 2,
                    name: "warm".to_owned(),
                    size_budget_bytes: 200 * 1024 * 1024,
                    placement_policy: PlacementPolicyKind::Lru,
                    priority_score: 20,
                },
            ],
            shards: vec![
                ShardDescriptor {
                    path: "dense/l0/attention_q_0.bin".to_owned(),
                    shard_type: ShardType::Dense,
                    tensor_name: Some("attention_q".to_owned()),
                    expert_key: None,
                    shape: vec![512, 1024],
                    dtype: "bf16".to_owned(),
                    tier_hint: StorageTier::Hot,
                    checksum_hex: fnv64_hex(b"dense-shard"),
                    hash_algorithm: "fnv64".to_owned(),
                    version: 1,
                    bytes: 11,
                },
                ShardDescriptor {
                    path: "experts/t2/g0/e1.bin".to_owned(),
                    shard_type: ShardType::Expert,
                    tensor_name: None,
                    expert_key: Some(ExpertKey::new(2, 0, 1)),
                    shape: vec![4096, 512],
                    dtype: "bf16".to_owned(),
                    tier_hint: StorageTier::Warm,
                    checksum_hex: fnv64_hex(b"expert-shard"),
                    hash_algorithm: "fnv64".to_owned(),
                    version: 3,
                    bytes: 12,
                },
            ],
            optim_state: Some(OptimStateRef {
                path: "optim/state.bin".to_owned(),
                checksum_hex: fnv64_hex(b"optim"),
                hash_algorithm: "fnv64".to_owned(),
                version: 2,
            }),
            router_state: Some(RouterStateRef {
                path: "router/state.bin".to_owned(),
                checksum_hex: fnv64_hex(b"router"),
                hash_algorithm: "fnv64".to_owned(),
                version: 7,
                base_seed: 42,
                layer_seeds: vec![RouterSeedRef { layer: 0, seed: 99 }],
            }),
            metadata_version: 1,
        }
    }

    #[test]
    fn manifest_roundtrip_is_lossless() {
        let manifest = sample_manifest();
        let serialized = manifest
            .to_canonical_string()
            .expect("serialization should succeed");
        let parsed =
            CheckpointManifest::from_canonical_string(&serialized).expect("parsing should succeed");

        assert_eq!(manifest, parsed);
    }

    #[test]
    fn shard_verification_detects_corruption() {
        let manifest = sample_manifest();
        let verifier = Fnv64HashVerifier;

        let mut shard_bytes = BTreeMap::new();
        shard_bytes.insert(
            "dense/l0/attention_q_0.bin".to_owned(),
            b"dense-shard".to_vec(),
        );
        shard_bytes.insert("experts/t2/g0/e1.bin".to_owned(), b"CORRUPTED".to_vec());

        assert!(manifest.verify_shards(&shard_bytes, &verifier).is_err());
    }
}
