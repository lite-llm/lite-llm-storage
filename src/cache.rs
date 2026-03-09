use std::collections::{BTreeMap, BTreeSet};

use crate::error::{StorageError, StorageResult};
use crate::placement::StorageTier;
use crate::types::ExpertKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub insertions: u64,
    pub evictions: u64,
    pub bytes_loaded: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheEntryMeta {
    pub source_tier: StorageTier,
    pub version: u32,
    pub pinned: bool,
    pub dirty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionRecord {
    pub key: ExpertKey,
    pub bytes: u64,
    pub demoted_to: StorageTier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertArtifact {
    pub bytes: Vec<u8>,
    pub source_tier: StorageTier,
    pub version: u32,
    pub checksum_hex: String,
    pub hash_algorithm: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadSource {
    HotCache,
    BackingStore,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LazyLoadOutcome {
    pub key: ExpertKey,
    pub source: LoadSource,
    pub bytes: u64,
}

pub trait ExpertStore {
    fn fetch_expert(&self, key: ExpertKey) -> StorageResult<ExpertArtifact>;
}

#[derive(Debug, Clone)]
struct CacheEntry {
    bytes: Vec<u8>,
    source_tier: StorageTier,
    _version: u32,
    pinned: bool,
    dirty: bool,
    ref_count: u32,
    access_count: u64,
    last_access_tick: u64,
}

impl CacheEntry {
    fn size_bytes(&self) -> u64 {
        self.bytes.len() as u64
    }
}

#[derive(Debug, Clone)]
pub struct HotExpertCache {
    capacity_bytes: u64,
    used_bytes: u64,
    seed: u64,
    tick: u64,
    entries: BTreeMap<ExpertKey, CacheEntry>,
    stats: CacheStats,
}

impl HotExpertCache {
    pub fn new(capacity_bytes: u64, seed: u64) -> StorageResult<Self> {
        if capacity_bytes == 0 {
            return Err(StorageError::InvalidConfig(
                "cache capacity must be greater than zero",
            ));
        }

        Ok(Self {
            capacity_bytes,
            used_bytes: 0,
            seed,
            tick: 0,
            entries: BTreeMap::new(),
            stats: CacheStats {
                hits: 0,
                misses: 0,
                insertions: 0,
                evictions: 0,
                bytes_loaded: 0,
            },
        })
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub fn contains(&self, key: ExpertKey) -> bool {
        self.entries.contains_key(&key)
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    pub fn get(&mut self, key: ExpertKey) -> Option<&[u8]> {
        self.tick = self.tick.saturating_add(1);

        if let Some(entry) = self.entries.get_mut(&key) {
            self.stats.hits = self.stats.hits.saturating_add(1);
            entry.access_count = entry.access_count.saturating_add(1);
            entry.last_access_tick = self.tick;
            Some(entry.bytes.as_slice())
        } else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            None
        }
    }

    pub fn begin_use(&mut self, key: ExpertKey) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.ref_count = entry.ref_count.saturating_add(1);
        }
    }

    pub fn end_use(&mut self, key: ExpertKey) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.ref_count = entry.ref_count.saturating_sub(1);
        }
    }

    pub fn pin(&mut self, key: ExpertKey, pinned: bool) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.pinned = pinned;
        }
    }

    pub fn mark_dirty(&mut self, key: ExpertKey, dirty: bool) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.dirty = dirty;
        }
    }

    pub fn insert(
        &mut self,
        key: ExpertKey,
        bytes: Vec<u8>,
        meta: CacheEntryMeta,
    ) -> StorageResult<Vec<EvictionRecord>> {
        let entry_size = bytes.len() as u64;
        if entry_size > self.capacity_bytes {
            return Err(StorageError::CapacityExceeded {
                required: entry_size,
                available: self.capacity_bytes,
            });
        }

        self.tick = self.tick.saturating_add(1);

        if let Some(existing) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(existing.size_bytes());
        }

        let required_total = self.used_bytes.saturating_add(entry_size);
        let mut evictions = Vec::new();

        if required_total > self.capacity_bytes {
            let needed = required_total - self.capacity_bytes;
            evictions = self.evict_bytes(needed)?;
        }

        self.entries.insert(
            key,
            CacheEntry {
                bytes,
                source_tier: meta.source_tier,
                _version: meta.version,
                pinned: meta.pinned,
                dirty: meta.dirty,
                ref_count: 0,
                access_count: 1,
                last_access_tick: self.tick,
            },
        );

        self.used_bytes = self.used_bytes.saturating_add(entry_size);
        self.stats.insertions = self.stats.insertions.saturating_add(1);
        self.stats.bytes_loaded = self.stats.bytes_loaded.saturating_add(entry_size);

        Ok(evictions)
    }

    fn evict_bytes(&mut self, mut bytes_needed: u64) -> StorageResult<Vec<EvictionRecord>> {
        let mut candidates: Vec<(ExpertKey, u64, u64)> = Vec::new();

        for (key, entry) in &self.entries {
            if entry.pinned || entry.ref_count > 0 {
                continue;
            }

            let recency = self.tick.saturating_sub(entry.last_access_tick);
            let frequency = entry.access_count;
            let size = entry.size_bytes();
            let tier_penalty = entry.source_tier.importance_penalty() as u64;
            let priority = (1_000_000_u64 / (frequency.saturating_add(1)))
                .saturating_add(recency.saturating_mul(64))
                .saturating_add(size / 4096)
                .saturating_add(tier_penalty.saturating_mul(128));

            let tie = seeded_key_hash(self.seed, *key);
            candidates.push((*key, priority, tie));
        }

        candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0)));

        let mut evictions = Vec::new();

        for (key, _, _) in candidates {
            if bytes_needed == 0 {
                break;
            }

            let entry = self.entries.remove(&key).ok_or(StorageError::NoEvictionCandidate)?;
            let size = entry.size_bytes();
            self.used_bytes = self.used_bytes.saturating_sub(size);
            bytes_needed = bytes_needed.saturating_sub(size);

            evictions.push(EvictionRecord {
                key,
                bytes: size,
                demoted_to: entry.source_tier.demote(),
            });

            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }

        if bytes_needed > 0 {
            return Err(StorageError::NoEvictionCandidate);
        }

        Ok(evictions)
    }

    pub fn load_experts_lazy<S, V>(
        &mut self,
        keys: &[ExpertKey],
        store: &S,
        verify: &V,
    ) -> StorageResult<Vec<LazyLoadOutcome>>
    where
        S: ExpertStore,
        V: Fn(ExpertKey, &ExpertArtifact) -> StorageResult<()>,
    {
        let mut unique = BTreeSet::new();
        for key in keys {
            unique.insert(*key);
        }

        let mut outcomes = Vec::new();
        for key in unique {
            if self.contains(key) {
                let _ = self.get(key);
                outcomes.push(LazyLoadOutcome {
                    key,
                    source: LoadSource::HotCache,
                    bytes: self
                        .entries
                        .get(&key)
                        .map(|entry| entry.size_bytes())
                        .unwrap_or(0),
                });
                continue;
            }

            let artifact = store.fetch_expert(key)?;
            verify(key, &artifact)?;

            self.insert(
                key,
                artifact.bytes.clone(),
                CacheEntryMeta {
                    source_tier: artifact.source_tier,
                    version: artifact.version,
                    pinned: false,
                    dirty: false,
                },
            )?;

            outcomes.push(LazyLoadOutcome {
                key,
                source: LoadSource::BackingStore,
                bytes: artifact.bytes.len() as u64,
            });
        }

        Ok(outcomes)
    }
}

fn seeded_key_hash(seed: u64, key: ExpertKey) -> u64 {
    let mut payload = [0u8; 18];
    payload[0..8].copy_from_slice(&seed.to_le_bytes());
    payload[8..10].copy_from_slice(&key.tier.to_le_bytes());
    payload[10..14].copy_from_slice(&key.group.to_le_bytes());
    payload[14..18].copy_from_slice(&key.expert.to_le_bytes());

    let mut hash = 0xcbf29ce484222325_u64;
    for byte in &payload {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{CacheEntryMeta, ExpertArtifact, ExpertStore, HotExpertCache, LoadSource};
    use crate::error::StorageResult;
    use crate::placement::StorageTier;
    use crate::types::ExpertKey;

    #[derive(Debug, Clone)]
    struct InMemoryStore {
        entries: BTreeMap<ExpertKey, ExpertArtifact>,
    }

    impl InMemoryStore {
        fn new(entries: BTreeMap<ExpertKey, ExpertArtifact>) -> Self {
            Self { entries }
        }
    }

    impl ExpertStore for InMemoryStore {
        fn fetch_expert(&self, key: ExpertKey) -> StorageResult<ExpertArtifact> {
            self.entries
                .get(&key)
                .cloned()
                .ok_or(crate::error::StorageError::MissingShard(key.encode()))
        }
    }

    #[test]
    fn deterministic_eviction_keeps_recent_hot_entries() {
        let mut cache = HotExpertCache::new(16, 99).expect("cache should initialize");

        let k1 = ExpertKey::new(1, 0, 1);
        let k2 = ExpertKey::new(1, 0, 2);
        let k3 = ExpertKey::new(1, 0, 3);

        cache
            .insert(
                k1,
                vec![1; 8],
                CacheEntryMeta {
                    source_tier: StorageTier::Hot,
                    version: 1,
                    pinned: false,
                    dirty: false,
                },
            )
            .expect("insert should succeed");
        cache
            .insert(
                k2,
                vec![2; 8],
                CacheEntryMeta {
                    source_tier: StorageTier::Warm,
                    version: 1,
                    pinned: false,
                    dirty: false,
                },
            )
            .expect("insert should succeed");

        let _ = cache.get(k1);

        let evictions = cache
            .insert(
                k3,
                vec![3; 8],
                CacheEntryMeta {
                    source_tier: StorageTier::Cold,
                    version: 1,
                    pinned: false,
                    dirty: false,
                },
            )
            .expect("evicting insert should succeed");

        assert!(!evictions.is_empty());
        assert!(cache.contains(k1));
    }

    #[test]
    fn lazy_loader_fetches_in_deterministic_key_order() {
        let mut entries = BTreeMap::new();
        let a = ExpertKey::new(1, 0, 1);
        let b = ExpertKey::new(1, 0, 2);

        entries.insert(
            a,
            ExpertArtifact {
                bytes: vec![1; 4],
                source_tier: StorageTier::Warm,
                version: 1,
                checksum_hex: "ok".to_owned(),
                hash_algorithm: "none".to_owned(),
            },
        );
        entries.insert(
            b,
            ExpertArtifact {
                bytes: vec![2; 4],
                source_tier: StorageTier::Warm,
                version: 1,
                checksum_hex: "ok".to_owned(),
                hash_algorithm: "none".to_owned(),
            },
        );

        let store = InMemoryStore::new(entries);
        let mut cache = HotExpertCache::new(32, 11).expect("cache should initialize");

        let outcomes = cache
            .load_experts_lazy(&[b, a, b], &store, &|_, _| Ok(()))
            .expect("lazy load should succeed");

        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0].key, a);
        assert_eq!(outcomes[1].key, b);
        assert_eq!(outcomes[0].source, LoadSource::BackingStore);
    }
}


