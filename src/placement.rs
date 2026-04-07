use std::collections::BTreeMap;

use crate::error::{StorageError, StorageResult};
use crate::types::{ExpertKey, TierId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StorageTier {
    Hot,
    Warm,
    Cold,
    Archive,
}

impl StorageTier {
    pub fn importance_penalty(self) -> u8 {
        match self {
            Self::Hot => 0,
            Self::Warm => 1,
            Self::Cold => 2,
            Self::Archive => 3,
        }
    }

    pub fn demote(self) -> Self {
        match self {
            Self::Hot => Self::Warm,
            Self::Warm => Self::Cold,
            Self::Cold => Self::Archive,
            Self::Archive => Self::Archive,
        }
    }

    pub fn promote(self) -> Self {
        match self {
            Self::Archive => Self::Cold,
            Self::Cold => Self::Warm,
            Self::Warm => Self::Hot,
            Self::Hot => Self::Hot,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementPolicyKind {
    Prioritized,
    Lru,
    Static,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierPlacementPlan {
    pub tier: TierId,
    pub target: StorageTier,
    pub size_bytes: u64,
    pub priority_score: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccessSignal {
    pub activation_count: u64,
    pub time_since_last_use: u64,
    pub size_bytes: u64,
    pub tier_importance: u8,
}

impl AccessSignal {
    pub fn priority(self) -> u64 {
        // Lower is better for residency; deterministic and integer-only.
        let frequency_term = 1_000_000_u64 / (self.activation_count.saturating_add(1));
        let recency_term = self.time_since_last_use.saturating_mul(64);
        let size_term = self.size_bytes / 4096;
        let tier_term = u64::from(self.tier_importance).saturating_mul(128);
        frequency_term
            .saturating_add(recency_term)
            .saturating_add(size_term)
            .saturating_add(tier_term)
    }
}

pub trait PlacementPolicy {
    fn placement_for_tier(&self, tier: TierId) -> StorageResult<StorageTier>;
    fn placement_for_expert(&self, key: ExpertKey) -> StorageResult<StorageTier>;
    fn update_expert_placement(&mut self, key: ExpertKey, signal: AccessSignal) -> StorageTier;
}

#[derive(Debug, Clone)]
pub struct AdaptivePlacementPolicy {
    tier_hints: BTreeMap<TierId, StorageTier>,
    expert_placements: BTreeMap<ExpertKey, StorageTier>,
    pub kind: PlacementPolicyKind,
    hot_threshold: u64,
    warm_threshold: u64,
    cold_threshold: u64,
}

impl AdaptivePlacementPolicy {
    pub fn new(tier_hints: BTreeMap<TierId, StorageTier>, kind: PlacementPolicyKind) -> Self {
        Self {
            tier_hints,
            expert_placements: BTreeMap::new(),
            kind,
            hot_threshold: 40_000,
            warm_threshold: 160_000,
            cold_threshold: 480_000,
        }
    }

    pub fn with_thresholds(
        mut self,
        hot_threshold: u64,
        warm_threshold: u64,
        cold_threshold: u64,
    ) -> Self {
        self.hot_threshold = hot_threshold;
        self.warm_threshold = warm_threshold;
        self.cold_threshold = cold_threshold;
        self
    }

    pub fn plan_for_tier(
        &self,
        tier: TierId,
        size_bytes: u64,
        priority_score: u32,
    ) -> StorageResult<TierPlacementPlan> {
        let target = self.placement_for_tier(tier)?;
        Ok(TierPlacementPlan {
            tier,
            target,
            size_bytes,
            priority_score,
        })
    }

    fn tier_for_priority(&self, priority: u64) -> StorageTier {
        if priority <= self.hot_threshold {
            StorageTier::Hot
        } else if priority <= self.warm_threshold {
            StorageTier::Warm
        } else if priority <= self.cold_threshold {
            StorageTier::Cold
        } else {
            StorageTier::Archive
        }
    }
}

impl PlacementPolicy for AdaptivePlacementPolicy {
    fn placement_for_tier(&self, tier: TierId) -> StorageResult<StorageTier> {
        self.tier_hints
            .get(&tier)
            .copied()
            .ok_or(StorageError::TierUnknown(tier))
    }

    fn placement_for_expert(&self, key: ExpertKey) -> StorageResult<StorageTier> {
        if let Some(tier) = self.expert_placements.get(&key) {
            return Ok(*tier);
        }
        self.placement_for_tier(key.tier)
    }

    fn update_expert_placement(&mut self, key: ExpertKey, signal: AccessSignal) -> StorageTier {
        let next = match self.kind {
            PlacementPolicyKind::Static => self
                .tier_hints
                .get(&key.tier)
                .copied()
                .unwrap_or(StorageTier::Cold),
            PlacementPolicyKind::Lru | PlacementPolicyKind::Prioritized => {
                self.tier_for_priority(signal.priority())
            }
        };

        self.expert_placements.insert(key, next);
        next
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        AccessSignal, AdaptivePlacementPolicy, PlacementPolicy, PlacementPolicyKind, StorageTier,
    };
    use crate::types::ExpertKey;

    #[test]
    fn adaptive_policy_promotes_hot_expert() {
        let mut tier_hints = BTreeMap::new();
        tier_hints.insert(1, StorageTier::Warm);

        let mut policy = AdaptivePlacementPolicy::new(tier_hints, PlacementPolicyKind::Prioritized);

        let tier = policy.update_expert_placement(
            ExpertKey::new(1, 0, 0),
            AccessSignal {
                activation_count: 10_000,
                time_since_last_use: 1,
                size_bytes: 1024,
                tier_importance: 0,
            },
        );

        assert_eq!(tier, StorageTier::Hot);
    }

    #[test]
    fn static_policy_keeps_hint_tier() {
        let mut tier_hints = BTreeMap::new();
        tier_hints.insert(10, StorageTier::Cold);

        let mut policy = AdaptivePlacementPolicy::new(tier_hints, PlacementPolicyKind::Static);

        let tier = policy.update_expert_placement(
            ExpertKey::new(10, 0, 1),
            AccessSignal {
                activation_count: 100,
                time_since_last_use: 100,
                size_bytes: 4096,
                tier_importance: 2,
            },
        );

        assert_eq!(tier, StorageTier::Cold);
    }
}
