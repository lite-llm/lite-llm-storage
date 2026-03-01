use crate::TierId;

#[derive(Debug, Clone)]
pub struct TierPlacementPlan {
    pub tier: TierId,
    pub target: StorageTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTier {
    Hot,
    Warm,
    Cold,
    Archive,
}

pub trait PlacementPolicy {
    fn placement_for(&self, tier: TierId) -> StorageTier;
}
