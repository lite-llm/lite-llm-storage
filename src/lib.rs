pub mod cache;
pub mod checkpoint;
pub mod cloud_backend;
pub mod error;
pub mod placement;
pub mod snapshot;
pub mod types;

pub use cache::{
    CacheEntryMeta, CacheStats, EvictionRecord, ExpertArtifact, ExpertStore, HotExpertCache,
    LazyLoadOutcome, LoadSource,
};
pub use checkpoint::{
    fnv64_hex, CheckpointManifest, Fnv64HashVerifier, OptimStateRef, RouterSeedRef, RouterStateRef,
    ShardDescriptor, ShardHashVerifier, ShardType, TierManifestEntry,
};
pub use cloud_backend::{
    build_backend, AsyncBackend, FilesystemBackend, StorageBackendConfig,
};
pub use error::{StorageError, StorageResult};
pub use placement::{
    AccessSignal, AdaptivePlacementPolicy, PlacementPolicy, PlacementPolicyKind, StorageTier,
    TierPlacementPlan,
};
pub use snapshot::{RestoredSnapshot, SnapshotRepository};
pub use types::{ExpertKey, TierId};
