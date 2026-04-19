# lite-llm-storage

Storage and checkpoint crate for Lite LLM (`SPEC-021` to `SPEC-030`).

## Overview
Implements deterministic storage semantics for tiered data placement, caching, checkpoint management, and cloud storage backends.

This crate provides the complete storage stack: adaptive tier placement policy (`hot`/`warm`/`cold`/`archive`), hot-cache management with deterministic eviction, lazy expert loading contract, checkpoint manifest schema with shard hash verification, snapshot commit/restore atomicity, and production async storage backends supporting local filesystem and S3-compatible cloud storage (AWS S3, GCS, MinIO).

## Features

### Feature Flag: `default` (empty)
No optional features enabled by default. The async filesystem backend is always available.

### Feature Flag: `cloud-backends` (optional)
Enables `aws-sdk-s3` and `aws-config` dependencies for S3/GCS/MinIO support. Activates `S3Backend` implementation of the `AsyncBackend` trait.

## Dependencies
| Crate | Version | Purpose |
|-------|---------|---------|
| tokio | 1 | Async runtime for filesystem and network I/O |
| async-trait | 0.1 | Async trait support for `AsyncBackend` |
| aws-sdk-s3 | 1 (optional) | S3-compatible cloud storage SDK |
| aws-config | 1 (optional) | AWS credential resolution |
| serde | 1.0 | Serialization for checkpoint metadata |
| serde_json | 1.0 | JSON encoding for manifests |
| url | 2 | URL parsing for S3 URIs |

## Key Modules
- `placement` — adaptive tier placement policy, tier planning, storage tier definitions
- `cache` — hot cache management, lazy expert loading, deterministic eviction
- `checkpoint` — checkpoint manifest model, shard hash verification, codec
- `snapshot` — snapshot repository, staged commit, restore validation
- `cloud_backend` — async storage backends: `FilesystemBackend`, `S3Backend`, `build_backend()`
- `types` — shared key/type contracts (`ExpertKey`, `TierId`)
- `error` — storage error model (I/O, invalid config, missing shard)

## Public API
### Core Types
- `AsyncBackend` — trait for async storage backends (read, write, exists, delete, list)
- `FilesystemBackend` — async local/NFS file I/O with atomic writes
- `S3Backend` — AWS S3/GCS/MinIO via AWS SDK (requires `cloud-backends` feature)
- `StorageBackendConfig` — configuration with factory methods: `filesystem()`, `s3()`, `minio()`
- `HotExpertCache` — in-memory hot cache with deterministic eviction
- `AdaptivePlacementPolicy` — tier placement decision engine
- `CheckpointManifest` — checkpoint manifest with shard descriptors and hash verifiers
- `SnapshotRepository` — atomic snapshot commit and restore
- `StorageTier` — tier enumeration (`Hot`, `Warm`, `Cold`, `Archive`)

### Core Functions
- `build_backend()` — factory function creating `AsyncBackend` from config
- `fnv64_hex()` — FNV-1a hash encoding for checksums

### Traits
- `AsyncBackend` — async storage interface for shards, checkpoints, and snapshots

## Quick Start
```rust
use lite_llm_storage::{
    FilesystemBackend, StorageBackendConfig, AsyncBackend,
    build_backend,
};

// Create a local filesystem backend
let backend = FilesystemBackend::new("/data/checkpoints")
    .await
    .expect("backend should create");

// Write and read data
backend.write("model/shard.bin", b"weights-data").await?;
assert!(backend.exists("model/shard.bin").await?);
let content = backend.read("model/shard.bin").await?;
assert_eq!(content, b"weights-data");

// Or use the config factory
let config = StorageBackendConfig::filesystem("/data/checkpoints");
let backend = build_backend(&config).await?;
```

## Running Tests
```bash
cargo fmt
cargo test
```

## Architecture
This crate implements the storage layer for the lite-llm platform, providing tiered data placement, checkpoint persistence, and snapshot management. The async backends (`cloud_backend` module) enable production deployments with local filesystem or cloud object storage (S3/GCS/MinIO). It integrates with `lite-llm-training` for distributed checkpointing and with `lite-llm-security` for encrypted artifact storage.

## Changelog
See `CHANGELOG.md`.

## License
See `LICENSE`.
