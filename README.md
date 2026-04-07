# lite-llm-storage

Storage and checkpoint crate for Lite LLM (`SPEC-021` to `SPEC-030`).

## Scope
Implements deterministic storage semantics:

- tier placement policy (`hot`/`warm`/`cold`/`archive`)
- hot-cache management and deterministic eviction
- lazy expert loading contract
- checkpoint manifest schema and shard hash verification hooks
- snapshot commit/restore atomicity and restore validation

## Modules
- `src/placement.rs`: adaptive placement policy and tier planning
- `src/cache.rs`: hot cache, lazy loading, deterministic eviction strategy
- `src/checkpoint.rs`: checkpoint manifest model and canonical codec
- `src/snapshot.rs`: snapshot repository, staged commit and restore
- `src/types.rs`: shared key/type contracts
- `src/error.rs`: storage error model

## Build and Test
```bash
cargo fmt
cargo test
```

## Documentation
- System docs: `../lite-llm-docs/README.md`
- Recovery docs: `../lite-llm-docs/recovery/recovery-playbook.md`

## Changelog
See `CHANGELOG.md`.

## License
See `LICENSE`.
