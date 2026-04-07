# Changelog

All notable changes to `lite-llm-storage` are documented in this file.

## [0.1.0] - 2026-03-01
### Added
- Tier placement and cache policy primitives for hot/warm/cold/archive flows.
- Lazy expert loading and deterministic eviction behavior.
- Full checkpoint manifest schema with pluggable shard hash verification hooks.
- Snapshot commit/restore atomicity with restore validation checks.
- Consistency and corruption-detection tests for restore/checkpoint flows.
