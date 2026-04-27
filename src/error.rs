use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    InvalidConfig(String),
    InvalidManifest(&'static str),
    ParseError(&'static str),
    TierUnknown(u16),
    ChecksumMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    MissingShard(String),
    CapacityExceeded {
        required: u64,
        available: u64,
    },
    NoEvictionCandidate,
    Io(String),
    IoError(String),
    LockHeld,
    SnapshotNotFound(String),
    CorruptedSnapshot(&'static str),
}

pub type StorageResult<T> = Result<T, StorageError>;

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::InvalidManifest(msg) => write!(f, "invalid manifest: {msg}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::TierUnknown(tier) => write!(f, "unknown tier: {tier}"),
            Self::ChecksumMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "checksum mismatch for {path}: expected {expected}, got {actual}"
            ),
            Self::MissingShard(path) => write!(f, "missing shard at path: {path}"),
            Self::CapacityExceeded {
                required,
                available,
            } => write!(
                f,
                "capacity exceeded: required {required} bytes, available {available} bytes"
            ),
            Self::NoEvictionCandidate => write!(f, "no eligible eviction candidate"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
            Self::IoError(msg) => write!(f, "io error: {msg}"),
            Self::LockHeld => write!(f, "snapshot lock is currently held"),
            Self::SnapshotNotFound(id) => write!(f, "snapshot not found: {id}"),
            Self::CorruptedSnapshot(msg) => write!(f, "corrupted snapshot: {msg}"),
        }
    }
}

impl Error for StorageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl From<std::io::Error> for StorageError {
    fn from(err: std::io::Error) -> Self {
        StorageError::Io(err.to_string())
    }
}
