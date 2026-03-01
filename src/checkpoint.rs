use crate::TierId;

#[derive(Debug, Clone)]
pub struct ExpertShardRef {
    pub tier: TierId,
    pub group: u32,
    pub expert: u32,
    pub uri: String,
    pub checksum_hex: String,
}

#[derive(Debug, Clone, Default)]
pub struct CheckpointManifest {
    pub schema_version: u32,
    pub run_id: String,
    pub shards: Vec<ExpertShardRef>,
}

impl CheckpointManifest {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_version == 0 {
            return Err("schema_version must be > 0");
        }
        if self.run_id.is_empty() {
            return Err("run_id must not be empty");
        }
        Ok(())
    }
}
