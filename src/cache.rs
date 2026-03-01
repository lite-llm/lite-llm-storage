use std::collections::HashMap;

use crate::TierId;

#[derive(Debug, Default, Clone, Copy)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

#[derive(Debug, Default)]
pub struct HotExpertCache {
    experts: HashMap<(TierId, u32, u32), Vec<u8>>,
    stats: CacheStats,
}

impl HotExpertCache {
    pub fn get(&mut self, key: (TierId, u32, u32)) -> Option<&[u8]> {
        if self.experts.contains_key(&key) {
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
        }
        self.experts.get(&key).map(Vec::as_slice)
    }

    pub fn put(&mut self, key: (TierId, u32, u32), weights: Vec<u8>) {
        self.experts.insert(key, weights);
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }
}
