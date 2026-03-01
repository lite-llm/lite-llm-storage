pub type TierId = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    pub tier: TierId,
    pub group: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub const fn new(tier: TierId, group: u32, expert: u32) -> Self {
        Self { tier, group, expert }
    }

    pub fn as_tuple(self) -> (TierId, u32, u32) {
        (self.tier, self.group, self.expert)
    }

    pub fn parse(value: &str) -> Option<Self> {
        let parts: Vec<&str> = value.split(':').collect();
        if parts.len() != 3 {
            return None;
        }
        let tier = parts[0].parse::<u16>().ok()?;
        let group = parts[1].parse::<u32>().ok()?;
        let expert = parts[2].parse::<u32>().ok()?;
        Some(Self::new(tier, group, expert))
    }

    pub fn encode(self) -> String {
        format!("{}:{}:{}", self.tier, self.group, self.expert)
    }
}
