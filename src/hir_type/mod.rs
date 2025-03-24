#[derive(Debug, Copy, Clone)]
pub struct Type {
    pub bits: u64,
}

impl Type {
    const fn from_bits(bits: u64) -> Self {
        Self { bits }
    }

    pub const fn union(self, other: Self) -> Self {
        Self { bits: self.bits | other.bits }
    }

    pub const fn intersection(self, other: Self) -> Self {
        Self { bits: self.bits & other.bits }
    }

    pub const fn bit_equal(self, other: Self) -> bool {
        self.bits == other.bits
    }

    pub const fn is_subtype(self, other: Self) -> bool {
        (self.bits & other.bits) == self.bits
    }

    pub fn to_string(self) -> String {
        format!("{self}")
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let mut bits = self.bits;
        let mut sep = "";
        for (name, pattern) in ALL_TYPES {
            if (bits & pattern.bits) == pattern.bits {
                write!(f, "{sep}{name}")?;
                bits &= !pattern.bits;
                if bits == 0 {
                    return Ok(())
                }
                sep = "|";
            }
        }
        unreachable!("Should have seen a matching bit pattern")
    }
}

include!("hir_type.inc.rs");
