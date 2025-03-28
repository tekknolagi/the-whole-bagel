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

#[cfg(test)]
mod type_tests {
    use expect_test::expect;
    use crate::hir_type::*;

    #[test]
    fn test_type_order() { assert!(ALL_TYPES.windows(2).all(|w| w[0].1.bits >= w[1].1.bits),
                "ALL_TYPES should be sorted in decreasing order by bit value");
    }

    #[test]
    fn test_int() {
        assert!(TSmallInt.is_subtype(TInt));
        assert!(TLargeInt.is_subtype(TInt));
        assert!(TLargeInt.is_subtype(TAny));
        assert!(TInt.is_subtype(TAny));
        assert!(TEmpty.is_subtype(TAny));
        assert!(TEmpty.is_subtype(TInt));

        assert!(!TInt.is_subtype(TSmallInt));
        assert!(!TInt.is_subtype(TLargeInt));
        assert!(!TAny.is_subtype(TInt));
        assert!(!TAny.is_subtype(TEmpty));
        assert!(!TInt.is_subtype(TEmpty));
    }

    #[test]
    fn test_union() {
        assert!(TSmallInt.union(TLargeInt).bit_equal(TInt));
    }

    #[test]
    fn test_display_base() {
        expect!["SmallInt"].assert_eq(&TSmallInt.to_string());
        expect!["LargeInt"].assert_eq(&TLargeInt.to_string());
        expect!["Int"].assert_eq(&TInt.to_string());
        expect!["Any"].assert_eq(&TAny.to_string());
        expect!["Empty"].assert_eq(&TEmpty.to_string());
    }

    #[test]
    fn test_display_union() {
        expect!["Int"].assert_eq(&TSmallInt.union(TLargeInt).to_string());
        expect!["Str"].assert_eq(&TSmallStr.union(TLargeStr).to_string());
        expect!["Str|SmallInt"].assert_eq(&TSmallInt.union(TStr).to_string());
        expect!["CBool|Str|SmallInt"].assert_eq(&TSmallInt.union(TStr).union(TCBool).to_string());
    }
}
