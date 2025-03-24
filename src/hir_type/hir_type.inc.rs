pub const TEmpty: Type = Type::from_bits(0x0);
pub const TSmallInt: Type = Type::from_bits(0x1);
pub const TLargeInt: Type = Type::from_bits(0x2);
pub const TInt: Type = Type::from_bits(0x3);
pub const TFloat: Type = Type::from_bits(0x4);
pub const TBool: Type = Type::from_bits(0x8);
pub const TSmallStr: Type = Type::from_bits(0x10);
pub const TImmediate: Type = Type::from_bits(0x19);
pub const TLargeStr: Type = Type::from_bits(0x20);
pub const TStr: Type = Type::from_bits(0x30);
pub const TClass: Type = Type::from_bits(0x40);
pub const TFunction: Type = Type::from_bits(0x80);
pub const TNil: Type = Type::from_bits(0x100);
pub const TFrame: Type = Type::from_bits(0x200);
pub const TObject: Type = Type::from_bits(0x3ff);
pub const TCInt8: Type = Type::from_bits(0x400);
pub const TCInt16: Type = Type::from_bits(0x800);
pub const TCInt32: Type = Type::from_bits(0x1000);
pub const TCInt64: Type = Type::from_bits(0x2000);
pub const TCSigned: Type = Type::from_bits(0x3c00);
pub const TCUInt8: Type = Type::from_bits(0x4000);
pub const TCUInt16: Type = Type::from_bits(0x8000);
pub const TCUInt32: Type = Type::from_bits(0x10000);
pub const TCUInt64: Type = Type::from_bits(0x20000);
pub const TCUnsigned: Type = Type::from_bits(0x3c000);
pub const TCInt: Type = Type::from_bits(0x3fc00);
pub const TCBool: Type = Type::from_bits(0x40000);
pub const TPrimitive: Type = Type::from_bits(0x7fc00);
pub const TVoid: Type = Type::from_bits(0x80000);
pub const TAny: Type = Type::from_bits(0xfffff);
pub const NUM_TYPE_BITS: usize = 20;
pub const ALL_TYPES: [(&'static str, Type); 30] = [
    ("Any", TAny),
    ("Void", TVoid),
    ("Primitive", TPrimitive),
    ("CBool", TCBool),
    ("CInt", TCInt),
    ("CUnsigned", TCUnsigned),
    ("CUInt64", TCUInt64),
    ("CUInt32", TCUInt32),
    ("CUInt16", TCUInt16),
    ("CUInt8", TCUInt8),
    ("CSigned", TCSigned),
    ("CInt64", TCInt64),
    ("CInt32", TCInt32),
    ("CInt16", TCInt16),
    ("CInt8", TCInt8),
    ("Object", TObject),
    ("Frame", TFrame),
    ("Nil", TNil),
    ("Function", TFunction),
    ("Class", TClass),
    ("Str", TStr),
    ("LargeStr", TLargeStr),
    ("Immediate", TImmediate),
    ("SmallStr", TSmallStr),
    ("Bool", TBool),
    ("Float", TFloat),
    ("Int", TInt),
    ("LargeInt", TLargeInt),
    ("SmallInt", TSmallInt),
    ("Empty", TEmpty),
];
