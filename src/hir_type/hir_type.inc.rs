pub const TEmpty: Type = Type::from_bits(0x0);
pub const TSmallInt: Type = Type::from_bits(0x1);
pub const TLargeInt: Type = Type::from_bits(0x2);
pub const TInt: Type = Type::from_bits(0x3);
pub const TFloat: Type = Type::from_bits(0x4);
pub const TBool: Type = Type::from_bits(0x8);
pub const TSmallStr: Type = Type::from_bits(0x10);
pub const TLargeStr: Type = Type::from_bits(0x20);
pub const TStr: Type = Type::from_bits(0x30);
pub const TClass: Type = Type::from_bits(0x40);
pub const TFunction: Type = Type::from_bits(0x80);
pub const TNil: Type = Type::from_bits(0x100);
pub const TFrame: Type = Type::from_bits(0x200);
pub const TObject: Type = Type::from_bits(0x3ff);
pub const TCBool: Type = Type::from_bits(0x400);
pub const TPrimitive: Type = Type::from_bits(0x400);
pub const TVoid: Type = Type::from_bits(0x800);
pub const TAny: Type = Type::from_bits(0xfff);
pub const NUM_TYPE_BITS: usize = 12;
pub const ALL_TYPES: [(&'static str, Type); 18] = [
    ("Any", TAny),
    ("Void", TVoid),
    ("CBool", TCBool),
    ("Primitive", TPrimitive),
    ("Object", TObject),
    ("Frame", TFrame),
    ("Nil", TNil),
    ("Function", TFunction),
    ("Class", TClass),
    ("Str", TStr),
    ("LargeStr", TLargeStr),
    ("SmallStr", TSmallStr),
    ("Bool", TBool),
    ("Float", TFloat),
    ("Int", TInt),
    ("LargeInt", TLargeInt),
    ("SmallInt", TSmallInt),
    ("Empty", TEmpty),
];
