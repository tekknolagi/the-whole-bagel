from __future__ import annotations
import dataclasses
from functools import reduce

@dataclasses.dataclass
class Type:
    name: str
    children: list[Type] = dataclasses.field(default_factory=list)
    bits: int = 0

    def add_child(self, name: str) -> Type:
        result = Type(name)
        self.children.append(result)
        return result

    def topo(self) -> list[Type]:
        result = []
        visited = set()
        def _topo(ty):
            if ty in visited:
                return
            for child in ty.children:
                _topo(child)
            result.append(ty)
        _topo(self)
        return result

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other) -> bool:
        return self.name == other.name

Any = Type("Any")
Object = Any.add_child("Object")
Instance = Object.add_child("Instance")
Int = Object.add_child("Int")
SmallInt = Int.add_child("SmallInt")
LargeInt = Int.add_child("LargeInt")
Float = Object.add_child("Float")
Bool = Object.add_child("Bool")
Str = Object.add_child("Str")
SmallStr = Str.add_child("SmallStr")
LargeStr = Str.add_child("LargeStr")
Class = Object.add_child("Class")
Function = Object.add_child("Function")
Nil = Object.add_child("Nil")
Frame = Object.add_child("Frame")
Closure = Object.add_child("Closure")

Primitive = Any.add_child("Primitive")
CInt = Primitive.add_child("CInt")
CBool = Primitive.add_child("CBool")
CSigned = CInt.add_child("CSigned")
CUnsigned = CInt.add_child("CUnsigned")
for size in [8, 16, 32, 64]:
    CUnsigned.add_child(f"CUInt{size}")
    CSigned.add_child(f"CInt{size}")

Void = Any.add_child("Void")

all_unions = []
def add_union(name, types):
    all_unions.append(Type(name, types))

add_union("Immediate", [SmallInt, SmallStr, Bool, Nil])

types_and_unions = Any.topo()+all_unions
num_bits = 0
for ty in types_and_unions:
    if not ty.children:
        ty.bits = 1 << num_bits
        num_bits += 1
    else:
        ty.bits = reduce(lambda acc, ty: acc | ty.bits, ty.children, 0)
types_and_unions.append(Type("Empty", [], 0))

for ty in sorted(types_and_unions, key=lambda ty: ty.bits):
    print(f"pub const T{ty.name}: Type = Type::from_bits(0x{ty.bits:x});")
print(f"pub const NUM_TYPE_BITS: usize = {num_bits};")


print(f"pub const ALL_TYPES: [(&'static str, Type); {len(types_and_unions)}] = [")
for ty in sorted(types_and_unions, key=lambda ty: -ty.bits):
    print(f"    (\"{ty.name}\", T{ty.name}),")
print("];")
