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

Primitive = Any.add_child("Primitive")
CBool = Primitive.add_child("CBool")

Void = Any.add_child("Void")

all_unions = []
def add_union(name, types):
    all_unions.append(Type(name, types))

add_union("Immediate", [SmallInt, SmallStr, Bool])

all_types = Any.topo()
num_bits = 0
for ty in all_types+all_unions:
    if not ty.children:
        ty.bits = 1 << num_bits
        num_bits += 1
    else:
        ty.bits = reduce(lambda acc, ty: acc | ty.bits, ty.children, 0)
all_types.append(Type("Empty", [], 0))

for ty in sorted(all_types, key=lambda ty: ty.bits):
    print(f"pub const T{ty.name}: Type = Type::from_bits(0x{ty.bits:x});")
print(f"pub const NUM_TYPE_BITS: usize = {num_bits};")


print(f"pub const ALL_TYPES: [(&'static str, Type); {len(all_types)}] = [")
for ty in sorted(all_types, key=lambda ty: -ty.bits):
    print(f"    (\"{ty.name}\", T{ty.name}),")
print("];")
