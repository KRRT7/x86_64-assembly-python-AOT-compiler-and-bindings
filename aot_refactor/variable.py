from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Generic, TYPE_CHECKING
from x86_64_assembly_bindings import MemorySize
from aot_refactor.type_imports import *



T = TypeVar("T")
@dataclass
class Variable(Generic[T]):

    name:str
    python_type:T
    _value:VariableValueType
    size:MemorySize = MemorySize.DWORD

    @property
    def value(self) -> tuple[LinesType, VariableValueType]:
        return self._value

    def set(self, other:Variable[T] | T) -> LinesType:
        from aot_refactor.utils import load
        lines, other_value = load(other)

        if isinstance(other, (int, bool)):
            lines.append(Ins("mov", self.value, other_value))
            return lines
        
        if isinstance(other, float):
            lines.append(Ins("mov", self.value, other_value))
            return lines
        
        elif self.size > other.size:
            if self.size > MemorySize.WORD and other.size < MemorySize.DWORD:
                lines.append(Ins("movsx", self.value, other_value))
                return lines
            
            elif other.size == MemorySize.DWORD and self.size == MemorySize.QWORD:
                lines.append(Ins("movsxd", self.value, other_value))
                return lines
            

        elif self.size < other.size:
            lines.append(Ins("mov", self.value, other_value))
            return lines
        
        else:
            lines.append(Ins("mov", self.value, other_value))
            return lines
        
        
    def __hash__(self):
        return hash(f"{self.name}{self.python_type.__name__}{self.size}{hash(self.value)}")
