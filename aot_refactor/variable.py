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
    size:MemorySize = MemorySize.QWORD

    @property
    def value(self) -> VariableValueType:
        return self._value

    def set(self, other:Variable[T] | T) -> LinesType:
        from aot_refactor.utils import load, CAST, type_from_object
        lines, other_value = load(other)

        if isinstance(other, (IntLiteral, bool)):
            if self.python_type not in {int, bool}:
                raise TypeError(
                    f"Attempted to set {self.python_type.__class__.__name__} type variable to {type(other).__name__}.  "
                    "All functions for compilation must be statically typed."
                )
            lines.append(Ins("mov", self.value, other_value))
            return lines
        
        if isinstance(other, FloatLiteral):
            if self.python_type is not float:
                raise TypeError(
                    f"Attempted to set {self.python_type.__class__.__name__} type variable to float.  "
                    "All functions for compilation must be statically typed."
                )
            lines.append(Ins("movsd", self.value, other_value))
            return lines
        
        if isinstance(self._value, Register) and self.python_type is float:
            if self.python_type is not float:
                raise TypeError(
                    f"Attempted to set {self.python_type.__class__.__name__} type variable to float.  "
                    "All functions for compilation must be statically typed."
                )
            lines.append(Ins("movsd", self.value, other_value))
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
