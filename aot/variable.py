from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar, Generic, TYPE_CHECKING
from x86_64_assembly_bindings import MemorySize
from aot.type_imports import *

TVal = TypeVar("TVal")
@dataclass
class Value(Generic[TVal]):

    python_type:TVal|Array
    _value:VariableValueType
    size:MemorySize = MemorySize.QWORD

    @property
    def value(self) -> VariableValueType:
        return self._value
    
    def __getitem__(self, value:Any):
        if isinstance(self.value, OffsetRegister):
            return self.value[value]
        elif isinstance(self.value, Register):
            return OffsetRegister(self.value, 0)[value]
        
    def __hash__(self):
        return hash(f"{self.python_type.__name__}{self.size}{hash(self.value)}")

T = TypeVar("T")
class Variable(Value, Generic[T]):

    @classmethod
    def from_value(cls, name:str, value:Value) -> Variable:
        # >> TODO: Perform mov into variable << #
        return cls(name, value.python_type, value._value, value.size)

    def __init__(self, name:str, python_type:T|Array,
        _value:VariableValueType, size:MemorySize = MemorySize.QWORD
    ):
        self.name = name
        super().__init__(python_type, _value, size)

    def set(self, other:Variable[T] | T, python_function: Any) -> LinesType:
        from aot.utils import load, CAST, type_from_object
        from aot.function import PythonFunction

        python_function: PythonFunction = python_function

        lines, other_value = load(other, python_function, no_mov=True)
        

        if self.python_type is int:
            if type_from_object(other) is not int:
                instrs, other_value = CAST.int(other_value, python_function)
                lines.extend(instrs)
            lines.append(Ins("mov", self.value, other_value))
            return lines
        
        if self.python_type is bool:
            if type_from_object(other) is not bool:
                instrs, other_value = CAST.bool(other_value, python_function)
                lines.extend(instrs)
            lines.append(Ins("mov", self.value, other_value))
            return lines
        
        elif self.python_type is float:
            if type_from_object(other) is not float:
                instrs, other_value = CAST.float(other_value, python_function)
                lines.extend(instrs)
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
