from __future__ import annotations

from aot.utility_types import NOOP_All_Dunders
"""
All types and type aliases for the AOT.  It is okay to * import this in almost all cases.
"""
from dataclasses import dataclass
from typing import Callable, Literal, TypeVar, Generic
from x86_64_assembly_bindings import (
    Register, RegisterData, OffsetRegister,
    StackVariable, OffsetStackVariable, Instruction,
    InstructionData, Function, Program, Memory,
    MemorySize, Block, current_os
)

@dataclass
class Template(NOOP_All_Dunders):
    name:str
    default:type|None = None

    def __hash__(self):
        return hash(f"{self.name}{self.default}")
    
    @classmethod
    def __class_getitem__(cls, template_name:str, default = None):
        return cls(template_name, default)

ArrayTypeTemplate = TypeVar("ArrayTypeTemplate")
ArrayLengthTemplate = TypeVar("ArrayLengthTemplate")
class Array(Generic[ArrayTypeTemplate, ArrayLengthTemplate]):
    
    def __init__(self, python_type:type, length:int):
        from aot.utils import memory_size_from_type
        self.python_type: type = python_type
        self.length = length
        self._type_size: MemorySize = memory_size_from_type(python_type)
        self.item_size: MemorySize = self._type_size

    @property
    def type_size(self):
        if isinstance(self._type_size, MemorySize):
            return self._type_size.value
        else:
            return self._type_size
            

    def __hash__(self):
        return hash(f"{self.python_type}{self.length}{self.type_size}")
    
    def get_offset(self, index:int):
        return self.type_size * index // 8
    
    @property
    def size(self):
        return self.type_size * self.length
    
    def __len__(self) -> int:
        return self.length
    
    def __eq__(self, value:Array):
        try:
            return self.python_type == value.python_type and \
                    self.length == value.length
        except:
            # Any non array should return false
            return False
    
    @property
    def type_tuple(self) -> tuple[type[Array], tuple[Literal["tuple"], type[ArrayTypeTemplate], int]]:
        return self.get_type_tuple(self.python_type, self.length)
    
    @classmethod
    def from_type_tuple(cls, type_tuple:tuple[type[Array], tuple[Literal["tuple"], type[ArrayTypeTemplate], int]]) -> Array:
        _, array_type, array_length = type_tuple[1]
        if isinstance(array_type, tuple):
            array_type = cls.from_type_tuple(array_type)
        return cls(array_type, array_length)
    
    @classmethod
    def get_type_tuple(cls, array_type:type, array_length:int):
        return (cls, ("tuple", array_type, array_length))
    
    @classmethod
    def __class_getitem__(cls, args:tuple[type,int]):
        return cls(*args)
    
    
        

class FloatLiteral(float):
    python_type: type = float
    size: MemorySize = MemorySize.QWORD

    def __add__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__add__(other))
    
    def __sub__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__sub__(other))

    def __mul__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__mul__(other))
    
    def __floordiv__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__floordiv__(other))
    
    def __mod__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__floordiv__(other))
    
    def __truediv__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__floordiv__(other))
    
    @property
    def value(self):
        from aot.utils import float_to_hex
        return float_to_hex(self)
    
class IntLiteral(int):
    python_type: type = int
    size: MemorySize = MemorySize.QWORD
    
    def __add__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__add__(other))\
            if isinstance(other, FloatLiteral) else\
            IntLiteral(super().__add__(other))
    
    def __sub__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__sub__(other))\
            if isinstance(other, FloatLiteral) else\
            IntLiteral(super().__sub__(other))

    def __mul__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__mul__(other))\
            if isinstance(other, FloatLiteral) else\
            IntLiteral(super().__mul__(other))
    
    def __floordiv__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__floordiv__(other))\
            if isinstance(other, FloatLiteral) else\
            IntLiteral(super().__floordiv__(other))
    
    def __mod__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__floordiv__(other))\
            if isinstance(other, FloatLiteral) else\
            IntLiteral(super().__floordiv__(other))
    
    def __truediv__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__floordiv__(other))

    @property
    def value(self):
        return self
    
class BoolLiteral(int):
    python_type: type = int
    size: MemorySize = MemorySize.QWORD
    
    def __add__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__add__(other))\
            if isinstance(other, FloatLiteral) else\
            IntLiteral(super().__add__(other))
    
    def __sub__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__sub__(other))\
            if isinstance(other, FloatLiteral) else\
            IntLiteral(super().__sub__(other))

    def __mul__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__mul__(other))\
            if isinstance(other, FloatLiteral) else\
            IntLiteral(super().__mul__(other))
    
    def __floordiv__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__floordiv__(other))\
            if isinstance(other, FloatLiteral) else\
            IntLiteral(super().__floordiv__(other))
    
    def __mod__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__floordiv__(other))\
            if isinstance(other, FloatLiteral) else\
            IntLiteral(super().__floordiv__(other))
    
    def __truediv__(self, other) -> IntLiteral|FloatLiteral:
        return FloatLiteral(super().__floordiv__(other))
    
    def __and__(self, value) -> BoolLiteral:
        return BoolLiteral(bool(self) and value)
    
    def __or__(self, value) -> BoolLiteral:
        return BoolLiteral(bool(self) or value)
    
    def __not__(self) -> BoolLiteral:
        return BoolLiteral(not bool(self))
    
    def __bool__(self):
        return True if self else False
    
    def __str__(self):
        return str(bool(self))
    
    def __int__(self):
        return int(self)
    
    @property
    def value(self):
        return self


# Type aliases

VariableValueType = Register|OffsetRegister|StackVariable|OffsetStackVariable

Comment = str

LinesType = list[Instruction | Block | Callable | Comment]

ScalarType = bool|IntLiteral|FloatLiteral

Reg = Register

RegD = RegisterData

Ins = Instruction

InsD = InstructionData