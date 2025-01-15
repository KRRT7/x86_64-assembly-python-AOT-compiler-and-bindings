from __future__ import annotations
"""
All types and type aliases for the AOT.  It is okay to * import this in almost all cases.
"""
from dataclasses import dataclass
from typing import Callable, TypeVar, Generic
from x86_64_assembly_bindings import (
    Register, RegisterData, OffsetRegister,
    StackVariable, OffsetStackVariable, Instruction,
    InstructionData, Function, Program, Memory,
    MemorySize, Block, current_os
)


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

