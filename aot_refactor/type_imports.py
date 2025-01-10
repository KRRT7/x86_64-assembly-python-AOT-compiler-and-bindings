from __future__ import annotations
"""
All types and type aliases for the AOT.  It is okay to * import this in almost all cases.
"""
from typing import Callable
from x86_64_assembly_bindings import (
    Register, RegisterData, OffsetRegister,
    StackVariable, OffsetStackVariable, Instruction,
    InstructionData, Function, Program, Memory,
    MemorySize, Block, current_os
)

class FloatLiteral(str):
    python_type: type = float
    size: MemorySize = MemorySize.QWORD

    def __new__(cls, *args, **kwargs):
        ret = super().__new__(cls, *args, **kwargs)
        for dunder in [
            "add","mul","floordiv","mod","truediv","sub"
        ]:
            setattr(
                ret, f"__{dunder}__",
                lambda self, other:cls(getattr(super(),f"__{dunder}__")(self, other))
            )
        return ret
    
    @property
    def value(self):
        return self
    
class IntLiteral(int):
    python_type: type = int
    size: MemorySize = MemorySize.QWORD

    def __new__(cls, *args, **kwargs):
        ret = super().__new__(cls, *args, **kwargs)
        for dunder in [
            "add","mul","floordiv","mod","truediv","sub" 
        ]:
            setattr(
                ret, f"__{dunder}__",
                lambda self, other:cls(getattr(super(),f"__{dunder}__")(self, other))
            )
        return ret
    
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

