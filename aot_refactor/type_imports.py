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

# Type aliases

VariableValueType = Register|OffsetRegister|StackVariable|OffsetStackVariable

Comment = str

LinesType = list[Instruction | Block | Callable | Comment]

ScalarType = bool|int|float

Reg = Register

RegD = RegisterData

Ins = Instruction

InsD = InstructionData