from __future__ import annotations

from typing import TYPE_CHECKING
from aot_refactor.type_imports import *
from aot_refactor.utils import CAST, load
from aot_refactor.variable import Variable

if TYPE_CHECKING:
    from aot_refactor.function import PythonFunction

def implicit_cast(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    if right_value.python_type is float:
        instrs, value = CAST.float(left_value)
        lines.extend(instrs)
        return lines, value
    
    return lines, left_value

def add_int_int(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, IntLiteral) and isinstance(right_value, IntLiteral):
        return lines, (left_value + right_value) # compiletime evaluate constants
    
    result_memory = Reg.request_64(lines=lines)
    
    instrs, loaded_left_value = load(left_value)
    lines.extend(instrs)
    
    lines.append(Ins("mov", result_memory, loaded_left_value))
    
    instrs, loaded_right_value = load(right_value)
    lines.extend(instrs)
    
    lines.append(Ins("add", result_memory, loaded_right_value))
    
    return lines, result_memory

def add_float_float(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, FloatLiteral) and isinstance(right_value, FloatLiteral):
        return lines, (left_value + right_value) # compiletime evaluate constants
    
    result_memory = Reg.request_float(lines=lines)
    
    instrs, loaded_left_value = load(left_value)
    lines.extend(instrs)
    
    lines.append(Ins("movsd", result_memory, loaded_left_value))
    
    instrs, loaded_right_value = load(right_value)
    lines.extend(instrs)
    
    lines.append(Ins("addsd", result_memory, loaded_right_value))
    
    return lines, result_memory

def sub_int_int(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, IntLiteral) and isinstance(right_value, IntLiteral):
        return lines, (left_value - right_value) # compiletime evaluate constants

    result_memory = Reg.request_64(lines=lines)

    instrs, loaded_left_value = load(left_value)
    lines.extend(instrs)

    lines.append(Ins("mov", result_memory, loaded_left_value))

    instrs, loaded_right_value = load(right_value)
    lines.extend(instrs)

    lines.append(Ins("sub", result_memory, loaded_right_value))

    return lines, result_memory

def sub_float_float(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, FloatLiteral) and isinstance(right_value, FloatLiteral):
        return lines, (left_value - right_value) # compiletime evaluate constants
    
    result_memory = Reg.request_float(lines=lines)
    
    instrs, loaded_left_value = load(left_value)
    lines.extend(instrs)
    
    lines.append(Ins("movsd", result_memory, loaded_left_value))
    
    instrs, loaded_right_value = load(right_value)
    lines.extend(instrs)
    
    lines.append(Ins("subsd", result_memory, loaded_right_value))
    
    return lines, result_memory

def mul_int_int(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, IntLiteral) and isinstance(right_value, IntLiteral):
        return lines, (left_value * right_value) # compiletime evaluate constants
    
    result_memory = Reg.request_64(lines=lines)
    
    instrs, loaded_left_value = load(left_value)
    lines.extend(instrs)
    
    lines.append(Ins("mov", result_memory, loaded_left_value))
    
    instrs, loaded_right_value = load(right_value)
    lines.extend(instrs)
    
    lines.append(Ins("imul", result_memory, loaded_right_value))
    
    return lines, result_memory

def mul_float_float(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, FloatLiteral) and isinstance(right_value, FloatLiteral):
        return lines, (left_value * right_value) # compiletime evaluate constants
    
    result_memory = Reg.request_float(lines=lines)
    
    instrs, loaded_left_value = load(left_value)
    lines.extend(instrs)
    
    lines.append(Ins("movsd", result_memory, loaded_left_value))
    
    instrs, loaded_right_value = load(right_value)
    lines.extend(instrs)
    
    lines.append(Ins("mulsd", result_memory, loaded_right_value))
    
    return lines, result_memory

def div_float_float(self:PythonFunction, left_value:VariableValueType|ScalarType|Variable, right_value:VariableValueType|ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, FloatLiteral) and isinstance(right_value, FloatLiteral):
        return lines, (left_value / right_value) # compiletime evaluate constants
    
    result_memory = Reg.request_float(lines=lines)
    
    instrs, loaded_left_value = load(left_value)
    lines.extend(instrs)
    
    lines.append(Ins("movsd", result_memory, loaded_left_value))
    
    instrs, loaded_right_value = load(right_value)
    lines.extend(instrs)
    
    lines.append(Ins("divsd", result_memory, loaded_right_value))
    
    return lines, result_memory

def floordiv_int_int(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, IntLiteral) and isinstance(right_value, IntLiteral):
        return lines, (left_value // right_value) # compiletime evaluate constants
    
    result_memory = Reg("rax")
    
    instrs, loaded_left_value = load(left_value)
    lines.extend(instrs)

    instrs, loaded_right_value = load(right_value)
    lines.extend(instrs)

    if isinstance(loaded_right_value, IntLiteral):
        # mov the second operand into a scratch register if it is an immediate value
        immediate_value_register = Reg.request_64(lines=lines)
        lines.append(Ins("mov", immediate_value_register, loaded_right_value))
        loaded_right_value = immediate_value_register

    lines.extend([
        Ins("mov", result_memory, loaded_left_value),
        Ins("cqo"), # Extend rax sign into rdx
        Ins("idiv", loaded_right_value)
    ])

    # Round down towards negative infinity check:
    floor_round_block = Block(
        prefix=f".{self.name}__BinOp_FloorDiv__round_toward_neg_inf_"
    )
    lines.extend([
        Ins("test", Reg("rdx"), Reg("rdx")),
        Ins("jz", floor_round_block),
        Ins("test", Reg("rax"), Reg("rax")),
        Ins("jns", floor_round_block),
        Ins("sub", Reg("rax"), 1),
        floor_round_block
    ])
                    
    return lines, result_memory

def mod_int_int(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, IntLiteral) and isinstance(right_value, IntLiteral):
        return lines, (left_value // right_value) # compiletime evaluate constants
    
    left_register = Reg("rax")
    
    instrs, loaded_left_value = load(left_value)
    lines.extend(instrs)

    instrs, loaded_right_value = load(right_value)
    lines.extend(instrs)

    if isinstance(loaded_right_value, IntLiteral):
        # mov the second operand into a scratch register if it is an immediate value
        immediate_value_register = Reg.request_64(lines=lines)
        lines.append(Ins("mov", immediate_value_register, loaded_right_value))
        loaded_right_value = immediate_value_register

    lines.extend([
        Ins("mov", left_register, loaded_left_value),
        Ins("cqo"), # Extend rax sign into rdx
        Ins("idiv", loaded_right_value)
    ])
                    
    return lines, Reg("rdx")