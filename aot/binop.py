from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from aot.type_imports import *
from aot.utils import CAST, load, reg_request_float, reg_request_int, type_from_object
from aot.variable import Variable

if TYPE_CHECKING:
    from aot.function import PythonFunction

import functools

def add_meta_type(python_type:type):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(reg_res := result[1], (Register, OffsetRegister)):
                reg_res.meta_tags.add(python_type)
                return result[0], reg_res
            else:
                return result
        return wrapper
    return decorator



def implicit_cast(self, operator:ast.operator, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[type, type, LinesType, VariableValueType|ScalarType, VariableValueType|ScalarType]:
    lines: LinesType = []
    new_left_value = left_value
    new_right_value = right_value
    type_pair = (type_from_object(left_value), type_from_object(right_value))
    if type_pair in {(float, int), (float, bool), (bool, float), (int, float)}:
        if isinstance(operator, (ast.Mod, ast.FloorDiv)):
            instrs, new_left_value = CAST.int(left_value, self)
            lines.extend(instrs)
            instrs, new_right_value = CAST.int(right_value, self)
            lines.extend(instrs)
            return float, float, lines, new_left_value, new_right_value
        else:
            instrs, new_left_value = CAST.float(left_value, self)
            lines.extend(instrs)
            instrs, new_right_value = CAST.float(right_value, self)
            lines.extend(instrs)
            return float, float, lines, new_left_value, new_right_value
    elif type_pair == (bool, bool):
        instrs, new_left_value = CAST.int(left_value, self)
        lines.extend(instrs)
        instrs, new_right_value = CAST.int(right_value, self)
        lines.extend(instrs)
        return int, int, lines, new_left_value, new_right_value
    elif type_pair == (int, bool):
        instrs, new_right_value = CAST.int(right_value, self)
        lines.extend(instrs)
        return int, int, lines, new_left_value, new_right_value
    elif type_pair == (bool, int):
        instrs, new_left_value = CAST.int(left_value, self)
        lines.extend(instrs)
        return int, int, lines, new_left_value, new_right_value
    
    return *type_pair, lines, left_value, right_value

@add_meta_type(int)
def add_int_int(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, IntLiteral) and isinstance(right_value, IntLiteral):
        return lines, (left_value + right_value) # compiletime evaluate constants
        
    instrs, result_memory = load(left_value, self)
    lines.extend(instrs)
        
    instrs, loaded_right_value = load(right_value, self, no_mov=True)
    lines.extend(instrs)
    
    lines.append(Ins("add", result_memory, loaded_right_value))
    
    return lines, result_memory

@add_meta_type(float)
def add_float_float(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, FloatLiteral) and isinstance(right_value, FloatLiteral):
        return lines, (left_value + right_value) # compiletime evaluate constants
        
    instrs, result_memory = load(left_value, self)
    lines.extend(instrs)
        
    instrs, loaded_right_value = load(right_value, self, no_mov=True)
    lines.extend(instrs)
    
    lines.append(Ins("addsd", result_memory, loaded_right_value))
    
    return lines, result_memory

@add_meta_type(int)
def sub_int_int(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, IntLiteral) and isinstance(right_value, IntLiteral):
        return lines, (left_value - right_value) # compiletime evaluate constants

    instrs, result_memory = load(left_value, self)
    lines.extend(instrs)
        
    instrs, loaded_right_value = load(right_value, self, no_mov=True)
    lines.extend(instrs)

    lines.append(Ins("sub", result_memory, loaded_right_value))

    return lines, result_memory

@add_meta_type(float)
def sub_float_float(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, FloatLiteral) and isinstance(right_value, FloatLiteral):
        return lines, (left_value - right_value) # compiletime evaluate constants
    
    instrs, result_memory = load(left_value, self)
    lines.extend(instrs)
        
    instrs, loaded_right_value = load(right_value, self, no_mov=True)
    lines.extend(instrs)
    
    lines.append(Ins("subsd", result_memory, loaded_right_value))
    
    return lines, result_memory

@add_meta_type(int)
def mul_int_int(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, IntLiteral) and isinstance(right_value, IntLiteral):
        return lines, (left_value * right_value) # compiletime evaluate constants
    
    instrs, result_memory = load(left_value, self)
    lines.extend(instrs)
        
    instrs, loaded_right_value = load(right_value, self, no_mov=True)
    lines.extend(instrs)
    
    lines.append(Ins("imul", result_memory, loaded_right_value))
    
    return lines, result_memory

@add_meta_type(float)
def mul_float_float(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, FloatLiteral) and isinstance(right_value, FloatLiteral):
        return lines, (left_value * right_value) # compiletime evaluate constants
    
    result_memory = Reg.request_float(lines=lines)
    
    instrs, loaded_left_value = load(left_value, self)
    lines.extend(instrs)
    
    lines.append(Ins("movsd", result_memory, loaded_left_value))
    
    instrs, loaded_right_value = load(right_value, self, no_mov=True)
    lines.extend(instrs)
    
    lines.append(Ins("mulsd", result_memory, loaded_right_value))
    
    return lines, result_memory

@add_meta_type(float)
def div_float_float(self:PythonFunction, left_value:VariableValueType|ScalarType|Variable, right_value:VariableValueType|ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, FloatLiteral) and isinstance(right_value, FloatLiteral):
        return lines, (left_value / right_value) # compiletime evaluate constants
    
    instrs, result_memory = load(left_value, self)
    lines.extend(instrs)
        
    instrs, loaded_right_value = load(right_value, self, no_mov=True)
    lines.extend(instrs)
    
    lines.append(Ins("divsd", result_memory, loaded_right_value))
    
    return lines, result_memory

@add_meta_type(int)
def floordiv_int_int(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, IntLiteral) and isinstance(right_value, IntLiteral):
        return lines, (left_value // right_value) # compiletime evaluate constants
    
    result_memory = Reg("rax", {int})
    
    instrs, loaded_left_value = load(left_value, self)
    lines.extend(instrs)
        
    instrs, loaded_right_value = load(right_value, self, no_mov=True)
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

@add_meta_type(float)
def floordiv_float_float(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, IntLiteral) and isinstance(right_value, IntLiteral):
        return lines, FloatLiteral(left_value // right_value) # compiletime evaluate constants
    
    result_memory = Reg("rax", {int})
    
    instrs, loaded_left_value = load(left_value, self)
    lines.extend(instrs)

    instrs, loaded_right_value = load(right_value, self, no_mov=True)
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
    
    instrs, result_memory = CAST.float(result_memory)
    lines.extend(instrs)

    return lines, result_memory

@add_meta_type(int)
def mod_int_int(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, IntLiteral) and isinstance(right_value, IntLiteral):
        return lines, (left_value % right_value) # compiletime evaluate constants
    
    left_register = Reg("rax", {int})
    
    instrs, loaded_left_value = load(left_value, self)
    lines.extend(instrs)

    instrs, loaded_right_value = load(right_value, self, no_mov=True)
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

    result_reg = reg_request_int(lines=lines)
    lines.append(Ins("mov", result_reg, Reg("rdx", {int})))
                    
    return lines, result_reg

@add_meta_type(float)
def mod_float_float(self:PythonFunction, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    "Incomming arguments will be integer, but it casts the ret to a float"
    lines: LinesType = []
    # Both are constants
    if isinstance(left_value, IntLiteral) and isinstance(right_value, IntLiteral):
        return lines, FloatLiteral(left_value % right_value) # compiletime evaluate constants
    
    left_register = Reg("rax", {int})
    
    instrs, loaded_left_value = load(left_value, self)
    lines.extend(instrs)

    instrs, loaded_right_value = load(right_value, self, no_mov=True)
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
    
    instrs, return_value = CAST.float(Reg("rdx", {int}), self)
    lines.extend(instrs)

    return lines, return_value
