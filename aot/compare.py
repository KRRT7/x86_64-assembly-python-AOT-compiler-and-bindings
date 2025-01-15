from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from aot.type_imports import *
from aot.utils import CAST, load, reg_request_bool, reg_request_float, reg_request_int, type_from_object
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



def implicit_cast_cmp(self, operator:ast.cmpop, left_value:ScalarType|Variable|VariableValueType, right_value:ScalarType|Variable|VariableValueType) -> tuple[type, type, LinesType, VariableValueType|ScalarType, VariableValueType|ScalarType]:
    lines: LinesType = []
    new_left_value = left_value
    new_right_value = right_value
    type_pair = (type_from_object(left_value), type_from_object(right_value))
    if type_pair in {(float, int), (float, bool), (bool, float), (int, float)}:
        if type_pair[0] is not float:
            instrs, new_left_value = CAST.float(left_value, self)
            lines.extend(instrs)
        if type_pair[1] is not float:
            instrs, new_right_value = CAST.float(right_value, self)
            lines.extend(instrs)
        return float, float, lines, new_left_value, new_right_value
    elif type_pair == (bool, bool):
        instrs, new_left_value = CAST.int(left_value)
        lines.extend(instrs)
        instrs, new_right_value = CAST.int(right_value)
        lines.extend(instrs)
        return int, int, lines, new_left_value, new_right_value
    elif type_pair == (int, bool):
        instrs, new_right_value = CAST.int(right_value)
        lines.extend(instrs)
        return int, int, lines, new_left_value, new_right_value
    elif type_pair == (bool, int):
        instrs, new_left_value = CAST.int(left_value)
        lines.extend(instrs)
        return int, int, lines, new_left_value, new_right_value
    
    return *type_pair, lines, left_value, right_value

def __compare_literals(set_cmp_ins_str:str, left_value:ScalarType, right_value:ScalarType) -> tuple[LinesType, VariableValueType|ScalarType]:
    return BoolLiteral({
        "sete": left_value == right_value,
        "setne": left_value != right_value,
        "setl": left_value < right_value,
        "setle": left_value <= right_value,
        "setg": left_value > right_value,
        "setge": left_value >= right_value,
    }[set_cmp_ins_str])
    

def __compare_operator_from_type(self, python_type:type, set_cmp_ins_str:str, left_value:ScalarType|Variable, right_value:ScalarType|Variable) -> tuple[LinesType, VariableValueType|ScalarType]:
    lines: LinesType = []
    # Both are constants
    if python_type is int and isinstance(left_value, IntLiteral) and isinstance(right_value, IntLiteral):
        return lines, __compare_literals(set_cmp_ins_str, left_value, right_value) # compiletime evaluate constants
    elif python_type is float and isinstance(left_value, FloatLiteral) and isinstance(right_value, FloatLiteral):
        return lines, __compare_literals(set_cmp_ins_str, left_value, right_value) # compiletime evaluate constants
    
    instrs, loaded_left_value = load(left_value, self, no_mov=True)
    lines.extend(instrs)
    
    instrs, loaded_right_value = load(right_value, self, no_mov=True)
    lines.extend(instrs)

    cmp_ins = {int:"cmp", bool:"cmp", float:"cmpsd"}[python_type]

    result_memory = reg_request_bool(lines=lines)

    match cmp_ins:
        case "cmp":
            lines.append(Ins("cmp", loaded_left_value, loaded_right_value))
            lines.append(Ins(set_cmp_ins_str, result_memory))
        case "cmpsd":
            float_cmp_result = reg_request_float(lines=lines)
            lines.append(Ins("movsd", float_cmp_result, loaded_left_value))
            lines.append(Ins("cmpsd", float_cmp_result, loaded_right_value, [
                "sete",
                "setl",
                "setle",
                "setnan", # one operand is nan, made up string
                "setne",
                "setge",
                "setg",
                "setnnan" # not nan, made up string
            ].index(set_cmp_ins_str)))
            lines.append(Ins("movmskpd", result_memory.cast_to(MemorySize.DWORD), float_cmp_result))
            float_cmp_result.free(lines=lines)

    
    return lines, result_memory

@add_meta_type(bool)
def compare_operator_from_type(self, type_pair:tuple[type, type], set_cmp_ins_str:str, left:ScalarType|Variable, right:ScalarType|Variable):
    lines: LinesType = []
    if type_pair == (int, int):
        instrs, local_result = __compare_operator_from_type(self, int, set_cmp_ins_str, left, right)
        lines.extend(instrs)
    elif type_pair == (float, float):
        instrs, local_result = __compare_operator_from_type(self, float, set_cmp_ins_str, left, right)
        lines.extend(instrs)

    return lines, local_result
