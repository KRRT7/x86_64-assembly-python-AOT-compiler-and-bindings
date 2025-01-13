import struct
from typing import Literal, TYPE_CHECKING
from aot_refactor.type_imports import *
from aot_refactor.variable import Variable


FUNCTION_ARGUMENTS = (
    [Reg(r, {int}) for r in ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]]
    if current_os == "Linux" else
    [Reg(r, {int}) for r in ["rcx", "rdx", "r8", "r9"]]
)

FUNCTION_ARGUMENTS_BOOL = (
    [Reg(r, {bool}) for r in ["dil", "sil", "dl", "cl", "r8b", "r9b"]]
    if current_os == "Linux" else
    [Reg(r, {bool}) for r in ["cl", "dl", "r8b", "r9b"]]
)

FUNCTION_ARGUMENTS_FLOAT = (
    [Reg(f"xmm{n}", {float}) for n in range(8)]
    if current_os == "Linux" else
    [Reg(f"xmm{n}", {float}) for n in range(4)]
)

def reg_request_float(lines: LinesType) -> Register|OffsetRegister:
    ret:Register|OffsetRegister = Reg.request_float(lines=lines)
    ret.meta_tags.add(float)
    return ret

def reg_request_int(lines: LinesType) -> Register|OffsetRegister:
    ret:Register|OffsetRegister = Reg.request_64(lines=lines)
    ret.meta_tags.add(int)
    return ret

def reg_request_bool(lines: LinesType) -> Register|OffsetRegister:
    ret:Register|OffsetRegister = Reg.request_8(lines=lines)
    ret.meta_tags.add(bool)
    return ret

def str_to_type(string: str) -> ScalarType:
    return {"int": int, "str": str, "float": float}[string]

def load(value: Variable|ScalarType) -> tuple[LinesType, VariableValueType|int|str|Literal[1,0]]:
    """
    Loads the specified value.

    If the value is already a VariableValueType then it will return the value as is.
    """
    from aot_refactor.variable import Variable
    from aot_refactor.function import PythonFunction
    lines: LinesType = [f"LOAD::{value}"]
    if isinstance(value, Variable):
        return lines, value.value
    elif isinstance(value, IntLiteral):
        return lines, IntLiteral(value)
    elif isinstance(value, FloatLiteral):
        float_reg = reg_request_float(lines=lines)
        float_hash = hash(float(value))
        hex_key = f"float_{'np'[float_hash > 0]}{abs(float_hash)}"
        if hex_key not in PythonFunction.jit_program.memory:
            PythonFunction.jit_program.memory[hex_key] = MemorySize.QWORD, [value]
        lines.append(Ins("movsd", float_reg, PythonFunction.jit_program.memory[hex_key].rel))
        return lines, float_reg
    elif isinstance(value, bool):
        return lines, BoolLiteral(value)
    elif value is None:
        raise TypeError("Cannot load None value.")
    else:
        return lines, value

def float_to_hex(f:FloatLiteral) -> str:
    # Pack the float into 8 bytes (64-bit IEEE 754 double precision)
    packed = struct.pack(">d", float(f))  # '>d' for big-endian double
    # Unpack the bytes to get the hexadecimal representation
    hex_rep = "qword 0x" + "".join(f"{b:02x}" for b in packed)
    return FloatLiteral(hex_rep)

def type_from_str(string:str) -> type:
    match string:
        case "int":
            return int
        case "bool":
            return bool
        case "float":
            return float
        case _:
            raise TypeError(f"{string} is not a valid type for python to assembly compilation.")

def type_from_object(obj:ScalarType|VariableValueType|None) -> type:
    if isinstance(obj, Variable):
        return obj.python_type
    elif isinstance(obj, (FloatLiteral, float)):
        return float
    elif isinstance(obj, (IntLiteral, int)):
        return int
    elif isinstance(obj, (BoolLiteral, bool)):
        return bool
    elif isinstance(obj, (Register, OffsetRegister)) and float in obj.meta_tags:
        return float
    elif isinstance(obj, (Register, OffsetRegister)) and int in obj.meta_tags:
        return int
    elif isinstance(obj, (Register, OffsetRegister)) and bool in obj.meta_tags:
        return bool
    else:
        raise TypeError(f"Invalid type {type(obj)}.")
    
class CAST:
    @staticmethod
    def int(value:Variable|VariableValueType|ScalarType) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = [f"CAST::{type(value).__name__} -> int"]
        if isinstance(value, (FloatLiteral, FloatLiteral)):
            return lines, IntLiteral(value)
        elif isinstance(value, Variable) and value.python_type is float:
            return_register = reg_request_int(lines=lines)
            instrs, loaded_value = load(value)
            lines.extend(instrs)
            lines.append(Ins("cvttsd2si", return_register, loaded_value))
            return lines, return_register
        elif isinstance(value, Variable) and value.python_type is bool:
            return_register = reg_request_int(lines=lines)
            instrs, loaded_value = load(value)
            lines.extend(instrs)
            lines.append(Ins("movsx", return_register, loaded_value))
            return lines, return_register
        elif isinstance(value, Register) and float in value.meta_tags:
            return_register = reg_request_int(lines=lines)
            instrs, loaded_value = load(value)
            lines.extend(instrs)
            lines.append(Ins("cvttsd2si", return_register, loaded_value))
            return lines, return_register
        elif isinstance(value, Register) and bool in value.meta_tags:
            return_register = reg_request_int(lines=lines)
            instrs, loaded_value = load(value)
            lines.extend(instrs)
            lines.append(Ins("movsx", return_register, loaded_value))
            return lines, return_register
        else:
            return lines, value
        
    @staticmethod
    def bool(value:Variable|VariableValueType|ScalarType) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = [f"CAST::{type(value).__name__} -> bool"]
        if isinstance(value, IntLiteral):
            return lines, BoolLiteral(value)
        elif isinstance(value, Variable) and value.python_type is float:
            return_register = reg_request_bool(lines=lines)
            instrs, loaded_value = load(value)
            lines.extend(instrs)
            lines.append(Ins("mov", return_register, loaded_value))
            return lines, return_register
        elif isinstance(value, Register) and float in value.meta_tags:
            int_register = reg_request_int(lines=lines)
            instrs, loaded_value = load(value)
            lines.extend(instrs)
            lines.append(Ins("cvttsd2si", int_register, loaded_value))
            return_register = reg_request_bool(lines=lines)
            instrs, loaded_value = load(value)
            lines.append(Ins("mov", return_register, int_register))
            return lines, return_register
        else:
            return lines, value
        
    @staticmethod
    def float(value:Variable|VariableValueType|ScalarType) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = [f"CAST::{type(value).__name__} -> float"]
        if isinstance(value, IntLiteral):
            return lines, FloatLiteral(value)
        elif isinstance(value, Variable) and value.python_type is int:
            return_register = reg_request_float(lines=lines)
            instrs, loaded_value = load(value)
            lines.extend(instrs)
            lines.append(Ins("cvtsi2sd", return_register, loaded_value))
            return lines, return_register
        elif isinstance(value, (Register, OffsetRegister)) and int in value.meta_tags:
            return_register = reg_request_float(lines=lines)
            instrs, loaded_value = load(value)
            lines.extend(instrs)
            lines.append(Ins("cvtsi2sd", return_register, loaded_value))
            return lines, return_register
        else:
            return lines, value