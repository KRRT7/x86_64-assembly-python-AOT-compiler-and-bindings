import struct
from typing import Literal, TYPE_CHECKING
from aot_refactor.type_imports import *
from aot_refactor.variable import Variable


FUNCTION_ARGUMENTS = (
    [Reg(r) for r in ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]]
    if current_os == "Linux" else
    [Reg(r) for r in ["rcx", "rdx", "r8", "r9"]]
)

FUNCTION_ARGUMENTS_FLOAT = (
    [Reg(f"xmm{n}") for n in range(8)]
    if current_os == "Linux" else
    [Reg(f"xmm{n}") for n in range(4)]
)

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
        # >> TODO : Make this automatically load floats << #
        return lines, value.value
    elif isinstance(value, IntLiteral):
        return lines, IntLiteral(value)
    elif isinstance(value, FloatLiteral):
        float_reg = Reg.request_float(lines=lines)
        float_hash = hash(float(value))
        hex_key = f"float_{'np'[float_hash > 0]}{abs(float_hash)}"
        if hex_key not in PythonFunction.jit_program.memory:
            PythonFunction.jit_program.memory[hex_key] = MemorySize.QWORD, [value]
        lines.append(Ins("movsd", float_reg, PythonFunction.jit_program.memory[hex_key].rel))
        return lines, float_reg
    elif isinstance(value, bool):
        return lines, IntLiteral(value)
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

def type_from_object(obj:FloatLiteral|float|int|Variable|None) -> type:
    if isinstance(obj, Variable):
        return obj.python_type
    elif any([isinstance(obj, t) for t in [FloatLiteral, float]]):
        return float
    elif any([isinstance(obj, t) for t in [IntLiteral, int]]):
        return int
    elif isinstance(obj, Register) and obj.is_float:
        return float
    elif isinstance(obj, Register) and not obj.is_float:
        return int
    else:
        raise TypeError(f"Invalid type {type(obj)}.")
    
class CAST:
    @staticmethod
    def int(value:Variable|VariableValueType|ScalarType) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = []
        if isinstance(value, FloatLiteral):
            return lines, IntLiteral(value)
        elif isinstance(value, Variable) and value.python_type is float:
            return_register = Reg.request_64(lines=lines)
            lines.append(Ins("cvttsd2si", return_register, load(value)))
            return lines, return_register
        elif isinstance(value, Register) and value.is_float:
            return_register = Reg.request_64(lines=lines)
            lines.append(Ins("cvttsd2si", return_register, load(value)))
            return lines, return_register
        else:
            return lines, value
        
    @staticmethod
    def float(value:Variable|VariableValueType|ScalarType) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = []
        if isinstance(value, IntLiteral):
            return lines, FloatLiteral(value)
        elif isinstance(value, Variable) and value.python_type is int:
            return_register = Reg.request_float(lines=lines)
            instrs, loaded_value = load(value)
            lines.extend(instrs)
            lines.append(Ins("cvtsi2sd", return_register, loaded_value))
            return lines, return_register
        elif isinstance(value, Register) and not value.is_float:
            return_register = Reg.request_float(lines=lines)
            instrs, loaded_value = load(value)
            lines.extend(instrs)
            lines.append(Ins("cvtsi2sd", return_register, loaded_value))
            return lines, return_register
        else:
            return lines, value