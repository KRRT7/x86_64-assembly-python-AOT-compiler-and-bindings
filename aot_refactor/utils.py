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
    from aot_refactor.variable import Variable
    lines: LinesType = []
    if isinstance(value, Variable):
        # >> TODO : Make this automatically load floats << #
        return lines, value.value
    elif isinstance(value, int):
        return lines, value
    elif isinstance(value, float):
        return lines, float_to_hex(value)
    elif isinstance(value, bool):
        return lines, int(value)

def float_to_hex(f:float) -> str:
    # Pack the float into 8 bytes (64-bit IEEE 754 double precision)
    packed = struct.pack(">d", f)  # '>d' for big-endian double
    # Unpack the bytes to get the hexadecimal representation
    hex_rep = "qword 0x" + "".join(f"{b:02x}" for b in packed)
    return hex_rep

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