import ast
from collections import OrderedDict
import struct
from typing import Any, Literal, TYPE_CHECKING
from aot.type_imports import *
from aot.variable import Value, Variable

if TYPE_CHECKING:
    from aot.function import PythonFunction


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

def reg_request_from_type(python_type:type, lines: LinesType, loaded:bool = True) -> Register|OffsetRegister:
    if python_type is float:
        return reg_request_float(lines, loaded)
    elif python_type is int:
        return reg_request_int(lines, loaded)
    elif python_type is bool:
        return reg_request_bool(lines, loaded)
    if isinstance(python_type, Array):
        return reg_request_int(lines, loaded)
    else:
        raise TypeError(f"Type {get_type_name(python_type)} is not supported for requesting registers.")

def reg_request_float(lines: LinesType, loaded:bool = True) -> Register|OffsetRegister:
    ret:Register|OffsetRegister = Reg.request_float(lines=lines)
    ret.meta_tags.add(float)
    if loaded:
        ret.meta_tags.add("loaded")
    return ret

def reg_request_int(lines: LinesType, loaded:bool = True) -> Register|OffsetRegister:
    ret:Register|OffsetRegister = Reg.request_64(lines=lines)
    ret.meta_tags.add(int)
    if loaded:
        ret.meta_tags.add("loaded")
    return ret

def reg_request_bool(lines: LinesType, loaded:bool = True) -> Register|OffsetRegister:
    ret:Register|OffsetRegister = Reg.request_8(lines=lines)
    ret.meta_tags.add(bool)
    if loaded:
        ret.meta_tags.add("loaded")
    return ret

def str_to_type(string: str) -> ScalarType:
    return {"int": int, "str": str, "float": float}[string]

def load(value: Variable|Value|ScalarType, python_function:Any, no_mov:bool = False) -> tuple[LinesType, VariableValueType|int|str|Literal[1,0]]:
    """
    Loads the specified value.

    If the value is already a VariableValueType then it will return the value as is.
    """
    from aot.variable import Variable
    from aot.function import PythonFunction

    python_function: PythonFunction = python_function
    
    jit_program = python_function.jit_program
    lines: LinesType = [f"LOAD::{value}"]
    if isinstance(value, (Value, Variable)):
        if no_mov:
            lines.append(" ^ NOOP")
            return lines, value.value
        elif value.python_type is float:
            float_reg = reg_request_float(lines=lines, loaded=True)
            lines.append(Ins("movsd", float_reg, value.value))
            return lines, float_reg
        elif value.python_type is int:
            int_reg = reg_request_int(lines=lines, loaded=True)
            lines.append(Ins("mov", int_reg, value.value))
            return lines, int_reg
        elif value.python_type is bool:
            bool_reg = reg_request_bool(lines=lines, loaded=True)
            lines.append(Ins("mov", bool_reg, value.value))
            return lines, bool_reg
        elif isinstance(value.python_type, Array):
            int_reg = reg_request_int(lines=lines, loaded=True)
            lines.append(Ins("lea", int_reg, value.value))
            return lines, int_reg
    if isinstance(value, OffsetRegister):
        if no_mov:
            lines.append(" ^ NOOP")
            return lines, value
        elif float in value.meta_tags:
            float_reg = reg_request_float(lines=lines, loaded=True)
            lines.append(Ins("movsd", float_reg, value))
            return lines, float_reg
        elif int in value.meta_tags:
            int_reg = reg_request_int(lines=lines, loaded=True)
            lines.append(Ins("mov", int_reg, value))
            return lines, int_reg
        elif bool in value.meta_tags:
            bool_reg = reg_request_bool(lines=lines, loaded=True)
            lines.append(Ins("mov", bool_reg, value))
            return lines, bool_reg
    elif isinstance(value, (IntLiteral, int)):
        int_reg = reg_request_int(lines=lines, loaded=True)
        lines.append(Ins("mov", int_reg, value))
        return lines, int_reg
    elif isinstance(value, (FloatLiteral, float)):
        float_reg = reg_request_float(lines=lines, loaded=True)
        float_hash = hash(float(value))
        hex_key = f"float_{'np'[float_hash > 0]}{abs(float_hash)}"
        if hex_key not in jit_program.memory:
            jit_program.memory[hex_key] = MemorySize.QWORD, [value]
        lines.append(Ins("movsd", float_reg, jit_program.memory[hex_key].rel))
        return lines, float_reg
    elif isinstance(value, (BoolLiteral, bool)):
        bool_reg = reg_request_bool(lines=lines, loaded=True)
        lines.append(Ins("mov", bool_reg, value))
        return lines, bool_reg
    elif value is None:
        raise TypeError("Cannot load None value.")
    else:
        lines.append(" ^ NOOP")
        return lines, value

def float_to_hex(f:FloatLiteral) -> str:
    # Pack the float into 8 bytes (64-bit IEEE 754 double precision)
    packed = struct.pack(">d", float(f))  # '>d' for big-endian double
    # Unpack the bytes to get the hexadecimal representation
    hex_rep = "qword 0x" + "".join(f"{b:02x}" for b in packed)
    return FloatLiteral(hex_rep)

def cast_literal(value) -> IntLiteral|BoolLiteral|FloatLiteral:
    if isinstance(value, int):
        return IntLiteral(value)
    elif isinstance(value, float):
        return FloatLiteral(value)
    elif isinstance(value, bool):
        return BoolLiteral(value)
    else:
        return value

def type_from_str(key:str|int, templates: OrderedDict[str, type]|None) -> type:
    if templates and key in templates:
        return cast_literal(templates[key])
    elif isinstance(key, ast.Constant):
        return key.value
    match key:
        case "int":
            return int
        case "bool":
            return bool
        case "float":
            return float
        case "Array":
            return Array
        case _:
            raise TypeError(f"{key} is not a valid type for python to assembly compilation.")
        
def type_from_annotation(annotation:ast.Name|ast.Subscript, templates: OrderedDict[str, type]|None = None) -> tuple[type, tuple[type, ...]|type]|type|Array:
    """
    Returns a 'type tuple'
    
    For example:

    `list[int] -> (list, int)`

    `list[int, float] -> (list, ("tuple", int, float))`
    
    `list[list[int]] -> (list, (list, int))`
    """
    if isinstance(annotation, ast.Tuple):
        return ("tuple", *[type_from_annotation(typ, templates) for typ in annotation.elts],)
    elif isinstance(annotation, ast.Name):
        return type_from_str(annotation.id, templates)
    elif isinstance(annotation, ast.Subscript):
        type_tuple = (type_from_annotation(annotation.value, templates), type_from_annotation(annotation.slice, templates))
        if type_tuple[0] is Array:
            return Array.from_type_tuple(type_tuple)
        return type_tuple
    else:
        return type_from_str(annotation, templates)
    
def memory_size_from_type(python_type:type) -> MemorySize:
    if python_type is bool:
        return MemorySize.BYTE.value
    elif python_type in {float, int}:
        return MemorySize.QWORD.value
    elif isinstance(python_type, Array):
        return MemorySize.QWORD.value # size of pointer to array
    elif isinstance(python_type, Template):
        return None
    else:
        raise TypeError(f"Type {get_type_name(python_type)} is not implemented yet.")

def get_type_name(thing:type|object):
    return thing.__name__ if hasattr(thing, "__name__") else type(thing).__name__

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
    elif isinstance(obj, Value):
        return obj.python_type
    else:
        raise TypeError(f"Invalid type {get_type_name(obj)}.")
    
class CAST:
    @staticmethod
    def int(value:Variable|VariableValueType|ScalarType, python_function) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = [f"CAST::{type(value).__name__} -> int"]
        if isinstance(value, (FloatLiteral, FloatLiteral)):
            return lines, IntLiteral(value)
        elif isinstance(value, Variable) and value.python_type is float:
            return_register = reg_request_int(lines=lines, loaded=True)
            instrs, loaded_value = load(value, python_function, no_mov=True)
            lines.extend(instrs)
            lines.append(Ins("cvttsd2si", return_register, loaded_value))
            return lines, return_register
        elif isinstance(value, Variable) and value.python_type is bool:
            return_register = reg_request_int(lines=lines, loaded=True)
            instrs, loaded_value = load(value, python_function, no_mov=True)
            lines.extend(instrs)
            lines.append(Ins("movsx", return_register, loaded_value))
            return lines, return_register
        elif isinstance(value, Register) and float in value.meta_tags:
            return_register = reg_request_int(lines=lines, loaded=True)
            instrs, loaded_value = load(value, python_function, no_mov=True)
            lines.extend(instrs)
            lines.append(Ins("cvttsd2si", return_register, loaded_value))
            return lines, return_register
        elif isinstance(value, Register) and bool in value.meta_tags:
            return_register = reg_request_int(lines=lines, loaded=True)
            instrs, loaded_value = load(value, python_function, no_mov=True)
            lines.extend(instrs)
            lines.append(Ins("movsx", return_register, loaded_value))
            return lines, return_register
        else:
            return lines, value
        
    @staticmethod
    def bool(value:Variable|VariableValueType|ScalarType, python_function) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = [f"CAST::{type(value).__name__} -> bool"]
        if isinstance(value, IntLiteral):
            return lines, BoolLiteral(value)
        elif isinstance(value, Variable) and value.python_type is float:
            return_register = reg_request_bool(lines=lines, loaded=True)
            instrs, loaded_value = load(value, python_function, no_mov=True)
            lines.extend(instrs)
            lines.append(Ins("mov", return_register, loaded_value))
            return lines, return_register
        elif isinstance(value, Register) and float in value.meta_tags:
            int_register = reg_request_int(lines=lines, loaded=True)
            instrs, loaded_value = load(value, python_function, no_mov=True)
            lines.extend(instrs)
            lines.append(Ins("cvttsd2si", int_register, loaded_value))
            return_register = reg_request_bool(lines=lines, loaded=True)
            lines.append(Ins("mov", return_register, int_register))
            return lines, return_register
        else:
            return lines, value
        
    @staticmethod
    def float(value:Variable|VariableValueType|ScalarType, python_function) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = [f"CAST::{type(value).__name__} -> float"]
        if isinstance(value, IntLiteral):
            return lines, FloatLiteral(value)
        elif isinstance(value, Variable) and value.python_type is int:
            return_register = reg_request_float(lines=lines, loaded=True)
            instrs, loaded_value = load(value, python_function, no_mov=True)
            lines.extend(instrs)
            lines.append(Ins("cvtsi2sd", return_register, loaded_value))
            return lines, return_register
        elif isinstance(value, (Register, OffsetRegister)) and int in value.meta_tags:
            return_register = reg_request_float(lines=lines, loaded=True)
            instrs, loaded_value = load(value, python_function, no_mov=True)
            lines.extend(instrs)
            lines.append(Ins("cvtsi2sd", return_register, loaded_value))
            return lines, return_register
        else:
            lines.append(" ^ NOOP")
            return lines, value