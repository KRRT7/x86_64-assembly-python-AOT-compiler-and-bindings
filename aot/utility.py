import ast
import functools
import inspect
import textwrap
import struct

from aot.stack import Stack
from x86_64_assembly_bindings import Instruction, MemorySize, OffsetRegister, Register, RegisterData, current_os

Reg = Register
Ins = Instruction
RegD = RegisterData

rdi = Reg("rdi")
rsi = Reg("rsi")
rdx = Reg("rdx")
rcx = Reg("rcx")
r8 = Reg("r8")
r9 = Reg("r9")

#scratch
r10 = Reg("r10")
r10d = Reg("r10d")
r10b = Reg("r10b")
r11 = Reg("r11")

#mains
rax = Reg("rax")
eax = Reg("eax")
edx = Reg("edx")
rdx = Reg("rdx")
rbp = Reg("rbp")
rsp = Reg("rsp")
ax = Reg("ax")
dx = Reg("dx")
xmm0 = Reg("xmm0")

FUNCTION_ARGUMENTS = [rdi,rsi,rdx,rcx,r8,r9] if current_os == "Linux" else [Reg(r) for r in ["rcx", "rdx", "r8", "r9"]]

FUNCTION_ARGUMENTS_FLOAT = [Reg(f"xmm{n}") for n in range(8)] if current_os == "Linux" else [Reg(f"xmm{n}") for n in range(4)]

def str_to_type(string:str) -> type:
    return {
        "int":int,
        "str":str,
        "float":float
    }[string]

def str_can_cast_int(string:str) -> bool:
    try:
        int(string)
        return True
    except:pass
    return False

def str_is_float(string:str) -> bool:
    parts = string.split(".")
    return "." in string and all(str_can_cast_int(sub_s) for sub_s in parts) and len(parts) == 2

def operand_is_float(v:Register|OffsetRegister|str) -> bool:
    return (isinstance(v, str) and v.startswith("qword 0x")) or (hasattr(v, "name") and v.name.startswith("xmm")) or (hasattr(v, "meta_tags") and "float" in v.meta_tags)

def float_to_hex(f):
    # Pack the float into 8 bytes (64-bit IEEE 754 double precision)
    packed = struct.pack('>d', f)  # '>d' for big-endian double
    # Unpack the bytes to get the hexadecimal representation
    hex_rep = "qword 0x" + ''.join(f'{b:02x}' for b in packed)
    return hex_rep

def load_floats(f, lines:list, ignore:bool = False, stack:Stack = None):
    if ignore:return f
    if isinstance(f, Register) and f.size == MemorySize.QWORD and "float" in f.meta_tags:
        lines.append(" -- LOADING FLOAT")
        lines.append(Ins("movq", ret_f:=Reg.request_float(lines=lines, offset=stack.current.stack_offset), f))
        ret_f.meta_tags.add("float")
        return ret_f
    elif isinstance(f, Register) and not f.name.startswith("xmm"):
        lines.append(" -- LOADING FLOAT")
        lines.append(Ins("movq", ret_f:=Reg.request_float(lines=lines, offset=stack.current.stack_offset), f))
        ret_f.meta_tags.add("float")
        return ret_f
    elif isinstance(f, str) and f.startswith("qword 0x"):
        lines.append(" -- LOADING FLOAT")
        lines.append(Ins("mov", reg64:=Reg.request_64(lines=lines, offset=stack.current.stack_offset), f))
        lines.append(Ins("movq", ret_f:=Reg.request_float(lines=lines, offset=stack.current.stack_offset), reg64))
        ret_f.meta_tags.add("float")
        return ret_f
    else:return f