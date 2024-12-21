from __future__ import annotations
from x86_64_assembly_bindings import (
    Register,
    Instruction,
    MemorySize,
    Program,
    Block,
    Function,
    OffsetRegister,
    Variable,
    RegisterData,
    InstructionData,
    Memory,
    current_os,
)
import ast
import functools
import inspect
import textwrap
import struct
from time import time, perf_counter_ns

Reg = Register
Ins = Instruction
RegD = RegisterData

rdi = Reg("rdi")
rsi = Reg("rsi")
rdx = Reg("rdx")
rcx = Reg("rcx")
r8 = Reg("r8")
r9 = Reg("r9")

# scratch
r10 = Reg("r10")
r10d = Reg("r10d")
r10b = Reg("r10b")
r11 = Reg("r11")

# mains
rax = Reg("rax")
eax = Reg("eax")
edx = Reg("edx")
rdx = Reg("rdx")
rbp = Reg("rbp")
rsp = Reg("rsp")
ax = Reg("ax")
dx = Reg("dx")
xmm0 = Reg("xmm0")

function_arguments = (
    [rdi, rsi, rdx, rcx, r8, r9]
    if current_os == "Linux"
    else [Reg(r) for r in ["rcx", "rdx", "r8", "r9"]]
)

float_function_arguments = (
    [Reg(f"xmm{n}") for n in range(8)]
    if current_os == "Linux"
    else [Reg(f"xmm{n}") for n in range(4)]
)


def str_to_type(string: str) -> type:
    return {"int": int, "str": str, "float": float}[string]


def str_can_cast_int(string: str) -> bool:
    try:
        int(string)
        return True
    except:
        pass
    return False


def str_is_float(string: str) -> bool:
    parts = string.split(".")
    return (
        "." in string
        and all(str_can_cast_int(sub_s) for sub_s in parts)
        and len(parts) == 2
    )


def operand_is_float(v: Register | OffsetRegister | str) -> bool:
    return (
        (isinstance(v, str) and v.startswith("qword 0x"))
        or (hasattr(v, "name") and v.name.startswith("xmm"))
        or (hasattr(v, "meta_tags") and "float" in v.meta_tags)
    )


def float_to_hex(f):
    # Pack the float into 8 bytes (64-bit IEEE 754 double precision)
    packed = struct.pack(">d", f)  # '>d' for big-endian double
    # Unpack the bytes to get the hexadecimal representation
    hex_rep = "qword 0x" + "".join(f"{b:02x}" for b in packed)
    return hex_rep


def load_floats(f, lines: list, ignore: bool = False, stack: Stack = None):
    if ignore:
        return f
    if (
        isinstance(f, Register)
        and f.size == MemorySize.QWORD
        and "float" in f.meta_tags
    ):
        lines.append(" -- LOADING FLOAT")
        lines.append(
            Ins(
                "movq",
                ret_f := Reg.request_float(
                    lines=lines, offset=stack.current.stack_offset
                ),
                f,
            )
        )
        ret_f.meta_tags.add("float")
        return ret_f
    elif isinstance(f, Register) and not f.name.startswith("xmm"):
        lines.append(" -- LOADING FLOAT")
        lines.append(
            Ins(
                "movq",
                ret_f := Reg.request_float(
                    lines=lines, offset=stack.current.stack_offset
                ),
                f,
            )
        )
        ret_f.meta_tags.add("float")
        return ret_f
    elif isinstance(f, str) and f.startswith("qword 0x"):
        lines.append(" -- LOADING FLOAT")
        lines.append(
            Ins(
                "mov",
                reg64 := Reg.request_64(lines=lines, offset=stack.current.stack_offset),
                f,
            )
        )
        lines.append(
            Ins(
                "movq",
                ret_f := Reg.request_float(
                    lines=lines, offset=stack.current.stack_offset
                ),
                reg64,
            )
        )
        ret_f.meta_tags.add("float")
        return ret_f
    else:
        return f


class Var:
    def __init__(
        self, stack_frame: StackFrame, name: str, size: MemorySize, py_type: type = int
    ):
        self.name = name
        self.size = size
        self.type = py_type
        self.stack_frame = stack_frame

    def cast(self, lines: list[Instruction | Block], py_type: type = int) -> Register:
        if py_type == float:
            lines.push(
                Ins(
                    "cvtsi2sd",
                    fpr := Reg.request_float(
                        lines=lines, offset=self.stack_frame.stack_offset
                    ),
                    self.get(),
                )
            )
            return fpr
        elif py_type == int:
            lines.push(
                Ins(
                    "cvttsd2si",
                    r := Reg.request_64(
                        lines=lines, offset=self.stack_frame.stack_offset
                    ),
                    self.get(),
                )
            )
            return r

    def get(self) -> OffsetRegister:
        return self.stack_frame[self.name]


class StackFrame:
    def __init__(self):
        self.variables: list[Var] = []

    @property
    def size(self):
        return len(self.variables)

    @property
    def stack_offset(self):
        offset = 0
        for v in self.variables:
            offset += v.size.value // 8
        return offset

    def alloca(
        self, name: str, size: MemorySize = MemorySize.QWORD, py_type: type = int
    ) -> Instruction:
        self.variables.append(Var(self, name, size, py_type))
        return Ins("sub", rsp, size.value // 8)

    def pop(self) -> Instruction | None:
        if self.stack_offset != 0:
            return Ins("add", rsp, self.stack_offset)
        return None

    def __contains__(self, key: str) -> bool:
        return key in self.variables

    def __getitem__(self, key: str) -> OffsetRegister:
        offset = 0
        for v in self.variables:
            offset += v.size.value // 8
            if v.name == key:
                return OffsetRegister(rbp, offset, True)
        raise KeyError(f'Variable "{key}" not found in stack frame.')

    def getvar(self, key: str) -> Var:
        for v in self.variables:
            if v.name == key:
                return v
        raise KeyError(f'Variable "{key}" not found in stack frame.')

    def add_variable(self, name: str, var: Var):
        self.offset += var.size.value // 8
        self.variables[name] = OffsetRegister(rbp, self.offset, True)


class Stack:
    def __init__(self):
        self.stack = [StackFrame()]
        self.cursor = -1
        self.push()
        self.__origin = True

    def get_is_origin(self):
        "This returns true only on the first ever call."
        so = self.__origin
        self.__origin = False
        return so

    @property
    def current(self) -> StackFrame:
        return self.stack[self.cursor]

    def alloca(
        self, name: str, size: MemorySize = MemorySize.QWORD, py_type: type = int
    ) -> Instruction:
        return self.current.alloca(name, size, py_type)

    def push(self):
        self.stack.append(StackFrame())
        self.cursor += 1

    def pop(self) -> Instruction | None:
        r = self.current.pop()
        self.cursor -= 1
        return r

    def __contains__(self, key: str) -> bool:
        for frame in self.stack:
            if key in frame:
                return True
        return False

    def __getitem__(self, key: str) -> OffsetRegister:
        for frame in reversed(self.stack[0 : self.cursor + 1]):

            if key in frame:
                return frame[key]
        raise KeyError(f'Variable "{key}" not found in function stack.')

    def getvar(self, key: str) -> Var:
        for frame in reversed(self.stack[0 : self.cursor + 1]):

            if key in frame:
                return frame.getvar(key)
        raise KeyError(f'Variable "{key}" not found in function stack.')


class PythonFunction:
    jit_prog: Program = Program("python_x86_64_jit")
    name: str
    arguments_dict: dict[str, Register | MemorySize]
    arguments: tuple[str, Register | MemorySize]
    lines: list[Instruction]
    python_ast: ast.FunctionDef
    ret: Reg | None

    def __init__(self, python_ast: ast.FunctionDef, stack: Stack):
        self.compiled = False
        self.python_ast = python_ast
        self.name = python_ast.name
        self.stack = stack
        self.arguments_dict = {}
        self.arguments: list[Register] = []
        self.arguments_type: list[type] = []
        self.arguments_type_dict: dict[str, type] = {}
        self.ret_py_type = None
        self.ret = None
        if self.python_ast.returns:
            match self.python_ast.returns.id:
                case "int":
                    self.ret = rax
                    self.ret_py_type = int
                case "float":
                    self.ret = Reg("xmm0")
                    self.ret_py_type = float
                case _:
                    raise SyntaxError(
                        f'Unsupported return type "{python_ast.returns.id}" for decorated function.'
                    )
        self.signed_args: set[int] = set()
        self.stack_arguments: list[Instruction] = []
        callee_saved_ref = [0]
        for a_n, argument in enumerate(python_ast.args.args):
            self.arguments_type_dict[argument.arg] = a_type = str_to_type(
                argument.annotation.id
            )
            self.arguments_type.append(a_type)
            final_arg = None
            match a_type.__name__:
                case "int":
                    if a_n < len(function_arguments):
                        final_arg = function_arguments[a_n]
                    else:
                        final_arg = OffsetRegister(
                            rbp,
                            (
                                lambda s, cs: (a_n + 2 + cs[0] - s) * 8,
                                [self.stack.current.size, callee_saved_ref],
                            ),
                            meta_tags={"int"},
                        )
                case "float":
                    if a_n < len(float_function_arguments):
                        final_arg = float_function_arguments[a_n]
                    else:
                        final_arg = OffsetRegister(
                            rbp,
                            (
                                lambda s, cs: (a_n + 2 + cs[0] - s) * 8,
                                [self.stack.current.size, callee_saved_ref],
                            ),
                            meta_tags={"float"},
                        )
                    self.signed_args.add(a_n)
            self.arguments_dict[argument.arg] = final_arg
            self.arguments.append(final_arg)
        self.function = Function(
            self.arguments,
            return_register=self.ret,
            label=self.name,
            return_signed=True,
            ret_py_type=self.ret_py_type,
            signed_args=self.signed_args,
            arguments_py_type=self.arguments_type,
        )
        self.gen_ret = lambda: self.function.ret
        self.is_stack_origin = self.stack.get_is_origin()
        self.lines, _ = self.gen_stmt(self.python_ast.body)
        callee_saved_ref[0] = len(self.function.callee_saved_regs)

    def __call__(self):
        self.function()
        if self.is_stack_origin:
            Ins("mov", rbp, rsp)()
            # Ins("sub", rsp, 8)()
        for line in self.lines:
            if line:
                if isinstance(line, str):
                    self.jit_prog.comment(line)
                else:
                    line()

        if hasattr(line, "name") and line.name != "return":
            if pi := self.stack.pop():
                pi()
            self.return_value()[0]()

        return self

    def return_value(self, ret_value: any = None) -> list[Instruction]:
        r = []
        match self.ret_py_type.__name__:
            case "int":
                r = (
                    [Ins("mov", self.ret, ret_value)]
                    if ret_value and self.ret.name != str(ret_value)
                    else []
                )
            case "float":
                r = []
                if ret_value:
                    f = load_floats(ret_value, r, stack=self.stack)
                    if self.ret.name != str(f):

                        r.append(Ins("movq", self.ret, f))
            case _:
                r = []
        if self.is_stack_origin:
            r.append(Ins("mov", rsp, rbp))
        else:
            r.append(self.stack.pop())
            self.stack.cursor += 1
        r.append(self.gen_ret())
        return r

    def gen_stmt(
        self, body: list[ast.stmt], loop_break_block: Block | None = None
    ) -> tuple[list[Instruction], Register | Block | None]:
        lines: list[Instruction] = []

        sec_ret = None
        for stmt in body:
            Register.free_all(lines)
            lines.append("    FREED SCRATCH MEMORY")
            match stmt.__class__.__name__:
                case "Assign":
                    lines.append("STMT::Assign")
                    _instrs, value = self.gen_expr(stmt.value)
                    lines.extend(_instrs)

                    for target in stmt.targets:
                        _instrs, key = self.gen_expr(target, py_type=stmt.type_comment)
                        lines.extend(_instrs)

                        if k_is_str := isinstance(key, str):
                            lines.append(self.stack.alloca(key))

                        dest = self.stack[key] if k_is_str else key

                        if str(dest) != str(value):
                            if type(value) in {Variable, OffsetRegister}:
                                lines.append(
                                    Ins(
                                        "mov",
                                        r64 := Reg.request_64(
                                            lines=lines,
                                            offset=self.stack.current.stack_offset,
                                        ),
                                        value,
                                    )
                                )
                                value = r64
                            value = load_floats(
                                value,
                                lines,
                                not dest.name.startswith("xmm"),
                                stack=self.stack,
                            )
                            lines.append(
                                Ins(
                                    "movq" if operand_is_float(value) else "mov",
                                    dest,
                                    value,
                                )
                            )
                case "AugAssign":
                    lines.append("STMT::AugAssign")
                    stmt: ast.AugAssign
                    _instrs, value = self.gen_expr(stmt.value)
                    lines.extend(_instrs)

                    _instrs, key = self.gen_expr(stmt.target)
                    lines.extend(_instrs)

                    dest = self.stack[key] if isinstance(key, str) else key

                    if type(value) in {Variable, OffsetRegister}:
                        lines.append(
                            Ins(
                                "mov",
                                r64 := Reg.request_64(
                                    lines=lines, offset=self.stack.current.stack_offset
                                ),
                                value,
                            )
                        )
                        value = r64
                    value = load_floats(
                        value, lines, not dest.name.startswith("xmm"), stack=self.stack
                    )
                    _pyt = int
                    if any(
                        (isinstance(v, str) and v.startswith("0x"))
                        for v in [value, dest]
                    ) or any(
                        (isinstance(v, Register) and v.name.startswith("xmm"))
                        for v in [value, dest]
                    ):
                        _pyt = float
                    # TODO call gen_operator with aug_assign flag
                    _instrs, _ = self.gen_operator(dest, stmt.op, value, _pyt, True)
                    lines.extend(_instrs)

                case "AnnAssign":
                    lines.append("STMT::AnnAssign")
                    stmt: ast.AnnAssign
                    _instrs, value = self.gen_expr(stmt.value)
                    lines.extend(_instrs)

                    target = stmt.target

                    alloca_type = int
                    match stmt.annotation.id:
                        case "int":
                            alloca_type = int
                        case "float":
                            alloca_type = float

                    _instrs, key = self.gen_expr(target, py_type=alloca_type)
                    lines.extend(_instrs)

                    if k_is_str := isinstance(key, str):
                        lines.append(self.stack.alloca(key, py_type=alloca_type))

                    dest = self.stack[key] if k_is_str else key
                    if str(dest) != str(value):
                        if type(value) in {Variable, OffsetRegister}:
                            lines.append(
                                Ins(
                                    "mov",
                                    r64 := Reg.request_64(
                                        lines=lines,
                                        offset=self.stack.current.stack_offset,
                                    ),
                                    value,
                                )
                            )
                            value = r64
                        value = load_floats(
                            value,
                            lines,
                            alloca_type.__name__ == "int",
                            stack=self.stack,
                        )
                        lines.append(
                            Ins(
                                "movq" if operand_is_float(value) else "mov",
                                dest,
                                value,
                            )
                        )

                case "Return":
                    lines.append("STMT::Return")
                    stmt: ast.Return

                    _instrs, value = self.gen_expr(stmt.value)
                    lines.extend(_instrs)

                    lines.extend(self.return_value(value))

                case "If":
                    lines.append("STMT::If")
                    false_block = Block()
                    else_ins, false_block_maybe = self.gen_stmt(
                        stmt.orelse, loop_break_block=loop_break_block
                    )
                    if false_block_maybe:
                        false_block = false_block_maybe

                    sc_block = Block()
                    cond_instrs, cond_val = self.gen_expr(
                        stmt.test, block=false_block, sc_block=sc_block
                    )

                    if_bod, _ = self.gen_stmt(
                        stmt.body, loop_break_block=loop_break_block
                    )
                    sec_ret = Block()
                    end_block = Block()
                    lines.extend(
                        [
                            sec_ret,
                            *cond_instrs,
                            Ins("test", cond_val, cond_val),
                            Ins("jz", false_block),
                            sc_block,
                            *if_bod,
                            Ins("jmp", end_block),
                            false_block,
                            *else_ins,
                            end_block,
                        ]
                    )

                case "Break":
                    lines.append("STMT::Break")
                    lines.append(Ins("jmp", loop_break_block))

                case "While":
                    lines.append("STMT::While")
                    stmt: ast.While
                    false_block = Block(prefix=f".{self.name}__while__false_")
                    else_ins, false_block_maybe = self.gen_stmt(stmt.orelse)
                    if false_block_maybe:
                        false_block = false_block_maybe

                    sc_block = Block(prefix=f".{self.name}__while__short_circuit_")
                    cond_instrs, cond_val = self.gen_expr(
                        stmt.test, block=false_block, sc_block=sc_block
                    )

                    sec_ret = Block(prefix=f".{self.name}__while__start_")
                    end_block = Block(prefix=f".{self.name}__while__else_end_")

                    while_bod, _ = self.gen_stmt(
                        stmt.body,
                        loop_break_block=end_block if len(else_ins) else false_block,
                    )
                    lines.extend(
                        [
                            sec_ret,
                            *cond_instrs,
                            Ins("test", cond_val, cond_val),
                            Ins("jz", false_block),
                            sc_block,
                            *while_bod,
                            Ins("jmp", sec_ret),
                            false_block,
                            *else_ins,
                            end_block if len(else_ins) else " ~ NO ELSE",
                        ]
                    )

        return lines, sec_ret

    def gen_operator(
        self,
        val1: Register | Variable,
        op: ast.operator,
        val2: Register | Variable | int,
        py_type: type = int,
        aug_assign: bool = False,
    ) -> tuple[list[Instruction | Block], Register]:
        # TODO add aug_assign parameter flag for when an AugAssign is evaluated to store result in val1 to reduce instruction count.
        # py_type specifies what the resultant type should be
        lines = []
        res = None
        is_float = py_type.__name__ == "float" or any(
            operand_is_float(v) for v in [val1, val2]
        )
        if is_float:
            py_type = float
        req_reg = lambda: (
            Reg.request_float(lines=lines, offset=self.stack.current.stack_offset)
            if is_float
            else Reg.request_64(lines=lines, offset=self.stack.current.stack_offset)
        )

        match op.__class__.__name__:
            case "Add":
                lines.append(f"BinOp::Add{f'(AUG)' if aug_assign else ''}")
                if aug_assign:
                    lines.append(
                        Ins(
                            InstructionData.from_py_type("add", py_type),
                            nval1 := load_floats(
                                val1, lines, not is_float, stack=self.stack
                            ),
                            load_floats(val2, lines, not is_float, stack=self.stack),
                        )
                    )
                    if nval1 != val1:
                        lines.append(Ins("movq", val1, nval1))
                else:
                    if not is_float:
                        lines.append(Ins("mov", nval1 := rax, val1))
                        val1 = nval1
                    elif (
                        isinstance(val1, Register) and val1 in float_function_arguments
                    ):
                        lines.append(Ins("movq", nval1 := req_reg(), val1))
                        val1 = nval1
                    lines.append(
                        Ins(
                            InstructionData.from_py_type("add", py_type),
                            nval1 := load_floats(
                                val1, lines, not is_float, stack=self.stack
                            ),
                            v2 := load_floats(
                                val2, lines, not is_float, stack=self.stack
                            ),
                        )
                    )
                    if str(val2) != str(v2):
                        v2.free(lines)
                        lines.append(f"    FREED ({v2})")

                    if nval1 == rax:
                        lines.append(Ins("mov", nval1 := req_reg(), val1))
                    val1 = nval1
                    res = val1
            case "Sub":
                lines.append(f"BinOp::Sub{f'(AUG)' if aug_assign else ''}")
                if aug_assign:
                    lines.append(
                        Ins(
                            InstructionData.from_py_type("sub", py_type),
                            nval1 := load_floats(
                                val1, lines, not is_float, stack=self.stack
                            ),
                            load_floats(val2, lines, not is_float, stack=self.stack),
                        )
                    )
                    if nval1 != val1:
                        lines.append(Ins("movq", val1, nval1))
                else:
                    if not is_float:
                        lines.append(Ins("mov", nval1 := rax, val1))
                        val1 = nval1
                    elif (
                        isinstance(val1, Register) and val1 in float_function_arguments
                    ):
                        lines.append(Ins("movq", nval1 := req_reg(), val1))
                        val1 = nval1
                    lines.append(
                        Ins(
                            InstructionData.from_py_type("sub", py_type),
                            nval1 := load_floats(
                                val1, lines, not is_float, stack=self.stack
                            ),
                            v2 := load_floats(
                                val2, lines, not is_float, stack=self.stack
                            ),
                        )
                    )
                    if str(val2) != str(v2):
                        v2.free(lines)
                        lines.append(f"    FREED ({v2})")

                    if nval1 == rax:
                        lines.append(Ins("mov", nval1 := req_reg(), val1))
                    val1 = nval1
                    res = val1
            case "Mult":
                if is_float:
                    lines.append(f"BinOp::Mult(FLOAT){f'(AUG)' if aug_assign else ''}")
                    if aug_assign:
                        lines.append(
                            Ins(
                                "mulsd",
                                nval1 := load_floats(val1, lines, stack=self.stack),
                                v2 := load_floats(
                                    val2,
                                    lines,
                                    not operand_is_float(val2),
                                    stack=self.stack,
                                ),
                            )
                        )
                        if str(val2) != str(v2):
                            v2.free(lines)
                            lines.append(f"    FREED ({v2})")
                        if nval1 != val1:
                            lines.append(Ins("movq", val1, nval1))
                    else:
                        if (
                            isinstance(val1, Register)
                            and val1 in float_function_arguments
                        ):
                            lines.append(Ins("movq", nval1 := req_reg(), val1))
                            potential_temp_r = val1 = nval1
                        lines.append(
                            Ins(
                                "mulsd",
                                nval1 := load_floats(
                                    val1,
                                    lines,
                                    not operand_is_float(val1),
                                    stack=self.stack,
                                ),
                                v2 := load_floats(
                                    val2,
                                    lines,
                                    not operand_is_float(val2),
                                    stack=self.stack,
                                ),
                            )
                        )
                        if str(val2) != str(v2):
                            v2.free(lines)
                            lines.append(f"    FREED ({v2})")
                        val1 = nval1
                        res = val1
                else:
                    lines.append(
                        f"BinOp::Mult(INTEGER){f'(AUG)' if aug_assign else ''}"
                    )
                    if aug_assign:
                        lines.append(Ins("imul", val1, val2))
                    else:
                        if str(val1) != str(rax):
                            lines.append(Ins("mov", rax, val1))
                            val1 = rax
                        lines.append(Ins("imul", val1, val2))

                        lines.append(
                            Ins(
                                "mov",
                                nres := Reg.request_64(
                                    lines=lines, offset=self.stack.current.stack_offset
                                ),
                                val1,
                            )
                        )
                        res = nres
            case "FloorDiv":
                lines.append(f"BinOp::FloorDiv{f'(AUG)' if aug_assign else ''}")

                original_val1 = val1
                if str(val1) != str(rax):
                    lines.append(Ins("mov", rax, val1))
                    val1 = rax
                lines.append(Ins("cqo"))
                if isinstance(val2, int):
                    lines.append(
                        Ins(
                            "mov",
                            val2 := Reg.request_64(
                                lines=lines, offset=self.stack.current.stack_offset
                            ),
                            val2,
                        )
                    )
                lines.append(Ins("idiv", val2))
                lines.append(Ins("test", rax, rax))
                floor_round_block = Block(
                    prefix=f".{self.name}__BinOp_FloorDiv__round_toward_neg_inf_"
                )
                lines.append(Ins("jns", floor_round_block))
                lines.append(Ins("sub", rax, 1))
                lines.append(floor_round_block)

                if aug_assign:
                    lines.append(Ins("mov", original_val1, rax))
                else:
                    lines.append(
                        Ins(
                            "mov",
                            res := Reg.request_64(
                                lines=lines, offset=self.stack.current.stack_offset
                            ),
                            rax,
                        )
                    )
            case "Div":
                lines.append(f"BinOp::Div{f'(AUG)' if aug_assign else ''}")
                if aug_assign:
                    lines.append(
                        Ins(
                            InstructionData.from_py_type("div", float),
                            nval1 := load_floats(
                                val1, lines, not is_float, stack=self.stack
                            ),
                            load_floats(val2, lines, not is_float, stack=self.stack),
                        )
                    )
                    if nval1 != val1:
                        lines.append(Ins("movq", val1, nval1))
                else:
                    if not is_float:
                        lines.append(Ins("mov", nval1 := req_reg(), val1))
                        val1 = nval1
                    elif (
                        isinstance(val1, Register) and val1 in float_function_arguments
                    ):
                        lines.append(Ins("movq", nval1 := req_reg(), val1))
                        val1 = nval1
                    lines.append(
                        Ins(
                            InstructionData.from_py_type("div", float),
                            nval1 := load_floats(
                                val1, lines, not is_float, stack=self.stack
                            ),
                            v2 := load_floats(
                                val2, lines, not is_float, stack=self.stack
                            ),
                        )
                    )
                    if str(val2) != str(v2):
                        v2.free(lines)
                        lines.append(f"    FREED ({v2})")
                    val1 = nval1
                    res = val1

            case "Mod":
                lines.append(f"BinOp::Mod{f'(AUG)' if aug_assign else ''}")
                original_val1 = val1
                lines.append(Ins("xor", rdx, rdx))
                if str(val1) != str(rax):
                    lines.append(Ins("mov", rax, val1))
                    val1 = rax
                lines.append(Ins("cqo"))
                if isinstance(val2, int):
                    lines.append(
                        Ins(
                            "mov",
                            des_r := Reg.request_64(
                                lines=lines, offset=self.stack.current.stack_offset
                            ),
                            val2,
                        )
                    )
                    val2 = des_r
                lines.append(Ins("idiv", val2))
                if aug_assign:
                    lines.append(Ins("mov", original_val1, rdx))
                else:
                    lines.append(
                        Ins(
                            "mov",
                            res := Reg.request_64(
                                lines=lines, offset=self.stack.current.stack_offset
                            ),
                            rdx,
                        )
                    )

        lines.append(f"BinOp::Expr::RETURN({res})")
        return lines, res

    def gen_cmp_operator(
        self,
        val1: Register | Variable,
        op: ast.cmpop,
        val2: Register | Variable | int,
        py_type: type | str = int,
    ) -> tuple[list[Instruction | Block], Register]:
        res = Reg.request_8()
        is_float = py_type.__name__ == "float" or any(
            operand_is_float(v) for v in [val1, val2]
        )
        lines = [
            Ins("xor", res.cast_to(MemorySize.QWORD), res.cast_to(MemorySize.QWORD))
        ]
        lines.append(
            Ins(
                InstructionData.from_py_type("cmp", float if is_float else int),
                load_floats(val1, lines, not is_float, stack=self.stack),
                load_floats(val2, lines, not is_float, stack=self.stack),
            )
        )

        match op.__class__.__name__:
            case "Eq":
                lines.append('CmpOp::Eq|"=="')
                lines.append(Ins("sete", res))
            case "NotEq":
                lines.append('CmpOp::NotEq|"!="')
                lines.append(Ins("setne", res))
            case "Lt":
                lines.append('CmpOp::Lt|"<"')
                lines.append(Ins("setl", res))
            case "LtE":
                lines.append('CmpOp::LtE|"<="')
                lines.append(Ins("setle", res))
            case "Gt":
                lines.append('CmpOp::Gt|">"')
                lines.append(Ins("setg", res))
            case "GtE":
                lines.append('CmpOp::GtE|">="')
                lines.append(Ins("setge", res))

            case "In":
                pass

            case "NotIn":
                pass

        return lines, res.cast_to(MemorySize.QWORD)

    def get_var(
        self, name: str, lines: list | None = None, allow_float_load: bool = False
    ) -> Register | str:
        try:
            if name in self.arguments_dict:
                return self.arguments_dict[name]
            else:
                v = self.stack.getvar(name)
                r = None
                if (
                    v.type.__name__ == "float"
                    and lines is not None
                    and allow_float_load
                ):
                    r = load_floats(v.get(), lines, stack=self.stack)
                return r if r else v.get()
        except KeyError:
            return name

    def gen_expr(
        self,
        expr: ast.expr,
        py_type: type = int,
        block: Block | None = None,
        sc_block: Block | None = None,
    ) -> tuple[list[Instruction], any]:
        lines = []
        sec_ret = None
        match expr.__class__.__name__:
            case "Constant":
                expr: ast.Constant
                if isinstance(expr.value, int):
                    sec_ret = int(expr.value)
                elif isinstance(expr.value, float):
                    sec_ret = float_to_hex(expr.value)
            case "Name":
                expr: ast.Name
                lines.append(f'label::"{expr.id}"')
                sec_ret = self.get_var(
                    expr.id, lines, not isinstance(expr.ctx, ast.Store)
                )
            case "BinOp":
                expr: ast.BinOp
                _instrs, val1 = self.gen_expr(expr.left)
                lines.extend(_instrs)
                _instrs, val2 = self.gen_expr(expr.right)
                lines.extend(_instrs)
                _pyt = py_type
                if any(
                    (isinstance(v, str) and v.startswith("0x")) for v in [val1, val2]
                ) or any(
                    (isinstance(v, Register) and v.name.startswith("xmm"))
                    for v in [val1, val2]
                ):
                    _pyt = float
                _instrs, res = self.gen_operator(val1, expr.op, val2, _pyt)
                lines.extend(_instrs)
                sec_ret = res

            case "BoolOp":
                expr: ast.BoolOp
                bres = Register.request_8()
                lines.append(
                    Ins(
                        "xor",
                        bres.cast_to(MemorySize.QWORD),
                        bres.cast_to(MemorySize.QWORD),
                    )
                )
                match expr.op.__class__.__name__:
                    case "And":
                        lines.append("BoolOp::AND")
                        local_sc_b = Block()
                        first_operand = None
                        for boperand in expr.values:
                            _instrs, operand = self.gen_expr(boperand)
                            lines.extend(_instrs)

                            if first_operand:
                                lines.append(
                                    Ins(
                                        InstructionData.from_py_type("and", py_type),
                                        first_operand,
                                        operand,
                                    )
                                )

                            if block:
                                lines.append(Ins("jz", block))
                            else:
                                lines.append(
                                    Ins("jz", local_sc_b)
                                )  # Jump to the short circuit assign

                            if not first_operand:
                                first_operand = operand
                        if not block:
                            lines.append(local_sc_b)

                        lines.append(Ins("setnz", bres))
                    case "Or":
                        lines.append("BoolOp::OR")
                        local_sc_b = Block()
                        first_operand = None
                        for boperand in expr.values:
                            _instrs, operand = self.gen_expr(boperand)
                            lines.extend(_instrs)

                            if first_operand:
                                lines.append(
                                    Ins(
                                        InstructionData.from_py_type("or", py_type),
                                        first_operand,
                                        operand,
                                    )
                                )

                            if block:
                                lines.append(Ins("jnz", sc_block))
                            else:
                                lines.append(
                                    Ins("jnz", local_sc_b)
                                )  # Jump to the short circuit assign

                            if not first_operand:
                                first_operand = operand
                        if not block:
                            lines.append(local_sc_b)

                        lines.append(Ins("setnz", bres))

                sec_ret = bres.cast_to(MemorySize.QWORD)

            case "Compare":
                expr: ast.Compare
                _instrs, val1 = self.gen_expr(expr.left, block=block, sc_block=sc_block)
                lines.extend(_instrs)

                for op_i, op in enumerate(expr.ops):

                    _instrs, val2 = self.gen_expr(
                        expr.comparators[op_i], block=block, sc_block=sc_block
                    )
                    lines.extend(_instrs)
                    _instrs, val1 = self.gen_cmp_operator(val1, op, val2)
                    lines.extend(_instrs)

                sec_ret = val1

        return lines, sec_ret


PF = PythonFunction


def x86_64_compile(no_bench: bool = False):
    def decorator(func):
        setattr(func, "is_emitted", False)
        setattr(func, "is_compiled", False)
        setattr(func, "is_linked", False)
        setattr(func, "asm_faster", False)
        setattr(func, "tested_python", False)
        setattr(func, "tested_asm", False)
        setattr(func, "asm_time", 0)
        setattr(func, "python_time", 0)
        # Parse the function's source code to an AST
        if not func.is_emitted:
            source_code = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source_code)
            # Find the function node in the AST by its name
            function_node = [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == func.__name__
            ][0]
            # print(ast.dump(function_node, indent=4))
            PF(function_node, Stack())()
            func.is_emitted = True

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            if not func.is_compiled:
                PF.jit_prog.compile()
                func.is_compiled = True
            if not func.is_linked:
                PF.jit_prog.link(
                    args={"shared": None},
                    output_extension=(".so" if current_os == "Linux" else ".dll"),
                )
                func.is_linked = True

            # Call the original function
            ret = None
            if no_bench:
                ret = PF.jit_prog.call(func.__name__, *args, **kwargs)
            elif not func.tested_asm:
                asm_time_start = perf_counter_ns()
                ret = PF.jit_prog.call(func.__name__, *args, **kwargs)
                func.asm_time = perf_counter_ns() - asm_time_start
                func.tested_asm = True
                if func.tested_python:
                    func.asm_faster = func.python_time > func.asm_time
            elif not func.tested_python:
                python_time_start = perf_counter_ns()
                ret = func(*args, **kwargs)
                func.python_time = perf_counter_ns() - python_time_start
                func.tested_python = True
                if func.tested_asm:
                    func.asm_faster = func.python_time > func.asm_time
            elif func.asm_faster:
                ret = PF.jit_prog.call(func.__name__, *args, **kwargs)
            else:
                ret = func(*args, **kwargs)
            return ret

        return wrapper

    return decorator


if __name__ == "__main__":
    from time import time

    @x86_64_compile()
    def add_a_b(a: int, b: int) -> int:
        random_float: float = 3.14
        random_float = random_float + 2.5
        counter: int = 0
        while counter < 1_000_000 or b != 2:
            a = a + b
            counter = counter + 1
        return a

    def python_add_a_b(a, b) -> int:
        random_float: float = 3.14
        random_float = random_float + 2.5
        counter: int = 0
        while counter < 1_000_000 or b != 2:
            a = a + b
            counter = counter + 1
        return a

    @x86_64_compile()
    def asm_add_floats(a: float, b: float) -> float:
        random_float: float = 3.14
        random_float = random_float + 2.5
        counter: int = 0
        while counter < 1_000_000 or b != 0.002:
            a = a + b
            counter = counter + 1
        return a

    def python_add_floats(a: float, b: float) -> float:
        random_float: float = 3.14
        random_float = random_float + 2.5
        counter: int = 0
        while counter < 1_000_000 or b != 0.002:
            a = a + b
            counter = counter + 1
        return a

    @x86_64_compile()
    def asm_f_add_test() -> float:
        f: float = 0.002
        f = f + 0.003
        return f + f

    def python_f_add_test() -> float:
        f: float = 0.002
        f = f + 0.003
        return f + f

    @x86_64_compile()
    def asm_f_mul_test() -> float:
        f: float = 0.002
        f *= 0.003
        return f * f

    def python_f_mul_test() -> float:
        f: float = 0.002
        f *= 0.003
        return f * f

    @x86_64_compile()
    def asm_f_div_test() -> float:
        f: float = 0.002
        f = f / 0.003
        return f / 0.15

    def python_f_div_test() -> float:
        f: float = 0.002
        f = f / 0.003
        return f / 0.15

    @x86_64_compile()
    def asm_f_dot(
        x1: float, y1: float, z1: float, x2: float, y2: float, z2: float
    ) -> float:
        f_n1: float = z2 * z1
        f: float = 3.1
        f_n2: float = z2
        return x1 * x2 + y1 * y2 + z1 * z2

    def python_f_dot(
        x1: float, y1: float, z1: float, x2: float, y2: float, z2: float
    ) -> float:
        f_n1: float = z2 * z1
        f: float = 3.1
        f_n2: float = z2
        return x1 * x2 + y1 * y2 + z1 * z2

    @x86_64_compile()
    def asm_i_dot(x1: int, y1: int, z1: int, x2: int, y2: int, z2: int) -> int:
        return x1 * x2 + y1 * y2 + z1 * z2

    def python_i_dot(x1: int, y1: int, z1: int, x2: int, y2: int, z2: int) -> int:
        return x1 * x2 + y1 * y2 + z1 * z2

    @x86_64_compile()
    def asm_aug_assign_f(inp: float) -> float:
        f: float = 200.34 + 22.3
        inp += 1.2 - f
        inp -= 0.1 * f
        inp /= 0.5 + f + f - inp
        inp *= 1.3 / f
        return inp

    def python_aug_assign_f(inp: float) -> float:
        f: float = 200.34 + 22.3
        inp += 1.2 - f
        inp -= 0.1 * f
        inp /= 0.5 + f + f - inp
        inp *= 1.3 / f
        return inp

    @x86_64_compile()
    def asm_aug_assign_i(inp: int) -> int:
        i: int = 2 + 22
        inp += 1 - i
        inp -= 3 * i
        inp //= 4 + i + i - inp + 1
        inp *= 500 // (i + 1)
        return inp

    def python_aug_assign_i(inp: int) -> int:
        i: int = 2 + 22
        inp += 1 - i
        inp -= 3 * i
        inp //= 4 + i + i - inp + 1
        inp *= 500 // (i + 1)
        return inp

    print("\n1_000_000 iteration test (int):\n")

    start = perf_counter_ns()
    totala = 3
    totala = add_a_b(totala, 2)
    print(f"assembly    returns = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

    start = perf_counter_ns()
    totalp = 3
    totalp = python_add_a_b(totalp, 2)
    print(f"python      returns = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

    assert totala == totalp, "1_000_000 iteration test (int) failed"

    print("\n1_000_000 iteration test (float):\n")

    start = perf_counter_ns()
    totala = 0.003
    totala = asm_add_floats(totala, 0.002)
    print(f"assembly    returns = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

    start = perf_counter_ns()
    totalp = 0.003
    totalp = python_add_floats(totalp, 0.002)
    print(f"python      returns = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

    assert totala == totalp, "1_000_000 iteration test (float) failed"

    print("\nf_add_test:\n")

    start = perf_counter_ns()
    totala = asm_f_add_test()
    print(
        f"assembly    f_add_test (0.002 + 0.003) * 2 = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms"
    )

    start = perf_counter_ns()
    totalp = python_f_add_test()
    print(
        f"python      f_add_test (0.002 + 0.003) * 2 = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms"
    )

    assert totala == totalp, "f_add_test failed"

    print("\nf_mul_test:\n")

    start = perf_counter_ns()
    totala = asm_f_mul_test()
    print(
        f"assembly    f_mul_test (0.002 * 0.003)^2 = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms"
    )

    start = perf_counter_ns()
    totalp = python_f_mul_test()
    print(
        f"python      f_mul_test (0.002 * 0.003)^2 = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms"
    )

    assert totala == totalp, "f_mul_test failed"

    print("\nf_div_test:\n")

    start = perf_counter_ns()
    totala = asm_f_div_test()
    print(
        f"assembly    f_div_test 0.002 / 0.003 / 0.15 = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms"
    )

    start = perf_counter_ns()
    totalp = python_f_div_test()
    print(
        f"python      f_div_test 0.002 / 0.003 / 0.15 = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms"
    )

    assert totala == totalp, "f_div_test failed"

    f_dot_args = (*(v1 := (5.3, 2.99, 5.2)), *(v2 := (50.2, 4.3, 1.2)))

    print("\nf dot prod test:\n")

    start = perf_counter_ns()
    totala = asm_f_dot(*f_dot_args)
    print(
        f"assembly    {v1} . {v2} = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms"
    )

    start = perf_counter_ns()
    totalp = python_f_dot(*f_dot_args)
    print(
        f"python      {v1} . {v2} = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms"
    )
    assert totala == totalp, "f dot prod test failed"

    print("\ni dot prod test:\n")

    start = perf_counter_ns()
    totala = asm_i_dot(*(5, 2, 5), *(3, 4, 1))
    print(
        f"assembly    (5,2,5) . (3,4,1) = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms"
    )

    start = perf_counter_ns()
    totalp = python_i_dot(*(5, 2, 5), *(3, 4, 1))
    print(
        f"python      (5,2,5) . (3,4,1) = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms"
    )
    assert totala == totalp, "i dot prod test failed"

    print("\nf dot prod speed tested test:\n")

    # run again for python benchmark
    asm_f_dot(*f_dot_args)

    start = perf_counter_ns()
    totala = asm_f_dot(*f_dot_args)
    print(
        f"assembly    {v1} . {v2} = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms"
    )

    start = perf_counter_ns()
    totalp = python_f_dot(*f_dot_args)
    print(
        f"python      {v1} . {v2} = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms"
    )
    assert totala == totalp, "f dot prod speed tested test failed"

    print("\n1_000_000 iteration speed tested test (float):\n")

    # run again for python benchmark
    asm_add_floats(totala, 0.002)

    start = perf_counter_ns()
    totala = 0.003
    totala = asm_add_floats(totala, 0.002)
    print(f"assembly    returns = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

    start = perf_counter_ns()
    totalp = 0.003
    totalp = python_add_floats(totalp, 0.002)
    print(f"python      returns = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

    assert totala == totalp, "1_000_000 iteration speed tested test (float) failed"

    print("\nAugAssign speed test (float):\n")

    start = perf_counter_ns()
    totala = 0.003
    totala = asm_aug_assign_f(3.14)
    print(f"assembly    returns = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

    start = perf_counter_ns()
    totalp = 0.003
    totalp = python_aug_assign_f(3.14)
    print(f"python      returns = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

    assert totala == totalp, "AugAssign speed test (float) failed"

    print("\nAugAssign speed test (int):\n")

    start = perf_counter_ns()
    totala = 3
    totala = asm_aug_assign_i(900)
    print(f"assembly    returns = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

    start = perf_counter_ns()
    totalp = 3
    totalp = python_aug_assign_i(900)
    print(f"python      returns = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

    assert totala == totalp, "AugAssign speed test (int) failed"
