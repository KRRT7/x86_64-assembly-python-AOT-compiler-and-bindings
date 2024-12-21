from __future__ import annotations

# local imports
from aot.stack import Stack
from aot.utility import (
    FUNCTION_ARGUMENTS,
    FUNCTION_ARGUMENTS_FLOAT,
    float_to_hex,
    load_floats,
    operand_is_float,
    str_to_type,
)
from x86_64_assembly_bindings import (
    Block,
    Function,
    Instruction,
    InstructionData,
    MemorySize,
    OffsetRegister,
    Program,
    Register,
    RegisterData,
    Variable,
    current_os,
)

# std lib imports
import ast
from typing import Any, Callable

# type aliases
Reg = Register
Ins = Instruction
RegD = RegisterData
ProgramLineType = str | Block | Instruction | Callable

# registers
rax: Reg = Reg("rax")
rbp: Reg = Reg("rbp")
rsp: Reg = Reg("rsp")
r11: Reg = Reg("r11")
rdx: Reg = Reg("rdx")


class PythonFunction:
    jit_prog: Program = Program("python_x86_64_jit")
    name: str
    arguments_dict: dict[str, Register | MemorySize]
    arguments: tuple[str, Register | MemorySize]
    lines: list[ProgramLineType]
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
        self.ret_py_type: type | None = None
        self.ret: Reg | None = None
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
                    if a_n < len(FUNCTION_ARGUMENTS):
                        final_arg = FUNCTION_ARGUMENTS[a_n]
                    else:
                        final_arg = OffsetRegister(
                            rbp,
                            16+8*(a_n-len(FUNCTION_ARGUMENTS)) if current_os == "Linux" else 32 + 16+8*(a_n-len(FUNCTION_ARGUMENTS)),
                            meta_tags={"int"},
                            negative=False
                        )
                case "float":
                    if a_n < len(FUNCTION_ARGUMENTS_FLOAT):
                        final_arg = FUNCTION_ARGUMENTS_FLOAT[a_n]
                    else:
                        final_arg = OffsetRegister(
                            rbp,
                            16+8*(a_n-len(FUNCTION_ARGUMENTS_FLOAT)) if current_os == "Linux" else 32 + 16+8*(a_n-len(FUNCTION_ARGUMENTS_FLOAT)),
                            meta_tags={"float"},
                            negative=False
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
        # if self.is_stack_origin:
        #     Ins("mov", rbp, rsp)()
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

    def return_value(self, ret_value: any | None = None) -> list[Instruction]:
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
        # if self.is_stack_origin:
        #     r.append(Ins("mov", rsp, rbp))
        # else:
        r.append(self.stack.pop())
        self.stack.cursor += 1
        r.append(self.gen_ret())
        return r

    def gen_stmt(
        self, body: list[ast.stmt], loop_break_block: Block | None = None
    ) -> tuple[list[Instruction], Register | Block | None]:
        lines: list[Instruction] = []

        sec_ret: Any | None = None
        for stmt in body:
            Register.free_all(lines)
            lines.append("    FREED SCRATCH MEMORY")
            match stmt.__class__.__name__:
                case "Assign":
                    lines.append("STMT::Assign")
                    stmt: ast.Assign
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
        # TODO Fix division so it mimicks python division.
        # py_type specifies what the resultant type should be
        lines: list[ProgramLineType] = []
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
                        isinstance(val1, Register) and val1 in FUNCTION_ARGUMENTS_FLOAT
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
                        isinstance(val1, Register) and val1 in FUNCTION_ARGUMENTS_FLOAT
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
                            and val1 in FUNCTION_ARGUMENTS_FLOAT
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
                floor_round_block = Block(
                    prefix=f".{self.name}__BinOp_FloorDiv__round_toward_neg_inf_"
                )
                lines.append(Ins("test", rdx, rdx))
                lines.append(Ins("jz", floor_round_block))
                lines.append(Ins("test", rax, rax))
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
                        isinstance(val1, Register) and val1 in FUNCTION_ARGUMENTS_FLOAT
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
        lines: list[ProgramLineType] = [
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
        self,
        name: str,
        lines: list[ProgramLineType] | None = None,
        allow_float_load: bool = False,
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
        lines: list[ProgramLineType] = []
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
