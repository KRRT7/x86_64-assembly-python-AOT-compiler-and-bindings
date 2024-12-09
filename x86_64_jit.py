from __future__ import annotations
from x86_64_assembly_bindings import (
    Register, Instruction, MemorySize, Program, Block, Function, OffsetRegister, Variable, RegisterData,
    InstructionData, Memory
)
import ast
import functools
import inspect
import textwrap
import struct

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

function_arguments = [rdi,rsi,rdx,rcx,r8,r9]

float_function_arguments = [Reg(f"xmm{n}") for n in range(0,8)]

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

def operand_is_float(v) -> bool:
    return (isinstance(v, str) and v.startswith("qword 0x")) or (hasattr(v, "name") and v.name.startswith("xmm"))

def float_to_hex(f):
    # Pack the float into 8 bytes (64-bit IEEE 754 double precision)
    packed = struct.pack('>d', f)  # '>d' for big-endian double
    # Unpack the bytes to get the hexadecimal representation
    hex_rep = "qword 0x" + ''.join(f'{b:02x}' for b in packed)
    return hex_rep

def load_floats(f, lines:list, ignore:bool = False):
    if ignore:return f

    if isinstance(f, Register) and not f.name.startswith("xmm"):
        lines.append(Ins("movq", ret_f:=Reg.request_float(), f))
        return ret_f
    elif isinstance(f, str) and f.startswith("qword 0x"):
        lines.append(Ins("mov", reg64:=Reg.request_64(), f))
        lines.append(Ins("movq", ret_f:=Reg.request_float(), reg64))
        return ret_f
    else:return f

class Var:
    def __init__(self, stack_frame:StackFrame, name:str, size:MemorySize, py_type:type = int):
        self.name = name
        self.size = size
        self.type = py_type
        self.stack_frame = stack_frame

    def cast(self, lines:list[Instruction|Block], py_type:type = int) -> Register:
        if py_type == float:
            lines.push(Ins("cvtsi2sd", fpr:=Reg.request_float(), self.get()))
            return fpr
        elif py_type == int:
            lines.push(Ins("cvttsd2si", r:=Reg.request_64(), self.get()))
            return r

    def get(self) -> OffsetRegister:
        return self.stack_frame[self.name]

class StackFrame:
    def __init__(self):
        frame_type = list[Var]
        self.variables:frame_type = []
    
    @property
    def stack_offset(self):
        offset = 0
        for v in self.variables:
            offset += v.size.value//8
        return offset

    def alloca(self, name:str, size:MemorySize = MemorySize.QWORD, py_type:type = int) -> Instruction:
        self.variables.append(Var(self, name, size, py_type))
        return Ins("sub", rsp, size.value//8)

    def pop(self) -> Instruction|None:
        if self.stack_offset != 0:
            return Ins("add", rsp, self.stack_offset)
        return None

    def __contains__(self, key:str) -> bool:
        for v  in self.variables:
            if v.name == key:
                return True
        
        return False

    def __getitem__(self, key:str) -> OffsetRegister:
        offset = 0
        for v in self.variables:
            offset += v.size.value//8
            if v.name == key:
                return OffsetRegister(rbp, offset, True)
        raise KeyError(f"Variable \"{key}\" not found in stack frame.")

    def getvar(self, key:str) -> Var:
        for v in self.variables:
            if v.name == key:
                return v
        raise KeyError(f"Variable \"{key}\" not found in stack frame.")

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

    def alloca(self, name:str, size:MemorySize = MemorySize.QWORD, py_type:type = int) -> Instruction:
        return self.current.alloca(name, size, py_type)

    def push(self):
        self.stack.append(StackFrame())
        self.cursor+=1

    def pop(self) -> Instruction|None:
        r = self.current.pop()
        self.cursor-=1
        return r

    def __contains__(self, key:str) -> bool:
        for frame in self.stack:
            if key in frame:return True
        return False

    def __getitem__(self, key:str) -> OffsetRegister:
        for frame in reversed(self.stack[0:self.cursor+1]):
            
            if key in frame:
                return frame[key]
        raise KeyError(f"Variable \"{key}\" not found in function stack.")

    def getvar(self, key:str) -> Var:
        for frame in reversed(self.stack[0:self.cursor+1]):
            
            if key in frame:
                return frame.getvar(key)
        raise KeyError(f"Variable \"{key}\" not found in function stack.")

class PythonFunction:
    jit_prog:Program = Program("python_x86_64_jit")
    name:str
    arguments_dict:dict[str,Register|MemorySize]
    arguments:(str, Register|MemorySize)
    lines:list[Instruction]
    python_ast:ast.FunctionDef
    ret:Reg|None

    def __init__(self, python_ast:ast.FunctionDef, stack:Stack):
        self.compiled = False
        self.python_ast = python_ast
        self.name = python_ast.name
        self.stack = stack
        self.arguments_dict = {}
        self.arguments = []
        self.arguments_type:dict[str,type] = {}
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
                    raise SyntaxError(f"Unsupported return type \"{python_ast.returns.id}\" for decorated function.")
        self.signed_args:set[int] = set()
        for a_n, argument in enumerate(python_ast.args.args):
            if a_n < len(function_arguments):
                self.arguments_type[argument.arg] = a_type = str_to_type(argument.annotation.id)
                match a_type.__name__:
                    case "int":
                        final_arg = function_arguments[a_n]
                    case "float":
                        final_arg = float_function_arguments[a_n]
                        self.signed_args.add(a_n)
                self.arguments_dict[argument.arg] = final_arg
                self.arguments.append(final_arg)
                
        self.function = Function(self.arguments, return_register=self.ret, label=self.name, return_signed=True, ret_py_type=self.ret_py_type, signed_args=self.signed_args)
        self.gen_ret = lambda:self.function.ret
        self.is_stack_origin = self.stack.get_is_origin()
        self.lines, _ = self.gen_stmt(self.python_ast.body)
        
            

    def __call__(self):
        self.function()
        if self.is_stack_origin:
            Ins("mov", rbp, rsp)()
            #Ins("sub", rsp, 8)()
        for line in self.lines:
            if line:
                if isinstance(line, str):
                    self.jit_prog.comment(line)
                else:
                    line()
        
        if hasattr(line, "name") and line.name != "return":
            if pi:=self.stack.pop():pi()
            self.return_value()[0]()
        
        return self

    def return_value(self, ret_value:any = None) -> list[Instruction]:
        r = []
        match self.ret_py_type.__name__:
            case "int":
                r = [Ins("mov", self.ret, ret_value)] if ret_value and self.ret.name != str(ret_value) else []
            case "float":
                r = []
                if ret_value:
                    f = load_floats(ret_value, r)
                    if self.ret.name != str(f):
                        
                        r.append(Ins("movq", self.ret, f))
            case _:
                r = []
        if self.is_stack_origin:
            r.append(Ins("mov", rsp, rbp))
        else:
            r.append(self.stack.pop())
            self.stack.cursor+=1
        r.append(self.gen_ret())
        return r

        
        
    def gen_stmt(self, body:list[ast.stmt], loop_break_block:Block|None = None) -> tuple[list[Instruction], Register|Block|None]:
        Register.free_all()
        lines:list[Instruction] = []
        sec_ret = None
        for stmt in body:
            match stmt.__class__.__name__:
                case "Assign":
                    lines.append("STMT::Assign")
                    _instrs, value = self.gen_expr(stmt.value)
                    lines.extend(_instrs)

                    for target in stmt.targets:
                        _instrs, key = self.gen_expr(target, py_type=stmt.type_comment)
                        lines.extend(_instrs)
                        
                        if k_is_str:=isinstance(key, str):
                            lines.append(self.stack.alloca(key))

                        dest = self.stack[key] if k_is_str else key
                        
                        if str(dest) != str(value):
                            if type(value) in {Variable, OffsetRegister}:
                                lines.append(Ins("mov", r64:=Reg.request_64(), value))
                                value = r64
                            value = load_floats(value, lines, not dest.name.startswith("xmm"))
                            lines.append(Ins("movq" if operand_is_float(value) else "mov", dest, value))


                case "AnnAssign":
                    lines.append("STMT::AnnAssign")
                    stmt:ast.AnnAssign
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
                    
                    if k_is_str:=isinstance(key, str):
                        lines.append(self.stack.alloca(key,py_type=alloca_type))

                    dest = self.stack[key] if k_is_str else key
                    if str(dest) != str(value):
                        if type(value) in {Variable, OffsetRegister}:
                            lines.append(Ins("mov", r64:=Reg.request_64(), value))
                            value = r64
                        value = load_floats(value, lines, alloca_type.__name__ == "int")
                        lines.append(Ins("movq" if operand_is_float(value) else "mov", dest, value))


                case "Return":
                    lines.append("STMT::Return")
                    stmt:ast.Return
                    
                    _instrs, value = self.gen_expr(stmt.value)
                    lines.extend(_instrs)
                    
                    lines.extend(self.return_value(value))
                    

                case "If":
                    lines.append("STMT::If")
                    false_block = Block()
                    else_ins, false_block_maybe = self.gen_stmt(stmt.orelse, loop_break_block = loop_break_block)
                    if false_block_maybe:
                        false_block = false_block_maybe

                    sc_block = Block()
                    cond_instrs, cond_val = self.gen_expr(stmt.test, block=false_block, sc_block=sc_block)
                    
                    if_bod, _ = self.gen_stmt(stmt.body, loop_break_block = loop_break_block)
                    sec_ret = Block()
                    end_block = Block()
                    lines.extend([
                        sec_ret,
                        *cond_instrs,
                        Ins("test", cond_val, cond_val),
                        Ins("jz", false_block),
                        sc_block,
                        *if_bod,
                        Ins("jmp", end_block),
                        false_block,
                        *else_ins,
                        end_block
                    ])

                case "Break":
                    lines.append("STMT::Break")
                    lines.append(Ins("jmp", loop_break_block))

                case "While":
                    lines.append("STMT::While")
                    stmt:ast.While
                    false_block = Block()
                    else_ins, false_block_maybe = self.gen_stmt(stmt.orelse)
                    if false_block_maybe:
                        false_block = false_block_maybe

                    sc_block = Block()
                    cond_instrs, cond_val = self.gen_expr(stmt.test, block=false_block, sc_block=sc_block)

                    
                    sec_ret = Block()
                    end_block = Block()

                    while_bod, _ = self.gen_stmt(stmt.body, loop_break_block=end_block)
                    lines.extend([
                        sec_ret,
                        *cond_instrs,
                        Ins("test", cond_val, cond_val),
                        Ins("jz", false_block),
                        sc_block,
                        *while_bod,
                        Ins("jmp", sec_ret),
                        false_block,
                        *else_ins,
                        end_block
                    ])

        return lines, sec_ret

    def gen_operator(self, val1:Register|Variable, op:ast.operator, val2:Register|Variable|int, py_type:type = int) -> tuple[list[Instruction|Block], Register]:
        lines = []
        res = None
        is_float = py_type.__name__ == "float" or any(operand_is_float(v) for v in [val1, val2])
        if is_float:
            py_type = float
        req_reg = lambda:(Reg.request_float() if is_float else Reg.request_64())
        

        match op.__class__.__name__:
            case "Add":
                lines.append("Add")
                if not is_float:
                    lines.append(Ins("mov", nval1:=req_reg(), val1))
                    val1 = nval1
                lines.append(Ins(InstructionData.from_py_type("add", py_type), nval1:=load_floats(val1, lines, not is_float), load_floats(val2, lines, not is_float)))
                val1 = nval1
                res = val1
            case "Sub":
                lines.append("Sub")
                if not is_float:
                    lines.append(Ins("mov", nval1:=req_reg(), val1))
                    val1 = nval1
                lines.append(Ins(InstructionData.from_py_type("sub", py_type), nval1:=load_floats(val1, lines, not is_float), load_floats(val2, lines, not is_float)))
                val1 = nval1
                res = val1
            case "Mult":
                if is_float:
                    lines.append("BinOp::Mult(FLOAT)")
                    lines.append(Ins(InstructionData.from_py_type("mul", py_type), nval1:=load_floats(val1, lines, not operand_is_float(val1)), load_floats(val2, lines, not operand_is_float(val2))))
                    val1 = nval1
                    res = val1
                else:
                    lines.append("BinOp::Mult(INTEGER)")
                    if str(val1) != str(rax):
                        lines.append(Ins("mov", rax, val1))
                        val1 = rax
                    lines.append(Ins("imul", val2))

                    lines.append(Ins("mov", res:=Reg.request_64(), val1))
            case "FloorDiv":
                lines.append("BinOp::FloorDiv")
                if str(val1) != str(rax):
                    lines.append(Ins("mov", rax, val1))
                    val1 = rax
                lines.append(Ins("cdq"))
                if isinstance(val2, int):
                    lines.append(Ins("mov", val2:=Reg.request_64(), val2))
                lines.append(Ins("idiv", val2))
                lines.append(Ins("mov", res:=Reg.request_64(), val1))
            case "Div":
                lines.append("BinOp::Div")
                if not is_float:
                    lines.append(Ins("mov", nval1:=req_reg(), val1))
                    val1 = nval1
                lines.append(Ins(InstructionData.from_py_type("div", float), nval1:=load_floats(val1, lines, not is_float), load_floats(val2, lines, not is_float)))
                val1 = nval1
                res = val1

            case "Mod":
                lines.append("BinOp::Mod")
                if str(val1) != str(rax):
                    lines.append(Ins("mov", rax, val1))
                    val1 = rax
                lines.append(Ins("cdq"))
                if isinstance(val2, int):
                    lines.append(Ins("mov", r11, val2))
                    val2 = r11
                lines.append(Ins("idiv", val2))
                lines.append(Ins("mov", res:=Reg.request_64(), rdx))

        
        return lines, res


    def gen_cmp_operator(self, val1:Register|Variable, op:ast.cmpop, val2:Register|Variable|int, py_type:type|str = int) -> tuple[list[Instruction|Block], Register]:
        res = Reg.request_8()
        is_float = py_type.__name__ == "float" or any(operand_is_float(v) for v in [val1, val2])
        lines = [
            Ins("xor", res.cast_to(MemorySize.QWORD), res.cast_to(MemorySize.QWORD))
        ]
        lines.append(Ins(InstructionData.from_py_type("cmp", float if is_float else int), load_floats(val1,lines, not is_float), load_floats(val2,lines, not is_float)))
        
        match op.__class__.__name__:
            case "Eq":
                lines.append("CmpOp::Eq|\"==\"")
                lines.append(Ins("sete", res))
            case "NotEq":
                lines.append("CmpOp::NotEq|\"!=\"")
                lines.append(Ins("setne", res))
            case "Lt":
                lines.append("CmpOp::Lt|\"<\"")
                lines.append(Ins("setl", res))
            case "LtE":
                lines.append("CmpOp::LtE|\"<=\"")
                lines.append(Ins("setle", res))
            case "Gt":
                lines.append("CmpOp::Gt|\">\"")
                lines.append(Ins("setg", res))
            case "GtE":
                lines.append("CmpOp::GtE|\">=\"")
                lines.append(Ins("setge", res))

            case "In":
                pass

            case "NotIn":
                pass

        
        
        return lines, res.cast_to(MemorySize.QWORD)

    def get_var(self, name:str, lines:list|None = None, allow_float_load:bool = False) -> Register|str:
        try:
            if name in self.arguments_dict:
                return self.arguments_dict[name]
            else:
                v = self.stack.getvar(name)
                r = None
                if v.type.__name__ == "float" and lines is not None and allow_float_load:
                    r = load_floats(v.get(), lines)
                return r if r else v.get()
        except KeyError:
            return name
        
    def gen_expr(self, expr:ast.expr, py_type:type = int, block:Block|None = None, sc_block:Block|None = None) -> tuple[list[Instruction], any]:
        lines = []
        sec_ret = None
        match expr.__class__.__name__:
            case "Constant":
                expr:ast.Constant
                if isinstance(expr.value, int):
                    sec_ret = int(expr.value)
                elif isinstance(expr.value, float):
                    sec_ret = float_to_hex(expr.value)
            case "Name":
                expr:ast.Name
                lines.append(f"label::\"{expr.id}\"")
                sec_ret = self.get_var(expr.id, lines, not isinstance(expr.ctx, ast.Store))
            case "BinOp":
                expr:ast.BinOp
                _instrs, val1 = self.gen_expr(expr.left)
                lines.extend(_instrs)
                _instrs, val2 = self.gen_expr(expr.right)
                lines.extend(_instrs)
                _pyt = py_type
                if any((isinstance(v, str) and v.startswith("0x")) for v in [val1,val2])\
                or any((isinstance(v, Register) and v.name.startswith("xmm")) for v in [val1,val2]):
                    _pyt = float
                _instrs, res = self.gen_operator(val1, expr.op, val2, _pyt)
                lines.extend(_instrs)
                sec_ret = res

            case "BoolOp":
                expr:ast.BoolOp
                bres = Register.request_8()
                lines.append(Ins("xor", bres.cast_to(MemorySize.QWORD), bres.cast_to(MemorySize.QWORD)))
                match expr.op.__class__.__name__:
                    case "And":
                        lines.append("BoolOp::AND")
                        local_sc_b = Block()
                        first_operand = None
                        for boperand in expr.values:
                            _instrs, operand = self.gen_expr(boperand)
                            lines.extend(_instrs)
                            
                            if first_operand:
                                lines.append(Ins(InstructionData.from_py_type("and", py_type), first_operand, operand))
                            
                            if block:
                                lines.append(Ins("jz", block))
                            else:
                                lines.append(Ins("jz", local_sc_b))# Jump to the short circuit assign
                            
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
                                lines.append(Ins(InstructionData.from_py_type("or", py_type), first_operand, operand))

                            if block:
                                lines.append(Ins("jnz", sc_block))
                            else:
                                lines.append(Ins("jnz", local_sc_b))# Jump to the short circuit assign
                            
                            if not first_operand:
                                first_operand = operand
                        if not block:
                            lines.append(local_sc_b)
                        
                        lines.append(Ins("setnz", bres))

                sec_ret = bres.cast_to(MemorySize.QWORD)

            case "Compare":
                expr:ast.Compare
                _instrs, val1 = self.gen_expr(expr.left, block = block, sc_block = sc_block)
                lines.extend(_instrs)
                
                for op_i, op in enumerate(expr.ops):
                    
                    _instrs, val2 = self.gen_expr(expr.comparators[op_i], block = block, sc_block = sc_block)
                    lines.extend(_instrs)
                    _instrs, val1 = self.gen_cmp_operator(val1, op, val2)
                    lines.extend(_instrs)


                sec_ret = val1

        return lines, sec_ret

PF = PythonFunction

def x86_compile():
    def decorator(func):
        setattr(func, "is_emitted", False)
        setattr(func, "is_compiled", False)
        setattr(func, "is_linked", False)
        # Parse the function's source code to an AST
        if not func.is_emitted:
            source_code = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source_code)
            # Find the function node in the AST by its name
            function_node = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == func.__name__][0]
            #print(ast.dump(function_node, indent=4))
            PF(function_node, Stack())()
            func.is_emitted = True

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not func.is_compiled:
                PF.jit_prog.compile()
                func.is_compiled = True
            if not func.is_linked:
                PF.jit_prog.link(args={"shared":None}, output_extension=".so")
                func.is_linked = True
                    
            # Call the original function
            return PF.jit_prog.call(func.__name__, *args)
    
        return wrapper
    return decorator

if __name__ == "__main__":
    from time import time

    @x86_compile()
    def add_a_b(a:int,b:int) -> int:
        random_float:float = 3.14
        random_float = random_float + 2.5
        counter:int = 0
        while counter < 1_000_000 or b != 2:
            a = a + b
            counter = counter + 1
        return a

    def python_add_a_b(a,b) -> int:
        random_float:float = 3.14
        random_float = random_float + 2.5
        counter:int = 0
        while counter < 1_000_000 or b != 2:
            a = a + b
            counter = counter + 1
        return a

    @x86_compile()
    def asm_add_floats(a:float,b:float) -> float:
        random_float:float = 3.14
        random_float = random_float + 2.5
        counter:int = 0
        while counter < 1_000_000 or b != 0.002:
            a = a + b
            counter = counter + 1
        return a

    def python_add_floats(a:float, b:float) -> float:
        random_float:float = 3.14
        random_float = random_float + 2.5
        counter:int = 0
        while counter < 1_000_000 or b != 0.002:
            a = a + b
            counter = counter + 1
        return a

    @x86_compile()
    def asm_f_add_test() -> float:
        f:float = 0.002
        f = f + 0.003
        return f + f

    def python_f_add_test() -> float:
        f:float = 0.002
        f = f + 0.003
        return f + f

    @x86_compile()
    def asm_f_mul_test() -> float:
        f:float = 0.002
        f = f * 0.003
        return f * f

    def python_f_mul_test() -> float:
        f:float = 0.002
        f = f * 0.003
        return f * f

    @x86_compile()
    def asm_f_div_test() -> float:
        f:float = 0.002
        f = f / 0.003
        return f / 0.15

    def python_f_div_test() -> float:
        f:float = 0.002
        f = f / 0.003
        return f / 0.15

    @x86_compile()
    def asm_f_dot(x1:float,y1:float,z1:float, x2:float,y2:float,z2:float) -> float:
        return x1*x2+y1*y2+z1*z2

    def python_f_dot(x1:float,y1:float,z1:float, x2:float,y2:float,z2:float) -> float:
        return x1*x2+y1*y2+z1*z2



    print("\n1_000_000 iteration test (int):")

    
    start = time()
    totala = 3
    totala = add_a_b(totala, 2)
    print(f"assembly    returns = {totala}    {(time()-start)*1000:.4f}ms")

    start = time()
    totalp = 3
    totalp = python_add_a_b(totalp, 2)
    print(f"python      returns = {totalp}    {(time()-start)*1000:.4f}ms")

    assert totala == totalp, "1_000_000 iteration test (int) failed"

    print("\n1_000_000 iteration test (float):")

    start = time()
    totala = 0.003
    totala = asm_add_floats(totala, 0.002)
    print(f"assembly    returns = {totala}    {(time()-start)*1000:.4f}ms")

    start = time()
    totalp = 0.003
    totalp = python_add_floats(totalp, 0.002)
    print(f"python      returns = {totalp}    {(time()-start)*1000:.4f}ms")
    
    assert totala == totalp, "1_000_000 iteration test (float) failed"


    print("\nf_add_test:")

    start = time()
    totala = asm_f_add_test()
    print(f"assembly    f_add_test (0.002 + 0.003) * 2 = {totala}    {(time()-start)*1000:.4f}ms")

    start = time()
    totalp = python_f_add_test()
    print(f"python      f_add_test (0.002 + 0.003) * 2 = {totalp}    {(time()-start)*1000:.4f}ms")

    assert totala == totalp, "f_add_test failed"

    print("\nf_mul_test:")

    start = time()
    totala = asm_f_mul_test()
    print(f"assembly    f_mul_test (0.002 * 0.003)^2 = {totala}    {(time()-start)*1000:.4f}ms")

    start = time()
    totalp = python_f_mul_test()
    print(f"python      f_mul_test (0.002 * 0.003)^2 = {totalp}    {(time()-start)*1000:.4f}ms")

    assert totala == totalp, "f_mul_test failed"

    print("\nf_div_test:")

    start = time()
    totala = asm_f_div_test()
    print(f"assembly    f_div_test 0.002 / 0.003 / 0.15 = {totala}    {(time()-start)*1000:.4f}ms")

    start = time()
    totalp = python_f_div_test()
    print(f"python      f_div_test 0.002 / 0.003 / 0.15 = {totalp}    {(time()-start)*1000:.4f}ms")

    assert totala == totalp, "f_div_test failed"

    print("\ndot prod test:")

    start = time()
    totala = asm_f_dot(*(5.3,2.5,5.2), *(3.2,4.3,1.2))
    print(f"assembly    (5.3,2.5,5.2) . (3.2,4.3,1.2) = {totala}    {(time()-start)*1000:.4f}ms")

    start = time()
    totalp = python_f_dot(*(5.3,2.5,5.2), *(3.2,4.3,1.2))
    print(f"python      (5.3,2.5,5.2) . (3.2,4.3,1.2) = {totalp}    {(time()-start)*1000:.4f}ms")
