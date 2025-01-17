from collections import OrderedDict
from aot.binop import add_float_float, add_int_int, div_float_float, floordiv_float_float, floordiv_int_int, implicit_cast, mod_float_float, mod_int_int, mul_float_float, mul_int_int, sub_float_float, sub_int_int
from aot.compare import compare_operator_from_type, implicit_cast_cmp
from aot.type_imports import *
from aot.stack import Stack
from aot.utils import CAST, FUNCTION_ARGUMENTS, FUNCTION_ARGUMENTS_BOOL, FUNCTION_ARGUMENTS_FLOAT, get_type_name, load, memory_size_from_type, reg_request_bool, reg_request_float, reg_request_from_type, type_from_annotation, type_from_object, type_from_str
from aot.variable import Value, Variable
from x86_64_assembly_bindings import (
    Program, Function, PtrType
)
import ast

def mangle_function_name(name:str, types:list[type]):
    if types:
        return "TEMPLATED_" + "_".join([(v.__name__ if isinstance(v, type) else str(v)) for v in types]) + "__" + name
    else:
        return name

class PythonFunction:
    current_jit_program: Program | None = None

    def __init__(self, python_function_ast: ast.FunctionDef, stack: Stack, templates:OrderedDict[Template, type], jit_program: Program|None = None):
        self.python_function_ast: ast.FunctionDef = python_function_ast
        self.stack: Stack = stack
        
        # Function data
        self.templates: OrderedDict[str, type] = OrderedDict({t.name:v for t, v in templates.items()})
        self.name: str = mangle_function_name(self.python_function_ast.name, self.templates.values())
        self.arguments: OrderedDict[str, Variable] = OrderedDict({})
        self.return_variable: Variable | None = None

        self.jit_program: Program = jit_program if jit_program else Program(f".py_x86_64_functions/python_x86_64_aot_{self.name}")
        self.current_jit_program = self.jit_program

        # arguments variables
        self.__init_get_args()

        # Get return variable
        if self.python_function_ast.returns:
            return_python_type = type_from_annotation(self.python_function_ast.returns.id, self.templates)
            if return_python_type is int:
                self.return_variable = Variable("RETURN", return_python_type, Reg("rax", {return_python_type, "variable"}))
            elif return_python_type is float:
                self.return_variable = Variable("RETURN", return_python_type, Reg("xmm0", {return_python_type, "variable"}))
            elif return_python_type is bool:
                self.return_variable = Variable("RETURN", return_python_type, Reg("al", {return_python_type, "variable"}))
            else:
                raise SyntaxError(
                    f'Unsupported return type "{self.python_function_ast.returns.id}"'
                    f' for compiled function {self.name}.'
                )
        else:        
            self.return_variable = Variable("RETURN", None, Reg("rax"))
                
        # Create the assembly function object
        self.function:Function = Function(
            arguments         = [v.value for v in self.arguments.values()],
            return_register   = self.return_variable.value,
            label             = self.name,
            return_signed     = True,
            ret_py_type       = self.return_variable.python_type,
            signed_args       = {i for i, v in enumerate(self.arguments.values())},
            arguments_py_type = [
                (PtrType(v.python_type.python_type) if isinstance(v.python_type, Array) else v.python_type)
                for v in self.arguments.values()
            ]
        )

        self.lines: LinesType = []
        for stmt in self.python_function_ast.body:
            self.lines.extend(self.gen_stmt(stmt))
                
    def __init_get_args(self):
        int_args = [*reversed(FUNCTION_ARGUMENTS)]
        float_args = [*reversed(FUNCTION_ARGUMENTS_FLOAT)]
        bool_args = [*reversed(FUNCTION_ARGUMENTS_BOOL)]
        for a_n, argument in enumerate(self.python_function_ast.args.args):
            python_type = type_from_annotation(argument.annotation, self.templates)
            variable_store = None
            size = MemorySize.QWORD
            if python_type is int:
                if current_os == "Linux" and len(int_args):
                    variable_store = int_args.pop()
                    bool_args.pop()
                elif a_n < len(FUNCTION_ARGUMENTS):
                    variable_store = FUNCTION_ARGUMENTS[a_n]
                else:
                    variable_store = OffsetRegister(
                        Reg("rbp",{int}),
                        16 + 8 * (a_n - len(FUNCTION_ARGUMENTS))
                        if current_os == "Linux"
                        else 32 + 16 + 8 * (a_n - len(FUNCTION_ARGUMENTS)),
                        meta_tags={int, "variable"},
                        negative=False,
                    )
            elif type(python_type) is Array:
                if current_os == "Linux" and len(int_args):
                    variable_store = int_args.pop()
                    bool_args.pop()
                elif a_n < len(FUNCTION_ARGUMENTS):
                    variable_store = FUNCTION_ARGUMENTS[a_n]
                else:
                    variable_store = OffsetRegister(
                        Reg("rbp",{python_type.python_type}),
                        16 + 8 * (a_n - len(FUNCTION_ARGUMENTS))
                        if current_os == "Linux"
                        else 32 + 16 + 8 * (a_n - len(FUNCTION_ARGUMENTS)),
                        meta_tags={python_type.python_type, "variable", "array", "pointer"},
                        negative=False,
                    )
            elif python_type is float:
                if current_os == "Linux" and len(float_args):
                    variable_store = float_args.pop()
                elif a_n < len(FUNCTION_ARGUMENTS_FLOAT):
                    variable_store = FUNCTION_ARGUMENTS_FLOAT[a_n]
                else:
                    variable_store = OffsetRegister(
                        Reg("rbp",{float}),
                        16 + 8 * (a_n - len(FUNCTION_ARGUMENTS_FLOAT))
                        if current_os == "Linux"
                        else 32 + 16 + 8 * (a_n - len(FUNCTION_ARGUMENTS_FLOAT)),
                        meta_tags={float, "variable"},
                        negative=False,
                    )
            elif python_type is bool:
                size = MemorySize.BYTE
                if current_os == "Linux" and len(bool_args):
                    variable_store = bool_args.pop()
                    int_args.pop()
                elif a_n < len(FUNCTION_ARGUMENTS_BOOL):
                    variable_store = FUNCTION_ARGUMENTS_BOOL[a_n]
                else:
                    variable_store = OffsetRegister(
                        Reg("rbp",{bool}),
                        16 + 8 * (a_n - len(FUNCTION_ARGUMENTS_BOOL)) - 7
                        if current_os == "Linux"
                        else 32 + 16 + 8 * (a_n - len(FUNCTION_ARGUMENTS_BOOL)) - 7,
                        meta_tags={bool, "variable"},
                        negative=False,
                        override_size=MemorySize.BYTE,
                    )
            elif python_type is None:
                raise TypeError(f"Function argument ({argument.arg}) type for compiled function cannot be None.")

            self.arguments[argument.arg] = Variable(argument.arg, python_type, variable_store, size)

    def get_var(self, key:str) -> Variable:
        if key in self.stack:
            return self.stack[key]
        elif key in self.arguments:
            return self.arguments[key]
        else:
            raise KeyError(f"Variable {key} not found.")
        
    def var_exists(self, key:str) -> bool:
        return key in self.stack or key in self.arguments
        
    def __call__(self):
        self.function()
        finished_with_return = False
        if self.stack.current.frame_size:
            # Allocate the stack
            Ins("sub", Reg("rsp"), self.stack.current.frame_size)()
        for line in self.lines:
            if line:
                if isinstance(line, Comment):
                    self.jit_program.comment(line)
                else:
                    finished_with_return = hasattr(line, "is_return")
                    line()
        
        if not finished_with_return:
            # return a default value if it fails to return
            try:
                default_return_value = {
                    int:IntLiteral(0),
                    bool:IntLiteral(0),
                    float:FloatLiteral(0.0),
                    None:None
                }[self.return_variable.python_type]
            except KeyError:
                raise TypeError("Invalid return type.")

            for line in self.return_value(default_return_value):
                if isinstance(line, str):
                    self.jit_program.comment(line)
                else:
                    line()
        

    def return_value(self, value:Variable|ScalarType|None = None) -> LinesType:
        
        lines: LinesType = []
        if self.return_variable.python_type: # in case it is None
            match self.return_variable.python_type.__name__:
                case "int"|"bool":
                    lines, loaded_value = load(value, self, no_mov=True)
                    lines.append(Ins("mov", self.return_variable.value, loaded_value))
                case "float":
                    lines, loaded_value = load(value, self, no_mov=True)
                    lines.append(Ins("movsd", self.return_variable.value, loaded_value))

        stack_frame_free_lines = self.stack.free()
        if stack_frame_free_lines:
            lines.extend(stack_frame_free_lines)
        function_ret = lambda *args:self.function.ret(*args)
        setattr(function_ret, "is_return", True)
        lines.append(function_ret)

        return lines
    
    def gen_expr(self, expr: ast.expr,
        true_short_circuit_block: Block | None = None,
        false_short_circuit_block: Block | None = None
    ) -> tuple[LinesType, Variable|Value|ScalarType|str|Array]:
        lines: LinesType = []
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, int):
                return lines, IntLiteral(int(expr.value))
            elif isinstance(expr.value, float):
                return lines, FloatLiteral(float(expr.value))
            elif isinstance(expr.value, bool):
                return lines, BoolLiteral(bool(expr.value))
            else:
                raise NotImplementedError(f"Constant Type {type(expr.value).__name__} has not been implemented yet.")
        elif isinstance(expr, ast.Name):
            lines.append(f'label::"{expr.id}"')
            if expr.id in self.templates:
                lines.append(f' ^ Template')
                return lines, type_from_str(expr.id, self.templates)
            elif self.var_exists(expr.id):
                return lines, self.get_var(expr.id)
            else:                
                return lines, expr.id
        elif isinstance(expr, ast.Subscript):
            instrs, array_memory = self.gen_expr(expr.value)
            lines.extend(instrs)

            instrs, index_memory = self.gen_expr(expr.slice)
            lines.extend(instrs)

            if not isinstance(array_memory.python_type, Array):
                raise TypeError(f"Subscript is only supported for Array type, not {array_memory.python_type.__name__}")
            
            instrs, loaded_index_memory = load(index_memory, self)
            lines.extend(instrs)

            result_register = reg_request_from_type(array_memory.python_type.python_type, lines)

            if isinstance(array_memory.python_type.python_type, Array):
                lines.append(Ins("mov", result_register, array_memory[f"{{}} + ({loaded_index_memory}*8)"]))
            else:
                lines.append(Ins("mov", result_register, array_memory[f"{{}} + ({loaded_index_memory}*8)"]))

            loaded_index_memory.free(lines)

            return lines, Value(array_memory.python_type.python_type, result_register, array_memory.python_type.type_size)
        elif isinstance(expr, ast.List):
            list_length = len(expr.elts)
            if list_length == 0:
                raise SyntaxError("Statically sized array literal must have at least one value."
                    "\n  This may change in the future when variable length arrays are added.")
            
            # Calculate first elt to get the type of the list
            instrs, value = self.gen_expr(expr.elts[0])
            lines.extend(instrs)

            python_type = type_from_object(value)

            size = memory_size_from_type(python_type)

            return_value = self.stack.allocate_value(Array(python_type, list_length), size)

            if isinstance(value, Value) and isinstance(value.python_type, Array):
                instrs, value = load(value, self)
                lines.extend(instrs)

                lines.append(Ins("mov", return_value.value, value))
                value.free(lines)
            else:
                lines.append(Ins("mov", return_value.value, value))

            for index, elt in enumerate(expr.elts[1::], 1):
                instrs, value = self.gen_expr(elt)
                lines.extend(instrs)
                if python_type != (_err_type:=type_from_object(value)):
                    raise TypeError(f"Array must be homogenious, found both {get_type_name(python_type)} and {get_type_name(_err_type)}")
                
                if isinstance(value, Value) and isinstance(value.python_type, Array):
                    instrs, value = load(value, self)
                    lines.extend(instrs)

                    lines.append(Ins("mov", return_value[f"{{}} + {index * (size//8)}"], value))
                    value.free(lines)
                else:
                    lines.append(Ins("mov", return_value[index * (size//8)], value))

            return lines, return_value
        elif isinstance(expr, ast.BinOp):
            instrs, evaluated_left = self.gen_expr(expr.left)
            lines.extend(instrs)
            
            instrs, evaluated_right = self.gen_expr(expr.right)
            lines.extend(instrs)

            instrs, evaluated_return = self.gen_binop(evaluated_left, expr.op, evaluated_right)
            lines.extend(instrs)

            return lines, evaluated_return
        elif isinstance(expr, ast.BoolOp):
            return self.gen_boolop(expr.op, expr.values, true_short_circuit_block, false_short_circuit_block)
        elif isinstance(expr, ast.Compare):
            return self.gen_compare(expr.left, expr.ops, expr.comparators, false_short_circuit_block)
        else:
            raise NotImplementedError(f"The ast.expr token {expr.__class__.__name__} is not implemented yet.")
        
    def gen_compare(self, left:ast.expr, operators:list[ast.cmpop], comparators:list[ast.expr],
    false_short_circuit_block:Block|None = None    
    ) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = []
        values:list[tuple[LinesType, ScalarType | Variable]] = []

        instrs, left = self.gen_expr(left)
        lines.extend(instrs)

        for value_expr in comparators:
            instrs, value = self.gen_expr(value_expr)
            values.append((instrs, value))


        aggregate_value = reg_request_bool(lines=lines)

        short_circuit_block = Block(prefix=".cmp_op_on_False_shortcircuit")
        
        for n, ((right_lines, right), op) in enumerate(zip(values, operators)):

            lines.append(f"COMPARE::{type(op).__name__}")

            left_type, right_type, instrs, left, right= implicit_cast_cmp(self, op, left, right)
            type_pair = left_type, right_type
            lines.extend(instrs)

            lines.extend(right_lines)

            local_result: VariableValueType | ScalarType | Variable | None = None
            if isinstance(op, ast.Eq):
                instrs, local_result = compare_operator_from_type(self, type_pair, "sete", left, right)
                lines.extend(instrs)
            elif isinstance(op, ast.NotEq):
                instrs, local_result = compare_operator_from_type(self, type_pair, "setne", left, right)
                lines.extend(instrs)
            elif isinstance(op, ast.Lt):
                instrs, local_result = compare_operator_from_type(self, type_pair, "setl", left, right)
                lines.extend(instrs)
            elif isinstance(op, ast.LtE):
                instrs, local_result = compare_operator_from_type(self, type_pair, "setle", left, right)
                lines.extend(instrs)
            elif isinstance(op, ast.Gt):
                instrs, local_result = compare_operator_from_type(self, type_pair, "setg", left, right)
                lines.extend(instrs)
            elif isinstance(op, ast.GtE):
                instrs, local_result = compare_operator_from_type(self, type_pair, "setge", left, right)
                lines.extend(instrs)
            elif isinstance(op, ast.Is):
                instrs, local_result = compare_operator_from_type(self, type_pair, "sete", left, right)
                lines.extend(instrs)
            elif isinstance(op, ast.IsNot):
                instrs, local_result = compare_operator_from_type(self, type_pair, "setne", left, right)
                lines.extend(instrs)
            elif isinstance(op, ast.In):
                instrs, local_result = compare_operator_from_type(self, type_pair, "sete", left, right)
                lines.extend(instrs)
            elif isinstance(op, ast.NotIn):
                instrs, local_result = compare_operator_from_type(self, type_pair, "setne", left, right)
                lines.extend(instrs)
            else:
                raise NotImplementedError(f"The comparison operator token {type(op).__name__} is not implemented yet")

            if not local_result:
                raise SyntaxError("Failed to evaluate the local_result.")
            if n > 0:
                lines.append(Ins("and", aggregate_value, local_result))
            else:
                lines.append(Ins("mov", aggregate_value, local_result))
                if len(values) > 1:
                    lines.append(Ins("jz", short_circuit_block
                        if not false_short_circuit_block
                        else false_short_circuit_block
                    ))

            left = right

            lines.append(f"FREED ({local_result})")
            local_result.free(lines)

        lines.append(short_circuit_block)

        return lines, aggregate_value
        

    def gen_boolop(self, operator:ast.operator, value_exprs:list[ast.expr],
        true_short_circuit_block:Block|None = None,
        false_short_circuit_block:Block|None = None
    ) -> tuple[LinesType, VariableValueType|ScalarType]:
        # >> TODO maybe move this function to a separate file ? << #
        
        lines: LinesType = []
        values:list[tuple[LinesType, ScalarType | Variable]] = []
        for value_expr in value_exprs:
            value_lines: LinesType = []

            instrs, value = self.gen_expr(value_expr)
            value_lines.extend(instrs)
            
            instrs, value = CAST.bool(value, self)
            value_lines.extend(instrs)

            values.append((value_lines, value))

        value_0_lines, value_0 = values[0]

        lines.extend(value_0_lines)

        instrs, loaded_value = load(value_0, self, no_mov=True)
        lines.extend(instrs)

        aggregate_value = reg_request_bool(lines=lines)

        lines.append(Ins("mov", aggregate_value, loaded_value))

        # Ensure that the aggregate value is populating the zero flag
        lines.append(Ins("test", aggregate_value, aggregate_value))
        
        short_circuit_block = Block(prefix=".boolop_short_circuit")
       
        for value_lines, value in values[1::]:
            lines.append(f"BOOLOP::{type(operator).__name__}")
            if isinstance(operator, ast.Or):
                # Short circuit to true block if true
                lines.append(Ins("jnz", short_circuit_block if short_circuit_block else true_short_circuit_block))
                lines.extend(value_lines)
                instrs, loaded_value = load(value, self, no_mov=True)
                lines.extend(instrs)
                lines.append(Ins("or", aggregate_value, loaded_value))
            elif isinstance(operator, ast.And):
                # Short circuit to false block if false
                lines.append(Ins("jz", short_circuit_block if short_circuit_block else false_short_circuit_block))
                instrs, loaded_value = load(value, self, no_mov=True)
                lines.extend(instrs)
                lines.append(Ins("and", aggregate_value, loaded_value))
            else:
                raise NotImplementedError(f"Operator Token {operator.__class__.__name__} is not implemented yet.")
        
        lines.append(short_circuit_block)

        return lines, aggregate_value

    def gen_binop(self, left:Variable|Value|VariableValueType|ScalarType, operator:ast.operator, right:Variable|Value|VariableValueType|ScalarType) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = []

        left_value_type, right_value_type, instrs, left_value, right_value = implicit_cast(self, operator, left, right)
        lines.extend(instrs)

        if isinstance(operator, ast.Add):
            # both are int
            if left_value_type is int and right_value_type is int:
                instrs, result_memory = add_int_int(self, left_value, right_value)
                lines.extend(instrs)
                return lines, result_memory
            # both are float
            elif left_value_type is float and right_value_type is float:
                instrs, result_memory = add_float_float(self, left_value, right_value)
                lines.extend(instrs)
                return lines, result_memory

        elif isinstance(operator, ast.Sub):
            # both are int
            if left_value_type is int and right_value_type is int:
                instrs, result_memory = sub_int_int(self, left_value, right_value)
                lines.extend(instrs)
                return lines, result_memory
            # both are float
            elif left_value_type is float and right_value_type is float:
                instrs, result_memory = sub_float_float(self, left_value, right_value)
                lines.extend(instrs)
                return lines, result_memory
            
        elif isinstance(operator, ast.Mult):
            # both are int
            if left_value_type is int and right_value_type is int:
                instrs, result_memory = mul_int_int(self, left_value, right_value)
                lines.extend(instrs)
                return lines, result_memory
            # both are float
            elif left_value_type is float and right_value_type is float:
                instrs, result_memory = mul_float_float(self, left_value, right_value)
                lines.extend(instrs)
                return lines, result_memory
            
        elif isinstance(operator, ast.FloorDiv):

            # both are int
            if left_value_type is int and right_value_type is int:
                instrs, result_memory = floordiv_int_int(self, left_value, right_value)
                lines.extend(instrs)
                return lines, result_memory
            # both are float
            if left_value_type is float and right_value_type is float:
                instrs, result_memory = floordiv_float_float(self, left_value, right_value)
                lines.extend(instrs)
                return lines, result_memory
            
        elif isinstance(operator, ast.Mod):

            # both are int
            if left_value_type is int and right_value_type is int:
                instrs, result_memory = mod_int_int(self, left_value, right_value)
                lines.extend(instrs)
                return lines, result_memory
            # both are float
            elif left_value_type is float and right_value_type is float:
                instrs, result_memory = mod_float_float(self, left_value, right_value)
                lines.extend(instrs)
                return lines, result_memory
            
        elif isinstance(operator, ast.Div):
            # both are float
            if left_value_type is float and right_value_type is float:
                instrs, result_memory = div_float_float(self, left_value, right_value)
                lines.extend(instrs)
                return lines, result_memory
            elif left_value_type is int and right_value_type is int:
                # cast both ints to floats

                instrs, left_value = CAST.float(left_value, self)
                lines.extend(instrs)
                
                instrs, right_value = CAST.float(right_value, self)
                lines.extend(instrs)

                instrs, result_memory = div_float_float(self, left_value, right_value)
                lines.extend(instrs)
                return lines, result_memory
            else:
                raise NotImplementedError(f"The ast.BinOp token {operator} is not implemented yet for {left_value_type.__name__} and {right_value_type.__name__} operations.")
        else:
            raise NotImplementedError(f"The ast.BinOp token {operator} is not implemented yet.")

        
    
    def gen_stmt(self, stmt: ast.stmt, parent_passed_block:Block|None = None) -> LinesType:
        lines: LinesType = []
        Register.free_all(lines)
        lines.append("    FREED SCRATCH MEMORY")

        if isinstance(stmt, ast.Assign):
            lines.append("STMT::Assign")
            instrs, value = self.gen_expr(stmt.value)
            lines.extend(instrs)

            for target in stmt.targets:
                instrs, variable = self.gen_expr(target)
                lines.extend(instrs)

                instrs = variable.set(value)
                lines.extend(instrs)


        elif isinstance(stmt, ast.AnnAssign):
            lines.append(f"STMT::AnnAssign({type_from_annotation(stmt.annotation, self.templates)})")
            instrs, value = self.gen_expr(stmt.value)
            lines.extend(instrs)

            target = stmt.target
            variable_type = type_from_annotation(stmt.annotation, self.templates)

            
            instrs, name = self.gen_expr(target)
            lines.extend(instrs)

            if type(value) is Value:
                self.stack.allocate_from_value(name, value)
            else:
                self.stack.allocate(name, variable_type)
                instrs = self.get_var(name).set(value, self)
                lines.extend(instrs)

        elif isinstance(stmt, ast.AugAssign):
            lines.append(f"STMT::AugAssign({stmt.op.__class__.__name__})")

            instrs, evaluated_target = self.gen_expr(stmt.target)
            lines.extend(instrs)

            instrs, evaluated_value = self.gen_expr(stmt.value)
            lines.extend(instrs)

            instrs, value = self.gen_binop(evaluated_target, stmt.op, evaluated_value)
            lines.extend(instrs)

            instrs, variable = self.gen_expr(stmt.target)
            lines.extend(instrs)

            instrs = variable.set(value, self)
            lines.extend(instrs)

        elif isinstance(stmt, ast.While):
            lines.append("STMT::While")

            while_start = Block(prefix=".while_start")
            while_else = Block(prefix=".while_else")
            while_end = Block(prefix=".while_end")

            test_instrs, test_result = self.gen_expr(stmt.test,
                false_short_circuit_block=while_else
            )

            body: LinesType = []

            for body_stmt in stmt.body:
                body.extend(self.gen_stmt(body_stmt, while_end))

            elses: LinesType = []

            for else_stmt in stmt.orelse:
                elses.extend(self.gen_stmt(else_stmt))

            test_res_lines, test_result = load(test_result, self, no_mov=True)

            lines.extend([
                while_start,
                *test_instrs,
                *test_res_lines,
                Ins("test", test_result, test_result),
                Ins("jz", while_else),
                *body,
                Ins("jmp", while_start),
                while_else,
                *elses,
                while_end
            ])

        elif isinstance(stmt, ast.Break):
            if parent_passed_block:
                lines.append("STMT::Break")
                lines.append(Ins("jmp", parent_passed_block))
            else:
                raise SyntaxError("'break' outside of loop")

        elif isinstance(stmt, ast.If):
            lines.append("STMT::If")

            jump_true = Block(prefix=".if_true")
            jump_false = Block(prefix=".if_false")

            instrs, test_result = self.gen_expr(stmt.test,
                true_short_circuit_block=jump_true,
                false_short_circuit_block=jump_false
            )
            lines.extend(instrs)

            body: LinesType = []

            for body_stmt in stmt.body:
                body.extend(self.gen_stmt(body_stmt))

            elses: LinesType = []

            for else_stmt in stmt.orelse:
                elses.extend(self.gen_stmt(else_stmt))
            
            test_res_lines, test_result = load(test_result, self, no_mov=True)

            lines.extend([
                *test_res_lines,
                Ins("test", test_result, test_result),
                Ins("jz", jump_false),
                *body,
                Ins("jmp", jump_true) if elses else "-- End of if chain.",
                jump_false,
                *elses,
                jump_true
            ])

        elif isinstance(stmt, ast.Return):
            lines.append("STMT::Return")
            if stmt.value:
                instrs, value = self.gen_expr(stmt.value)
                lines.extend(instrs)

                lines.extend(self.return_value(value))
            else:
                lines.extend(self.return_value())
        else:
            raise NotImplementedError(f"The ast.stmt token {stmt.__class__.__name__} is not implemented yet.")

        return lines


                    