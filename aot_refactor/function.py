from collections import OrderedDict
from aot_refactor.binop import add_float_float, add_int_int, div_float_float, floordiv_float_float, floordiv_int_int, implicit_cast, mod_float_float, mod_int_int, mul_float_float, mul_int_int, sub_float_float, sub_int_int
from aot_refactor.compare import compare_operator_from_type, implicit_cast_cmp
from aot_refactor.type_imports import *
from aot_refactor.stack import Stack
from aot_refactor.utils import CAST, FUNCTION_ARGUMENTS, FUNCTION_ARGUMENTS_BOOL, FUNCTION_ARGUMENTS_FLOAT, load, reg_request_bool, reg_request_float, type_from_object, type_from_str
from aot_refactor.variable import Variable
from x86_64_assembly_bindings import (
    Program, Function
)
import ast

class PythonFunction:
    jit_program: Program = Program("python_x86_64_jit")


    def __init__(self, python_function_ast: ast.FunctionDef, stack: Stack):
        self.python_function_ast: ast.FunctionDef = python_function_ast
        self.stack: Stack = stack
        
        # Function data
        self.name: str = self.python_function_ast.name
        self.arguments: OrderedDict[str, Variable] = OrderedDict({})
        self.return_variable: Variable | None = None

        # arguments variables
        self.__init_get_args()

        # Get return variable
        if self.python_function_ast.returns:
            match self.python_function_ast.returns.id:
                case "int":
                    self.return_variable = Variable("RETURN", int, Reg("rax", {int}))
                case "float":
                    self.return_variable = Variable("RETURN", float, Reg("xmm0", {float}))
                case "bool":
                    self.return_variable = Variable("RETURN", bool, Reg("al", {bool}))
                case _:
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
            arguments_py_type = [v.python_type for v in self.arguments.values()]
        )


        self.lines: LinesType = []
        for stmt in self.python_function_ast.body:
            self.lines.extend(self.gen_stmt(stmt))
                
    def __init_get_args(self):
        int_args = [*reversed(FUNCTION_ARGUMENTS)]
        float_args = [*reversed(FUNCTION_ARGUMENTS_FLOAT)]
        bool_args = [*reversed(FUNCTION_ARGUMENTS_BOOL)]
        for a_n, argument in enumerate(self.python_function_ast.args.args):
            variable_store = python_type = None
            size = MemorySize.QWORD
            match argument.annotation.id:
                case "int":
                    python_type = int
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
                            meta_tags={int},
                            negative=False,
                        )
                case "float":
                    python_type = float
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
                            meta_tags={float},
                            negative=False,
                        )
                case "bool":
                    python_type = bool
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
                            meta_tags={bool},
                            negative=False,
                            override_size=MemorySize.BYTE,
                        )
            if python_type is None:
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
                line()
        

    def return_value(self, value:Variable|ScalarType|None = None) -> LinesType:
        
        lines: LinesType = []
        if self.return_variable.python_type: # in case it is None
            match self.return_variable.python_type.__name__:
                case "int"|"bool":
                    lines, loaded_value = load(value)
                    lines.append(Ins("mov", self.return_variable.value, loaded_value))
                case "float":
                    lines, loaded_value = load(value)
                    lines.append(Ins("movsd", self.return_variable.value, loaded_value))

        stack_pop_lines = self.stack.pop()
        if stack_pop_lines:
            lines.extend(stack_pop_lines)
        function_ret = lambda *args:self.function.ret(*args)
        setattr(function_ret, "is_return", True)
        lines.append(function_ret)

        return lines
    
    def gen_expr(self, expr: ast.expr, variable_python_type: type | None = None) -> tuple[LinesType, Variable|ScalarType]:
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
            if self.var_exists(expr.id):
                return lines, self.get_var(expr.id)
            elif variable_python_type:
                instrs = self.stack.allocate(expr.id, variable_python_type)
                lines.extend(instrs)
                return lines, self.get_var(expr.id)
            else:
                raise NotImplementedError("Expected variable_python_type argument to be set.")
        elif isinstance(expr, ast.BinOp):
            return self.gen_binop(expr.left, expr.op, expr.right)
        elif isinstance(expr, ast.BoolOp):
            return self.gen_boolop(expr.op, expr.values)
        elif isinstance(expr, ast.Compare):
            return self.gen_compare(expr.left, expr.ops, expr.comparators)
        else:
            raise NotImplementedError(f"The ast.expr token {expr.__class__.__name__} is not implemented yet.")
        
    def gen_compare(self, left:ast.expr, operators:list[ast.cmpop], comparators:list[ast.expr],
    false_short_circuit_block:Block|None = None    
    ) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = []
        values:list[ScalarType | Variable] = []

        instrs, left = self.gen_expr(left)
        lines.extend(instrs)

        for value_expr in comparators:
            instrs, value = self.gen_expr(value_expr)
            lines.extend(instrs)
            values.append(value)

        all_raw_values = [left] + values

        instrs, left = load(left)
        lines.extend(instrs)

        aggregate_value = reg_request_bool(lines=lines)
        lines.append(Ins("mov", aggregate_value, 1))

        short_circuit_block = false_short_circuit_block \
                        if false_short_circuit_block    \
                        else Block(prefix=".cmp_op_on_False_shortcircuit")
        
        for n, (right, op) in enumerate(zip(values, operators)):
            
            instrs, right = load(right)
            lines.extend(instrs)

            left_type, right_type, instrs, left, right= implicit_cast_cmp(op, left, right)
            type_pair = left_type, right_type

            local_result: VariableValueType | ScalarType | Variable | None = None
            if isinstance(op, ast.Eq):
                instrs, local_result = compare_operator_from_type(type_pair, "sete", left, right, short_circuit_block)
                lines.extend(instrs)
            elif isinstance(op, ast.NotEq):
                instrs, local_result = compare_operator_from_type(type_pair, "setne", left, right, short_circuit_block)
                lines.extend(instrs)
            elif isinstance(op, ast.Lt):
                instrs, local_result = compare_operator_from_type(type_pair, "setl", left, right, short_circuit_block)
                lines.extend(instrs)
            elif isinstance(op, ast.LtE):
                instrs, local_result = compare_operator_from_type(type_pair, "setle", left, right, short_circuit_block)
                lines.extend(instrs)
            elif isinstance(op, ast.Gt):
                instrs, local_result = compare_operator_from_type(type_pair, "setg", left, right, short_circuit_block)
                lines.extend(instrs)
            elif isinstance(op, ast.GtE):
                instrs, local_result = compare_operator_from_type(type_pair, "setge", left, right, short_circuit_block)
                lines.extend(instrs)
            elif isinstance(op, ast.Is):
                instrs, local_result = compare_operator_from_type(type_pair, "sete", left, right, short_circuit_block)
                lines.extend(instrs)
            elif isinstance(op, ast.IsNot):
                instrs, local_result = compare_operator_from_type(type_pair, "setne", left, right, short_circuit_block)
                lines.extend(instrs)
            elif isinstance(op, ast.In):
                instrs, local_result = compare_operator_from_type(type_pair, "sete", left, right, short_circuit_block)
                lines.extend(instrs)
            elif isinstance(op, ast.NotIn):
                instrs, local_result = compare_operator_from_type(type_pair, "setne", left, right, short_circuit_block)
                lines.extend(instrs)
            else:
                raise NotImplementedError(f"The comparison operator token {type(op).__name__} is not implemented yet")

            if not local_result:
                raise SyntaxError("Failed to evaluate the local_result.")
            lines.append(Ins("and", aggregate_value, local_result))
            left = right

            local_result.free(lines)

        lines.append(short_circuit_block)

        return lines, aggregate_value
        

    def gen_boolop(self, operator:ast.operator, value_exprs:list[ast.expr],
        true_short_circuit_block:Block|None = None, false_short_circuit_block:Block|None = None
    ) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = []
        values:list[ScalarType | Variable] = []
        for value_expr in value_exprs:
            instrs, value = self.gen_expr(value_expr)
            lines.extend(instrs)
            instrs, value = CAST.bool(value)
            lines.extend(instrs)
            values.append(value)

        instrs, loaded_value = load(values[0])
        lines.extend(instrs)

        aggregate_value = reg_request_bool(lines=lines)

        lines.append(Ins("mov", aggregate_value, loaded_value))

        # Ensure that the aggregate value is populating the zero flag
        lines.append(Ins("test", aggregate_value, aggregate_value))

        higher_order_short_circuit_passed = true_short_circuit_block or false_short_circuit_block
        
        short_circuit_block = None
        if not higher_order_short_circuit_passed:
            short_circuit_block = Block(prefix=".boolop_short_circuit")
        
        for value in values[1::]:
            if isinstance(operator, ast.Or):
                if true_short_circuit_block:
                    # Short circuit to true block if true
                    lines.append(Ins("jnz", short_circuit_block if short_circuit_block else true_short_circuit_block))
                instrs, loaded_value = load(value)
                lines.extend(instrs)
                lines.append(Ins("or", aggregate_value, loaded_value))
            elif isinstance(operator, ast.And):
                # Short circuit to false block if false
                lines.append(Ins("jz", short_circuit_block if short_circuit_block else false_short_circuit_block))
                instrs, loaded_value = load(value)
                lines.extend(instrs)
                lines.append(Ins("and", aggregate_value, loaded_value))
            else:
                raise NotImplementedError(f"Operator Token {operator.__class__.__name__} is not implemented yet.")
        
        if short_circuit_block:
            lines.append(short_circuit_block)


        return lines, aggregate_value

    def gen_binop(self, left:ast.expr, operator:ast.operator, right:ast.expr) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = []
        instrs, left_value = self.gen_expr(left)
        lines.extend(instrs)
        instrs, right_value = self.gen_expr(right)
        lines.extend(instrs)

        left_value_type, right_value_type, instrs, left_value, right_value = implicit_cast(operator, left_value, right_value)
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

                instrs, left_value = CAST.float(left_value)
                lines.extend(instrs)
                
                instrs, right_value = CAST.float(right_value)
                lines.extend(instrs)

                instrs, result_memory = div_float_float(self, left_value, right_value)
                lines.extend(instrs)
                return lines, result_memory
            else:
                raise NotImplementedError(f"The ast.BinOp token {operator} is not implemented yet for {left_value_type.__name__} and {right_value_type.__name__} operations.")
        else:
            raise NotImplementedError(f"The ast.BinOp token {operator} is not implemented yet.")

        
    
    def gen_stmt(self, stmt: ast.stmt) -> LinesType:
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
            lines.append(f"STMT::AnnAssign({stmt.annotation.id})")
            instrs, value = self.gen_expr(stmt.value)
            lines.extend(instrs)

            target = stmt.target
            variable_type = type_from_str(stmt.annotation.id)

            instrs, variable = self.gen_expr(target, variable_python_type=variable_type)
            lines.extend(instrs)
            instrs = variable.set(value)
            lines.extend(instrs)

        elif isinstance(stmt, ast.AugAssign):
            lines.append(f"STMT::AugAssign({stmt.op.__class__.__name__})")

            instrs, value = self.gen_binop(stmt.target, stmt.op, stmt.value)
            lines.extend(instrs)

            instrs, variable = self.gen_expr(stmt.target)
            lines.extend(instrs)

            instrs = variable.set(value)
            lines.extend(instrs)

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


                    