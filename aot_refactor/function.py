from collections import OrderedDict
from aot_refactor.binop import add_float_float, add_int_int, div_float_float, floordiv_int_int, implicit_cast, mod_int_int, mul_float_float, mul_int_int, sub_float_float, sub_int_int
from aot_refactor.type_imports import *
from aot_refactor.stack import Stack
from aot_refactor.utils import CAST, FUNCTION_ARGUMENTS, FUNCTION_ARGUMENTS_FLOAT, load, type_from_object, type_from_str
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
                    self.return_variable = Variable("RETURN", int, Reg("rax"))
                case "float":
                    self.return_variable = Variable("RETURN", float, Reg("xmm0"))
                case _:
                    raise SyntaxError(
                        f'Unsupported return type "{self.python_function_ast.returns.id}"'
                        f' for compiled function {self.name}.'
                    )
        else:        
            self.return_variable = Variable("RETURN", None, Reg("xmm0"))
                
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
        for a_n, argument in enumerate(self.python_function_ast.args.args):
            variable_store = python_type = None
            match argument.annotation.id:
                case "int":
                    python_type = int
                    if current_os == "Linux" and len(int_args):
                        variable_store = int_args.pop()
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
                            Reg("rbp",{int}),
                            16 + 8 * (a_n - len(FUNCTION_ARGUMENTS_FLOAT))
                            if current_os == "Linux"
                            else 32 + 16 + 8 * (a_n - len(FUNCTION_ARGUMENTS_FLOAT)),
                            meta_tags={float},
                            negative=False,
                        )
            if python_type is None:
                raise TypeError(f"Function argument ({argument.arg}) type for compiled function cannot be None.")
            self.arguments[argument.arg] = Variable(argument.arg, python_type, variable_store)

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

        lines.extend(self.stack.pop())
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
        elif isinstance(expr, ast.Name):
            lines.append(f'label::"{expr.id}"')
            if self.var_exists(expr.id):
                return lines, self.get_var(expr.id)
            elif variable_python_type:
                instrs = self.stack.allocate(expr.id, variable_python_type)
                lines.extend(instrs)
                return lines, self.get_var(expr.id)
            else:
                raise TypeError("Expected variable_python_type argument to be set.")
        elif isinstance(expr, ast.BinOp):
            return self.gen_binop(expr.left, expr.op, expr.right)
        else:
            raise SyntaxError(f"The ast.expr token {expr.__class__.__name__} is not implemented yet.")
        
    def gen_binop(self, left:ast.expr, operator:ast.operator, right:ast.expr) -> tuple[LinesType, VariableValueType|ScalarType]:
        lines: LinesType = []
        instrs, left_value = self.gen_expr(left)
        lines.extend(instrs)
        instrs, right_value = self.gen_expr(right)
        lines.extend(instrs)

        if type_from_object(left_value) is not type_from_object(right_value):
            instrs, left_value, right_value = implicit_cast(self, left_value, right_value)
            lines.extend(instrs)

        left_value_type = type_from_object(left_value)
        right_value_type = type_from_object(right_value)

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
            
        elif isinstance(operator, ast.Mod):

            # both are int
            if left_value_type is int and right_value_type is int:
                instrs, result_memory = mod_int_int(self, left_value, right_value)
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
                raise SyntaxError(f"The ast.BinOp token {operator} is not implemented yet for {left_value_type.__name__} and {right_value_type.__name__} operations.")
        else:
            raise SyntaxError(f"The ast.BinOp token {operator} is not implemented yet.")

        
    
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
            lines.append("STMT::AnnAssign")
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
            raise SyntaxError(f"The ast.stmt token {stmt.__class__.__name__} is not implemented yet.")

        return lines


                    