from __future__ import annotations
from collections import OrderedDict

# local imports
from x86_64_assembly_bindings import (
    Register,
    Instruction,
    MemorySize,
    Program,
    Block,
    Function,
    OffsetRegister,
    StackVariable,
    RegisterData,
    InstructionData,
    Memory,
    current_os,
)
from aot_refactor.stack import Stack, StackFrame
from aot_refactor.function import PythonFunction

# std lib imports
import textwrap
import inspect
import ast
import functools
from time import perf_counter_ns
from typing import Callable

# types and type aliases
PF = PythonFunction


class MetaCompiledFunction(Callable):
    is_emitted: bool
    is_compiled: bool
    is_linked: bool
    asm_faster: bool
    tested_python: bool
    tested_asm: bool
    asm_time: int
    python_time: int

class CompiledFunction(Callable):
    original_function: Callable

def x86_64_compile(no_bench: bool = False):
    def decorator(func: MetaCompiledFunction) -> CompiledFunction:
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
                PF.jit_program.compile()
                func.is_compiled = True
            if not func.is_linked:
                PF.jit_program.link(
                    args=OrderedDict({"shared": None}),
                    output_extension=(".so" if current_os == "Linux" else ".dll"),
                )
                func.is_linked = True

            # Call the original function
            ret = None
            if no_bench:
                ret = PF.jit_program.call(func.__name__, *args, **kwargs)
            elif not func.tested_asm:
                asm_time_start = perf_counter_ns()
                ret = PF.jit_program.call(func.__name__, *args, **kwargs)
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
                ret = PF.jit_program.call(func.__name__, *args, **kwargs)
            else:
                ret = func(*args, **kwargs)
            return ret
        
        setattr(wrapper, "original_function", func)

        return wrapper

    return decorator
