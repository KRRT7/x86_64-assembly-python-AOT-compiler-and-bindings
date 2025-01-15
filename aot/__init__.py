from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from types import ModuleType

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
from aot.stack import Stack, StackFrame
from aot.function import PythonFunction, mangle_function_name

# std lib imports
import textwrap
import inspect
import ast
import functools
from time import perf_counter_ns
from typing import Callable, TypeVar

# types and type aliases
PF = PythonFunction

@dataclass
class Benchmark:
    asm_faster: bool = False
    tested_python: bool = False
    tested_asm: bool = False
    asm_time: int = 0
    python_time: int = 0
    

class CompiledFunction:

    def __init__(self, template_keys: list[TypeVar]|None, no_bench:bool):
        self.original_function: Callable|None = None
        self.is_node_found: bool = False
        self.compiled_functions: dict[tuple[type, ...]|None, tuple[PythonFunction, Benchmark]] = {}
        self.functon_ast_node: ast.FunctionDef | None = None
        self.template_keys = template_keys
        self.no_bench = no_bench
        
    def find_function(self, original_function: Callable):
        # Parse the function's source code to an AST
        if not self.is_node_found:
            self.original_function: Callable|None = original_function
            source_code = textwrap.dedent(inspect.getsource(self.original_function))
            tree = ast.parse(source_code)
            # Find the function node in the AST by its name
            self.functon_ast_node = [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == self.original_function.__name__
            ][0]
            
            self.is_node_found = True
        
    def get_function(self, function: PythonFunction, benchmark: Benchmark, force_lookup: bool) -> Callable:
        "performs benchmarking if needed"
        ret = None
        if self.no_bench:
            ret = lambda *args, **kwargs: function.jit_program.call(function.name, *args, **kwargs, force_lookup=force_lookup)
        elif not benchmark.tested_asm:
            asm_time_start = perf_counter_ns()
            ret = lambda *args, **kwargs: function.jit_program.call(function.name, *args, **kwargs, force_lookup=force_lookup)
            benchmark.asm_time = perf_counter_ns() - asm_time_start
            benchmark.tested_asm = True
            if benchmark.tested_python:
                benchmark.asm_faster = benchmark.python_time > benchmark.asm_time
        elif not benchmark.tested_python:
            python_time_start = perf_counter_ns()
            ret = lambda *args, **kwargs: self.original_function(*args, **kwargs)
            benchmark.python_time = perf_counter_ns() - python_time_start
            benchmark.tested_python = True
            if benchmark.tested_asm:
                benchmark.asm_faster = benchmark.python_time > benchmark.asm_time
        elif benchmark.asm_faster:
            # TODO: Make it call the mangled template name rather than the actual function name
            ret = lambda *args, **kwargs: function.jit_program.call(function.name, *args, **kwargs, force_lookup=force_lookup)
        else:
            ret = lambda *args, **kwargs: self.original_function(*args, **kwargs)
        return ret
    
    def __getitem__(self, template_types:tuple[type, ...]) -> Callable:
        "This gets a specific template directly"
        if template_types and not isinstance(template_types, tuple):
            template_types = (template_types,)
        force_lookup = template_types not in self.compiled_functions
        self.compile(template_types)
        return self.get_function(*self.compiled_functions[template_types], force_lookup=force_lookup)

    def __call__(self, *args):
        "This calls the default function."
        force_lookup = None not in self.compiled_functions
        self.compile(None)
        return self.get_function(*self.compiled_functions[None], force_lookup=force_lookup)(*args)

    def create_template_dict(self, template_types:tuple[type, ...]|type|None):
        if (template_types and self.template_keys) and len(template_types) != len(self.template_keys):
            raise LookupError(f"Expected {len(self.template_keys)} template arguments for function, recieved {len(template_types)}")
        if not template_types:
            return {}
        else:
            ret: OrderedDict[TypeVar, type] = OrderedDict({})
            for i in range(len(self.template_keys)):
                ret[self.template_keys[i]] = template_types[i]

            return ret
                

    def compile(self, template_types:tuple[type, ...]|type|None):
        if template_types and not isinstance(template_types, tuple):
            template_types = (template_types,)
        if template_types not in self.compiled_functions:
            python_func = PF(self.functon_ast_node, Stack(), self.create_template_dict(template_types))
            python_func()
            self.compiled_functions[template_types] = (python_func, Benchmark())
            python_func.jit_program.compile()
            python_func.jit_program.link(
                args=OrderedDict({"shared": None}),
                output_extension=(".so" if current_os == "Linux" else ".dll"),
            )

class X86_64_Function:
    def __init__(self, templates:list[TypeVar]|None = None, no_bench: bool = False):
        self.compiled_function = CompiledFunction(templates, no_bench)

    def __call__(self, func):
        self.compiled_function.find_function(func)
        return self.compiled_function