# local imports
from typing import Any, Callable
import unittest
from aot import CompiledFunction, X86_64_Function, Array, Template

from time import perf_counter_ns

@X86_64_Function(no_bench=True)
def asm_assign(t:int):
    val:int = t
    return

@X86_64_Function(no_bench=True)
def asm_assign_and_ret(t:int) -> int:
    val:int = t
    return val

@X86_64_Function(no_bench=True)
def asm_assign_binary_add_constants(t:int) -> int:
    val:int = 2 + 3
    return val

@X86_64_Function(no_bench=True)
def asm_assign_binary_add_argument(t:int) -> int:
    val:int = t + t
    return val

@X86_64_Function(no_bench=True)
def asm_assign_binary_floordiv_constants(t:int) -> int:
    val:int = 3 // 2
    return val

@X86_64_Function(no_bench=True)
def asm_assign_binary_floordiv_argument(t:int) -> int:
    val:int = t // t
    return val

@X86_64_Function(no_bench=True)
def asm_assign_binary_floordiv_argument_and_constant(t:int) -> int:
    val:int = t // 2
    return val

@X86_64_Function(no_bench=True)
def asm_assign_binary_mod_constants(t:int) -> int:
    val:int = 3 % 2
    return val

@X86_64_Function(no_bench=True)
def asm_assign_binary_mod_argument(t:int) -> int:
    val:int = t % t
    return val

@X86_64_Function(no_bench=True)
def asm_assign_binary_mod_argument_and_constant(t:int) -> int:
    val:int = t % 2
    return val

@X86_64_Function(no_bench=True)
def asm_assign_binary_add_argument_and_constant_implicit_cast_float(t:int) -> float:
    val:float = t + 2.5
    return val

@X86_64_Function(no_bench=True)
def asm_assign_binary_sub_argument_and_constant_implicit_cast_float(t:int) -> float:
    val:float = t - 2.5
    return val

@X86_64_Function(no_bench=True)
def asm_assign_binary_mul_argument_and_constant_implicit_cast_float(t:int) -> float:
    val:float = t * 2.5
    return val

@X86_64_Function(no_bench=True)
def asm_assign_binary_div_argument_and_constant_implicit_cast_float(t:int) -> float:
    val:float = t / 2.5
    return val

@X86_64_Function(no_bench=True)
def asm_div_int_arg_and_int_const(t:int) -> float:
    val:float = t / 2
    return val

@X86_64_Function(no_bench=True)
def asm_lots_of_random_stuff(arg1:int, arg2:float, arg3:int) -> float:
    val:float = arg1 / arg2
    val += arg3
    return val

@X86_64_Function(no_bench=True)
def asm_casting_check(arg1:int, arg2:float, arg3:int) -> float:
    val:float = arg1/arg2+arg3
    return val

@X86_64_Function(no_bench=True)
def asm_boolean_add(arg1:bool, arg2:bool) -> int:
    return arg1 + arg2

@X86_64_Function(no_bench=True)
def asm_boolean_add_int(arg1:bool, arg2:int) -> int:
    return arg1 + arg2

@X86_64_Function(no_bench=True)
def asm_boolean_sub_int(arg1:bool, arg2:int) -> int:
    return arg1 - arg2

@X86_64_Function(no_bench=True)
def asm_boolean_fdiv_int(arg1:bool, arg2:int) -> int:
    return arg1 // arg2

@X86_64_Function(no_bench=True)
def asm_boolean_fdiv_bool(arg1:bool, arg2:bool) -> int:
    return arg1 // arg2

@X86_64_Function(no_bench=True)
def asm_boolean_mod_bool(arg1:bool, arg2:bool) -> int:
    return arg1 % arg2

@X86_64_Function(no_bench=True)
def asm_boolean_mod_int(arg1:bool, arg2:int) -> int:
    return arg1 % arg2

@X86_64_Function(no_bench=True)
def asm_boolean_mod_float(arg1:bool, arg2:float) -> float:
    return arg1 % arg2

@X86_64_Function(no_bench=True)
def asm_boolean_and(arg1:bool, arg2:bool) -> bool:
    return arg1 and arg2

@X86_64_Function(no_bench=True)
def asm_boolean_or(arg1:bool, arg2:bool) -> bool:
    return arg1 or arg2

@X86_64_Function(no_bench=True)
def asm_compare_random(arg1:int, arg2:float, arg3:int) -> bool:
    return 2 <= arg1 < arg2 or arg3 == arg1

@X86_64_Function(no_bench=True)
def is_even_add_3(arg1:int) -> int:
    if arg1 == 2:
        return arg1 + 7
    elif arg1 % 2 == 0:
        return arg1 + 3
    else:
        return arg1
    
@X86_64_Function(no_bench=True)
def is_even_add_3_nested(arg1:int, cond:bool) -> int:
    if arg1 == 2:
        return arg1 + 7
    elif arg1 % 2 == 0:
        if cond:
            return arg1 + 3
        else:
            return 0
    else:
        return arg1
    
@X86_64_Function(no_bench=True)
def while_loop(arg1:int) -> int:
    counter:int = 0
    ret:int = 0
    while counter < arg1:
        ret += 2
        counter += 1
    return ret

T = Template["T"]
@X86_64_Function([T], no_bench=True)
def while_loop_template(arg1: T) -> T:
    counter: T = 0.0
    ret: T = 0.0
    while counter < arg1:
        ret += 2.0
        counter += 1.0
    return ret

@X86_64_Function([T], no_bench=True)
def index_array(arg1: Array[T, 5], arg2: int) -> T:
    return arg1[arg2]

SizeT = Template["SizeT"]
@X86_64_Function([T, SizeT], no_bench=True)
def index_array_templated_size(arg1: Array[T, SizeT], arg2: int) -> T:
    return arg1[arg2]

Const = Template["Const"]
@X86_64_Function([T, SizeT, Const], no_bench=True)
def index_array_templated_const(arg1: Array[T, SizeT], arg2: int) -> T:
    return arg1[arg2] + Const

@X86_64_Function(no_bench=True)
def add_many_floats(a1:float,a2:float,a3:float,a4:float,a5:float,a6:float,a7:float,a8:float,a9:float,a10:float) -> float:
    return a1+a2+a3+a4+a5+a6+a7+a8+a9+a10

@X86_64_Function([SizeT], no_bench=True)
def compiled_sum(values: Array[int, SizeT]) -> int:
    i:int = 0
    total:int = 0
    while i < SizeT:
        total += values[i]
        i += 1
    
    return total

@X86_64_Function(no_bench=True)
def array_literal_test(index:int) -> int:
    my_array:Array[int, 5] = [1,2,3,4,5]
    return my_array[index]

@X86_64_Function(no_bench=True)
def array_nested_literal_test(index:int, index_2:int) -> int:
    my_array:Array[Array[int, 3], 3] = [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ]
    return my_array[index][index_2]

class TestAOT(unittest.TestCase):
    
    def setUp(self):
        print(f"\nRunning {self._testMethodName}:")

    def test_assign(self):
        asm_assign(5)

    def bench_mark_run(self, func:CompiledFunction, args:tuple[Any, ...], templates:tuple[type, ...]|None = None, python_function_override:Callable|None = None):
        if templates:
            func[templates](*args)
            start_asm = perf_counter_ns()
            asm_res = func[templates](*args)
            end_asm = perf_counter_ns()
            asm_bench = end_asm - start_asm
        else:
            func(*args)
            start_asm = perf_counter_ns()
            asm_res = func(*args)
            end_asm = perf_counter_ns()
            asm_bench = end_asm - start_asm

        if python_function_override:
            start_pyt = perf_counter_ns()
            pyt_res = python_function_override(*args)
            end_pyt = perf_counter_ns()
            pyt_bench = end_pyt - start_pyt
        else:
            start_pyt = perf_counter_ns()
            pyt_res = func.original_function(*args)
            end_pyt = perf_counter_ns()
            pyt_bench = end_pyt - start_pyt
        
        print(f"""
  Run with args {args}:

      ASM BENCH {asm_bench / 1_000_000 : 20.5f}

      PYT BENCH {pyt_bench / 1_000_000 : 20.5f}

      ASM RES {asm_res}

      PYT RES {pyt_res}

""")
        self.assertEqual(asm_res, pyt_res)


    def test_assign_and_ret(self):
        self.bench_mark_run(asm_assign_and_ret, (5,))

    def test_assign_binary_add_constants(self):
        self.bench_mark_run(asm_assign_binary_add_constants, (5,))

    def test_assign_binary_add_variables(self):
        self.bench_mark_run(asm_assign_binary_add_argument, (5,))

    def test_assign_binary_floordiv_constants(self):
        self.bench_mark_run(asm_assign_binary_floordiv_constants, (5,))

    def test_assign_binary_floordiv_variables(self):
        self.bench_mark_run(asm_assign_binary_floordiv_argument, (5,))

    def test_assign_binary_floordiv_variables(self):
        self.bench_mark_run(asm_assign_binary_floordiv_argument_and_constant, (5,))

    def test_assign_binary_mod_constants(self):
        self.bench_mark_run(asm_assign_binary_mod_constants, (5,))

    def test_assign_binary_mod_variables(self):
        self.bench_mark_run(asm_assign_binary_mod_argument, (5,))

    def test_assign_binary_mod_variables(self):
        self.bench_mark_run(asm_assign_binary_mod_argument_and_constant, (5,))

    def test_assign_binary_add_argument_and_constant_implicit_cast_float(self):
        self.bench_mark_run(asm_assign_binary_add_argument_and_constant_implicit_cast_float, (5,))

    def test_assign_binary_sub_argument_and_constant_implicit_cast_float(self):
        self.bench_mark_run(asm_assign_binary_sub_argument_and_constant_implicit_cast_float, (5,))

    def test_assign_binary_mul_argument_and_constant_implicit_cast_float(self):
        self.bench_mark_run(asm_assign_binary_mul_argument_and_constant_implicit_cast_float, (5,))

    def test_assign_binary_div_argument_and_constant_implicit_cast_float(self):
        self.bench_mark_run(asm_assign_binary_div_argument_and_constant_implicit_cast_float, (5,))

    def test_asm_div_int_arg_and_int_const(self):
        self.bench_mark_run(asm_div_int_arg_and_int_const, (6,))

    def test_asm_lots_of_random_stuff(self):
        self.bench_mark_run(asm_lots_of_random_stuff, (6,4.0,3))

    def test_asm_casting_check(self):
        self.bench_mark_run(asm_casting_check, (6,4.0,3))

    def test_asm_boolean_operation1(self):
        self.bench_mark_run(asm_boolean_add, (True,True))

    def test_asm_boolean_operation2(self):
        self.bench_mark_run(asm_boolean_add_int, (True,2))

    def test_asm_boolean_operation3(self):
        self.bench_mark_run(asm_boolean_fdiv_bool, (True,True))

    def test_asm_boolean_operation4(self):
        self.bench_mark_run(asm_boolean_fdiv_int, (True,7))

    def test_asm_boolean_operation5(self):
        self.bench_mark_run(asm_boolean_mod_bool, (True,True))

    def test_asm_boolean_operation6(self):
        self.bench_mark_run(asm_boolean_mod_int, (True,7))

    def test_asm_boolean_operation7(self):
        self.bench_mark_run(asm_boolean_mod_float, (True,7.0))

    def test_asm_boolean_operation8(self):
        self.bench_mark_run(asm_boolean_and, (True,True))

    def test_asm_boolean_operation9(self):
        self.bench_mark_run(asm_boolean_or, (True,True))

    def test_asm_compare_random(self):
        self.bench_mark_run(asm_compare_random, (7,5.0,2))

    def test_is_even_add_3(self):
        self.bench_mark_run(is_even_add_3, (4,))
        self.bench_mark_run(is_even_add_3, (3,))
        self.bench_mark_run(is_even_add_3, (2,))

    def test_is_even_add_3(self):
        self.bench_mark_run(is_even_add_3_nested, (4,True) )
        self.bench_mark_run(is_even_add_3_nested, (4,False))
        self.bench_mark_run(is_even_add_3_nested, (3,True) )
        self.bench_mark_run(is_even_add_3_nested, (3,False))
        self.bench_mark_run(is_even_add_3_nested, (2,True) )
        self.bench_mark_run(is_even_add_3_nested, (2,False))
        
    def test_while_loop(self):
        self.bench_mark_run(while_loop, (5,))

    def test_while_loop_template(self):
        self.bench_mark_run(while_loop_template, (        5.0,), templates=(float,))
        self.bench_mark_run(while_loop_template, (        5,),   templates=(int,)  )
        self.bench_mark_run(while_loop_template, (   50_000.0,), templates=(float,))
        self.bench_mark_run(while_loop_template, (   50_000,),   templates=(int,)  )
        self.bench_mark_run(while_loop_template, (5_000_000.0,), templates=(float,))
        self.bench_mark_run(while_loop_template, (5_000_000,),   templates=(int,)  )

    def test_while_loop_template_after(self):
        self.bench_mark_run(while_loop, (7,))

    def test_index_array(self):
        self.bench_mark_run(index_array, ([1,2,3,4,5], 3), templates=(int,))

    def test_index_array_templated_size(self):
        self.bench_mark_run(index_array_templated_size, ([1,2,3,4,5], 3), templates=(int, 5))
        self.bench_mark_run(index_array_templated_size, ([1,2,3,4,5,6], 3), templates=(int, 6))

    def test_add_many_floats(self):
        self.bench_mark_run(add_many_floats, (1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,))

    def test_index_array_templated_size(self):
        self.assertEqual(index_array_templated_const[int, 5, 3]([1,2,3,4,5], 3), 7)
        self.assertEqual(index_array_templated_const[int, 7, 4]([1,2,3,4,5,6,7], 4), 9)

    def test_compiled_sum(self):
        numbers = [*range(0,20)]
        numbers = [*range(0,25)]
        self.bench_mark_run(compiled_sum, (numbers,), (len(numbers),), sum)
        self.bench_mark_run(compiled_sum, (numbers,), (len(numbers),), sum)


    def test_array_literal(self):
        self.bench_mark_run(array_literal_test, (3,))

    def test_array_nested_literal(self):
        self.bench_mark_run(array_nested_literal_test, (2, 2))

if __name__ == '__main__':
    unittest.main(testRunner=TestAOT())