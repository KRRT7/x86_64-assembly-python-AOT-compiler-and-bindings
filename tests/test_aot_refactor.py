# local imports
from aot_refactor import x86_64_compile

# std lib imports
from time import perf_counter_ns
import unittest
import random

@x86_64_compile()
def asm_assign(t:int):
    val:int = t
    return

@x86_64_compile()
def asm_assign_and_ret(t:int) -> int:
    val:int = t
    return val

@x86_64_compile()
def asm_assign_binary_add_constants(t:int) -> int:
    val:int = 2 + 3
    return val

@x86_64_compile()
def asm_assign_binary_add_argument(t:int) -> int:
    val:int = t + t
    return val

@x86_64_compile()
def asm_assign_binary_floordiv_constants(t:int) -> int:
    val:int = 3 // 2
    return val

@x86_64_compile()
def asm_assign_binary_floordiv_argument(t:int) -> int:
    val:int = t // t
    return val

@x86_64_compile()
def asm_assign_binary_floordiv_argument_and_constant(t:int) -> int:
    val:int = t // 2
    return val

@x86_64_compile()
def asm_assign_binary_mod_constants(t:int) -> int:
    val:int = 3 % 2
    return val

@x86_64_compile()
def asm_assign_binary_mod_argument(t:int) -> int:
    val:int = t % t
    return val

@x86_64_compile()
def asm_assign_binary_mod_argument_and_constant(t:int) -> int:
    val:int = t % 2
    return val

@x86_64_compile()
def asm_assign_binary_add_argument_and_constant_implicit_cast_float(t:int) -> float:
    val:float = t + 2.5
    return val

@x86_64_compile()
def asm_assign_binary_sub_argument_and_constant_implicit_cast_float(t:int) -> float:
    val:float = t - 2.5
    return val

@x86_64_compile()
def asm_assign_binary_mul_argument_and_constant_implicit_cast_float(t:int) -> float:
    val:float = t * 2.5
    return val

@x86_64_compile()
def asm_assign_binary_div_argument_and_constant_implicit_cast_float(t:int) -> float:
    val:float = t / 2.5
    return val

@x86_64_compile()
def asm_div_int_arg_and_int_const(t:int) -> float:
    val:float = t / 2
    return val

@x86_64_compile()
def asm_lots_of_random_stuff(arg1:int, arg2:float, arg3:int) -> float:
    val:float = arg1 / arg2
    val += arg3
    return val

def lots_of_random_stuff(arg1:int, arg2:float, arg3:int) -> float:
    val:float = arg1 / arg2
    val += arg3
    return val

@x86_64_compile()
def asm_casting_check(arg1:int, arg2:float, arg3:int) -> float:
    val:float = arg1/arg2+arg3
    return val

class TestAOT(unittest.TestCase):
    
    def setUp(self):
        print(f"\nRunning {self._testMethodName}:")

    def test_assign(self):
        asm_assign(5)

    def test_assign_and_ret(self):
        self.assertEqual(asm_assign_and_ret(5), 5)

    def test_assign_binary_add_constants(self):
        self.assertEqual(asm_assign_binary_add_constants(5), 5)

    def test_assign_binary_add_variables(self):
        self.assertEqual(asm_assign_binary_add_argument(5), 10)

    def test_assign_binary_floordiv_constants(self):
        self.assertEqual(asm_assign_binary_floordiv_constants(5), 3//2)

    def test_assign_binary_floordiv_variables(self):
        self.assertEqual(asm_assign_binary_floordiv_argument(5), 5//5)

    def test_assign_binary_floordiv_variables(self):
        self.assertEqual(asm_assign_binary_floordiv_argument_and_constant(5), 5//2)

    def test_assign_binary_mod_constants(self):
        self.assertEqual(asm_assign_binary_mod_constants(5), 3%2)

    def test_assign_binary_mod_variables(self):
        self.assertEqual(asm_assign_binary_mod_argument(5), 5%5)

    def test_assign_binary_mod_variables(self):
        self.assertEqual(asm_assign_binary_mod_argument_and_constant(5), 5%2)

    def test_assign_binary_add_argument_and_constant_implicit_cast_float(self):
        self.assertEqual(asm_assign_binary_add_argument_and_constant_implicit_cast_float(5), 5+2.5)

    def test_assign_binary_sub_argument_and_constant_implicit_cast_float(self):
        self.assertEqual(asm_assign_binary_sub_argument_and_constant_implicit_cast_float(5), 5-2.5)

    def test_assign_binary_mul_argument_and_constant_implicit_cast_float(self):
        self.assertEqual(asm_assign_binary_mul_argument_and_constant_implicit_cast_float(5), 5*2.5)

    def test_assign_binary_div_argument_and_constant_implicit_cast_float(self):
        self.assertEqual(asm_assign_binary_div_argument_and_constant_implicit_cast_float(5), 5/2.5)

    def test_asm_div_int_arg_and_int_const(self):
        self.assertEqual(asm_div_int_arg_and_int_const(6), 6/2)

    def test_asm_lots_of_random_stuff(self):
        self.assertEqual(asm_lots_of_random_stuff(6,4.0,3), lots_of_random_stuff(6,4.0,3))

    def test_asm_casting_check(self):
        self.assertEqual(asm_casting_check(6,4.0,3), 6/4.0+3)

if __name__ == '__main__':
    unittest.main(testRunner=TestAOT())