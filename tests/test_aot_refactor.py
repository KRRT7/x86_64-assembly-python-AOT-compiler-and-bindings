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

@x86_64_compile()
def asm_boolean_add(arg1:bool, arg2:bool) -> int:
    return arg1 + arg2

@x86_64_compile()
def asm_boolean_add_int(arg1:bool, arg2:int) -> int:
    return arg1 + arg2

@x86_64_compile()
def asm_boolean_sub_int(arg1:bool, arg2:int) -> int:
    return arg1 - arg2

@x86_64_compile()
def asm_boolean_fdiv_int(arg1:bool, arg2:int) -> int:
    return arg1 // arg2

@x86_64_compile()
def asm_boolean_fdiv_bool(arg1:bool, arg2:bool) -> int:
    return arg1 // arg2

@x86_64_compile()
def asm_boolean_mod_bool(arg1:bool, arg2:bool) -> int:
    return arg1 % arg2

@x86_64_compile()
def asm_boolean_mod_int(arg1:bool, arg2:int) -> int:
    return arg1 % arg2

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

    def test_asm_boolean_operation1(self):
        self.assertEqual(asm_boolean_add(True,True), True+True)
        
    def test_asm_boolean_operation2(self):
        self.assertEqual(asm_boolean_add_int(True,2), True+2)

    def test_asm_boolean_operation3(self):
        self.assertEqual(asm_boolean_fdiv_bool(True,True), True//True)

    def test_asm_boolean_operation4(self):
        self.assertEqual(asm_boolean_fdiv_int(True,7), True//7)

    def test_asm_boolean_operation5(self):
        self.assertEqual(asm_boolean_mod_bool(True,True), True%True)

    def test_asm_boolean_operation6(self):
        self.assertEqual(asm_boolean_mod_int(True,7), True%7)
        

if __name__ == '__main__':
    unittest.main(testRunner=TestAOT())