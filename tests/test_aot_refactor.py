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

@x86_64_compile()
def asm_boolean_mod_float(arg1:bool, arg2:float) -> float:
    return arg1 % arg2

@x86_64_compile()
def asm_boolean_and(arg1:bool, arg2:bool) -> bool:
    return arg1 and arg2

@x86_64_compile()
def asm_boolean_or(arg1:bool, arg2:bool) -> bool:
    return arg1 or arg2

@x86_64_compile()
def asm_compare_random(arg1:int, arg2:float, arg3:int) -> bool:
    return 2 <= arg1 < arg2 or arg3 == arg1

@x86_64_compile()
def is_even_add_3(arg1:int) -> int:
    if arg1 == 2:
        return arg1 + 7
    elif arg1 % 2 == 0:
        return arg1 + 3
    else:
        return arg1
    
@x86_64_compile()
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
    

class TestAOT(unittest.TestCase):
    
    def setUp(self):
        print(f"\nRunning {self._testMethodName}:")

    def test_assign(self):
        asm_assign(5)

    def test_assign_and_ret(self):
        self.assertEqual(asm_assign_and_ret(5), asm_assign_and_ret.original_function(5))

    def test_assign_binary_add_constants(self):
        self.assertEqual(asm_assign_binary_add_constants(5), asm_assign_binary_add_constants.original_function(5))

    def test_assign_binary_add_variables(self):
        self.assertEqual(asm_assign_binary_add_argument(5), asm_assign_binary_add_argument.original_function(5))

    def test_assign_binary_floordiv_constants(self):
        self.assertEqual(asm_assign_binary_floordiv_constants(5), asm_assign_binary_floordiv_constants.original_function(5))

    def test_assign_binary_floordiv_variables(self):
        self.assertEqual(asm_assign_binary_floordiv_argument(5), asm_assign_binary_floordiv_argument.original_function(5))

    def test_assign_binary_floordiv_variables(self):
        self.assertEqual(asm_assign_binary_floordiv_argument_and_constant(5), asm_assign_binary_floordiv_argument_and_constant.original_function(5))

    def test_assign_binary_mod_constants(self):
        self.assertEqual(asm_assign_binary_mod_constants(5), asm_assign_binary_mod_constants.original_function(5))

    def test_assign_binary_mod_variables(self):
        self.assertEqual(asm_assign_binary_mod_argument(5), asm_assign_binary_mod_argument.original_function(5))

    def test_assign_binary_mod_variables(self):
        self.assertEqual(asm_assign_binary_mod_argument_and_constant(5), asm_assign_binary_mod_argument_and_constant.original_function(5))

    def test_assign_binary_add_argument_and_constant_implicit_cast_float(self):
        self.assertEqual(asm_assign_binary_add_argument_and_constant_implicit_cast_float(5), asm_assign_binary_add_argument_and_constant_implicit_cast_float.original_function(5))

    def test_assign_binary_sub_argument_and_constant_implicit_cast_float(self):
        self.assertEqual(asm_assign_binary_sub_argument_and_constant_implicit_cast_float(5), asm_assign_binary_sub_argument_and_constant_implicit_cast_float.original_function(5))

    def test_assign_binary_mul_argument_and_constant_implicit_cast_float(self):
        self.assertEqual(asm_assign_binary_mul_argument_and_constant_implicit_cast_float(5), asm_assign_binary_mul_argument_and_constant_implicit_cast_float.original_function(5))

    def test_assign_binary_div_argument_and_constant_implicit_cast_float(self):
        self.assertEqual(asm_assign_binary_div_argument_and_constant_implicit_cast_float(5), asm_assign_binary_div_argument_and_constant_implicit_cast_float.original_function(5))

    def test_asm_div_int_arg_and_int_const(self):
        self.assertEqual(asm_div_int_arg_and_int_const(6), asm_div_int_arg_and_int_const.original_function(6))

    def test_asm_lots_of_random_stuff(self):
        self.assertEqual(asm_lots_of_random_stuff(6,4.0,3), asm_lots_of_random_stuff.original_function(6,4.0,3))

    def test_asm_casting_check(self):
        self.assertEqual(asm_casting_check(6,4.0,3), asm_casting_check.original_function(6,4.0,3))

    def test_asm_boolean_operation1(self):
        self.assertEqual(asm_boolean_add(True,True), asm_boolean_add.original_function(True,True))

    def test_asm_boolean_operation2(self):
        self.assertEqual(asm_boolean_add_int(True,2), asm_boolean_add_int.original_function(True,2))

    def test_asm_boolean_operation3(self):
        self.assertEqual(asm_boolean_fdiv_bool(True,True), asm_boolean_fdiv_bool.original_function(True,True))

    def test_asm_boolean_operation4(self):
        self.assertEqual(asm_boolean_fdiv_int(True,7), asm_boolean_fdiv_int.original_function(True,7))

    def test_asm_boolean_operation5(self):
        self.assertEqual(asm_boolean_mod_bool(True,True), asm_boolean_mod_bool.original_function(True,True))

    def test_asm_boolean_operation6(self):
        self.assertEqual(asm_boolean_mod_int(True,7), asm_boolean_mod_int.original_function(True,7))

    def test_asm_boolean_operation7(self):
        self.assertEqual(asm_boolean_mod_float(True,7.0), asm_boolean_mod_float.original_function(True,7.0))

    def test_asm_boolean_operation8(self):
        self.assertEqual(str(asm_boolean_and(True,True)), str(asm_boolean_and.original_function(True,True)))

    def test_asm_boolean_operation9(self):
        self.assertEqual(str(asm_boolean_or(True,True)), str(asm_boolean_or.original_function(True,True)))

    def test_asm_compare_random(self):
        self.assertEqual(asm_compare_random(7,5.0,2), asm_compare_random.original_function(7,5.0,2))

    def test_is_even_add_3(self):
        self.assertEqual(is_even_add_3(4), is_even_add_3.original_function(4))
        self.assertEqual(is_even_add_3(3), is_even_add_3.original_function(3))
        self.assertEqual(is_even_add_3(2), is_even_add_3.original_function(2))

    def test_is_even_add_3(self):
        self.assertEqual(is_even_add_3_nested(4, True), is_even_add_3_nested.original_function(4, True))
        self.assertEqual(is_even_add_3_nested(4, False), is_even_add_3_nested.original_function(4, False))
        self.assertEqual(is_even_add_3_nested(3, True), is_even_add_3_nested.original_function(3, True))
        self.assertEqual(is_even_add_3_nested(3, False), is_even_add_3_nested.original_function(3, False))
        self.assertEqual(is_even_add_3_nested(2, True), is_even_add_3_nested.original_function(2, True))
        self.assertEqual(is_even_add_3_nested(2, False), is_even_add_3_nested.original_function(2, False))
        
        

if __name__ == '__main__':
    unittest.main(testRunner=TestAOT())