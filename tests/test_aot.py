
# local imports
from aot import x86_64_compile

# std lib imports
from time import perf_counter_ns
import unittest

@x86_64_compile()
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

@x86_64_compile()
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

@x86_64_compile()
def asm_f_add_test() -> float:
    f:float = 0.002
    f = f + 0.003
    return f + f

def python_f_add_test() -> float:
    f:float = 0.002
    f = f + 0.003
    return f + f

@x86_64_compile()
def asm_f_mul_test() -> float:
    f:float = 0.002
    f *= 0.003
    return f * f

def python_f_mul_test() -> float:
    f:float = 0.002
    f *= 0.003
    return f * f

@x86_64_compile()
def asm_f_div_test() -> float:
    f:float = 0.002
    f = f / 0.003
    return f / 0.15

def python_f_div_test() -> float:
    f:float = 0.002
    f = f / 0.003
    return f / 0.15

@x86_64_compile()
def asm_f_dot(x1:float,y1:float,z1:float, x2:float,y2:float,z2:float) -> float:
    f_n1:float = z2 * z1
    f:float = 3.1
    f_n2:float = z2
    return x1*x2+y1*y2+z1*z2

def python_f_dot(x1:float,y1:float,z1:float, x2:float,y2:float,z2:float) -> float:
    f_n1:float = z2 * z1
    f:float = 3.1
    f_n2:float = z2
    return x1*x2+y1*y2+z1*z2

@x86_64_compile()
def asm_i_dot(x1:int,y1:int,z1:int, x2:int,y2:int,z2:int) -> int:
    return x1*x2+y1*y2+z1*z2

def python_i_dot(x1:int,y1:int,z1:int, x2:int,y2:int,z2:int) -> int:
    return x1*x2+y1*y2+z1*z2

@x86_64_compile()
def asm_aug_assign_f(inp:float) -> float:
    f:float = 200.34 + 22.3
    inp += 1.2 -f
    inp -= 0.1 * f
    inp /= 0.5 + f + f - inp
    inp *= 1.3 / f
    return inp

def python_aug_assign_f(inp:float) -> float:
    f:float = 200.34 + 22.3
    inp += 1.2 - f
    inp -= 0.1 * f
    inp /= 0.5 + f + f - inp
    inp *= 1.3 / f
    return inp

@x86_64_compile()
def asm_aug_assign_i(inp:int) -> int:
    i:int = 2 + 22
    inp += 1 - i
    inp -= 3 * i
    inp //= 4 + i + i - inp + 1
    inp *= 500 // (i + 1)
    return inp


def python_aug_assign_i(inp:int) -> int:
    i:int = 2 + 22
    inp += 1 - i
    inp -= 3 * i
    inp //= 4 + i + i - inp + 1
    inp *= 500 // (i + 1)
    return inp

class TestAOT(unittest.TestCase):
    
    def setUp(self):
        print(f"\nRunning {self._testMethodName}:")

    def test_1_000_000_itterations_int(self):
        start = perf_counter_ns()
        totala = 3
        totala = add_a_b(totala, 2)
        print(f"\tassembly    returns = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        start = perf_counter_ns()
        totalp = 3
        totalp = python_add_a_b(totalp, 2)
        print(f"\tpython      returns = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        self.assertEqual(totala, totalp)

    def test_1_000_000_itterations_float(self):
        start = perf_counter_ns()
        totala = 0.003
        totala = asm_add_floats(totala, 0.002)
        print(f"\tassembly    returns = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        start = perf_counter_ns()
        totalp = 0.003
        totalp = python_add_floats(totalp, 0.002)
        print(f"\tpython      returns = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        self.assertEqual(totala, totalp)

    def test_f_add(self):
        start = perf_counter_ns()
        totala = asm_f_add_test()
        print(f"\tassembly    f_add_test (0.002 + 0.003) * 2 = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        start = perf_counter_ns()
        totalp = python_f_add_test()
        print(f"\tpython      f_add_test (0.002 + 0.003) * 2 = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        self.assertEqual(totala, totalp)

    def test_f_mul(self):
        start = perf_counter_ns()
        totala = asm_f_mul_test()
        print(f"\tassembly    f_mul_test (0.002 * 0.003)^2 = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        start = perf_counter_ns()
        totalp = python_f_mul_test()
        print(f"\tpython      f_mul_test (0.002 * 0.003)^2 = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        self.assertEqual(totala, totalp)

    def test_f_div(self):
        start = perf_counter_ns()
        totala = asm_f_div_test()
        print(f"\tassembly    f_div_test 0.002 / 0.003 / 0.15 = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        start = perf_counter_ns()
        totalp = python_f_div_test()
        print(f"\tpython      f_div_test 0.002 / 0.003 / 0.15 = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        self.assertEqual(totala, totalp)

    def test_f_dot_prod(self):
        f_dot_args = (*(v1:=(5.3,2.99,5.2)), *(v2:=(50.2,4.3,1.2)))

        start = perf_counter_ns()
        totala = asm_f_dot(*f_dot_args)
        print(f"\tassembly    {v1} . {v2} = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        start = perf_counter_ns()
        totalp = python_f_dot(*f_dot_args)
        print(f"\tpython      {v1} . {v2} = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        self.assertEqual(totala, totalp)

    def test_i_dot_prod(self):
        i_dot_args = (*(v1:=(5,2,5)), *(v2:=(3,4,1)))

        start = perf_counter_ns()
        totala = asm_i_dot(*i_dot_args)
        print(f"\tassembly    {v1} . {v2} = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        start = perf_counter_ns()
        totalp = python_i_dot(*i_dot_args)
        print(f"\tpython      {v1} . {v2} = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        self.assertEqual(totala, totalp)

    def test_f_dot_prod_benchmarked(self):
        f_dot_args = (*(v1:=(5.3,2.99,5.2)), *(v2:=(50.2,4.3,1.2)))

        # run twice for python benchmark
        a = asm_f_dot(*f_dot_args)
        p = python_f_dot(*f_dot_args)
        self.assertEqual(a, p)

        a = asm_f_dot(*f_dot_args)
        p = python_f_dot(*f_dot_args)
        self.assertEqual(a, p)

        start = perf_counter_ns()
        totala = asm_f_dot(*f_dot_args)
        print(f"\tassembly    {v1} . {v2} = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        start = perf_counter_ns()
        totalp = python_f_dot(*f_dot_args)
        print(f"\tpython      {v1} . {v2} = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")
        
        self.assertEqual(totala, totalp)

    def test_1_000_000_itterations_float_benchmarked(self):

        # run twice for python benchmark
        a = 0.003
        a = asm_add_floats(a, 0.002)
        p = 0.003
        p = python_add_floats(p, 0.002)
        self.assertEqual(a, p)

        a = 0.003
        a = asm_add_floats(a, 0.002)
        p = 0.003
        p = python_add_floats(p, 0.002)
        self.assertEqual(a, p)

        start = perf_counter_ns()
        totala = 0.003
        totala = asm_add_floats(totala, 0.002)
        print(f"\tassembly    returns = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        start = perf_counter_ns()
        totalp = 0.003
        totalp = python_add_floats(totalp, 0.002)
        print(f"\tpython      returns = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        self.assertEqual(totala, totalp)

    def test_augassign_float(self):
        start = perf_counter_ns()
        totala = asm_aug_assign_f(3.14)
        print(f"\tassembly    returns = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        start = perf_counter_ns()
        totalp = python_aug_assign_f(3.14)
        print(f"\tpython      returns = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        self.assertEqual(totala, totalp)

    def test_augassign_int(self):
        start = perf_counter_ns()
        totala = asm_aug_assign_i(900)
        print(f"\tassembly    returns = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        start = perf_counter_ns()
        totalp = python_aug_assign_i(900)
        print(f"\tpython      returns = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

        self.assertEqual(totala, totalp)

if __name__ == '__main__':
    unittest.main(testRunner=TestAOT())