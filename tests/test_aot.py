
# local imports
from aot import x86_64_compile

# std lib imports
from time import perf_counter_ns
import unittest
import random

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
    return x1*x2+y1*y2+z1*z2

def python_f_dot(x1:float,y1:float,z1:float, x2:float,y2:float,z2:float) -> float:
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

@x86_64_compile()
def asm_add_9_f(n1:float, n2:float, n3:float, n4:float, n5:float, n6:float, n7:float, n8:float, n9:float) -> float:
    return n1+n2+n3+n4+n5+n6+n7+n8+n9

def python_add_9_f(n1:float, n2:float, n3:float, n4:float, n5:float, n6:float, n7:float, n8:float, n9:float) -> float:
    return n1+n2+n3+n4+n5+n6+n7+n8+n9

@x86_64_compile()
def asm_add_9_i(n1:int, n2:int, n3:int, n4:int, n5:int, n6:int, n7:int, n8:int, n9:int) -> int:
    return n1+n2+n3+n4+n5+n6+n7+n8+n9

def python_add_9_i(n1:int, n2:int, n3:int, n4:int, n5:int, n6:int, n7:int, n8:int, n9:int) -> int:
    return n1+n2+n3+n4+n5+n6+n7+n8+n9

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
        r=lambda:float(f"{random.randint(-100,100)}.{random.randint(0,10000)}")
        for v1, v2 in [((r(),r(),r()),(r(),r(),r())) for _ in range(0,30)]:
            f_dot_args = (*v1, *v2)

            start = perf_counter_ns()
            totala = asm_f_dot(*f_dot_args)
            print(f"\tassembly    {v1} . {v2} = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

            start = perf_counter_ns()
            totalp = python_f_dot(*f_dot_args)
            print(f"\tpython      {v1} . {v2} = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

            self.assertEqual(totala, totalp)

    def test_i_dot_prod(self):
        r=lambda:random.randint(-100,100)
        for v1, v2 in [((r(),r(),r()),(r(),r(),r())) for _ in range(0,30)]:
            i_dot_args = (*v1, *v2)

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
        for arg in range(-100,101):
            print(f"\t  input = {arg}")
            start = perf_counter_ns()
            totala = asm_aug_assign_i(arg)
            print(f"\tassembly    returns = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

            start = perf_counter_ns()
            totalp = python_aug_assign_i(arg)
            print(f"\tpython      returns = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

            self.assertEqual(totala, totalp, f"Failed on input {arg}")

    def test_add_9_f(self):
        r=lambda:float(f"{random.randint(-100,100)}.{random.randint(0,10000)}")
        for args in [tuple(r() for _ in range(0,9)) for _ in range(0,50)]:

            start = perf_counter_ns()
            totala = asm_add_9_f(*args)
            print(f"\tassembly    {' + '.join(str(i) for i in args)} = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

            start = perf_counter_ns()
            totalp = python_add_9_f(*args)
            print(f"\tpython      {' + '.join(str(i) for i in args)} = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

            self.assertEqual(totala, totalp)

    def test_add_9_i(self):
        r=lambda:random.randint(-100,100)
        for args in [tuple(r() for _ in range(0,9)) for _ in range(0,50)]:

            start = perf_counter_ns()
            totala = asm_add_9_i(*args)
            print(f"\tassembly    {' + '.join(str(i) for i in args)} = {totala}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

            start = perf_counter_ns()
            totalp = python_add_9_i(*args)
            print(f"\tpython      {' + '.join(str(i) for i in args)} = {totalp}    {(perf_counter_ns()-start)/ 1e6:.4f}ms")

            self.assertEqual(totala, totalp)

if __name__ == '__main__':
    unittest.main(testRunner=TestAOT())