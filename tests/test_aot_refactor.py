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

class TestAOT(unittest.TestCase):
    
    def setUp(self):
        print(f"\nRunning {self._testMethodName}:")

    def test_assign(self):
        asm_assign(5)

if __name__ == '__main__':
    unittest.main(testRunner=TestAOT())