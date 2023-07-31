import unittest

from superoptimizer import Superoptimizer, Program, Instruction


class TestSuperOptimizer(unittest.TestCase):
    def test_swap(self):
        def f():
            a, b = 3, 5
            a, b = b, a
            return a

        program = Program.from_function(f)
        optimizer = Superoptimizer(program)
        ret = optimizer.search()
        expected = [
            Instruction("LOAD_CONST", 1),
            Instruction("UNPACK_SEQUENCE", 2),
            Instruction("STORE_FAST", 1),
            Instruction("STORE_FAST", 0),
            Instruction("LOAD_FAST", 0),
            Instruction("RETURN_VALUE", None),
        ]
        self.assertEqual(ret.instructions, expected)

    def test_redundant_STORE_FAST(self):
        def f():
            a, a = 1, 2
            return a

        program = Program.from_function(f)
        optimizer = Superoptimizer(program)
        ret = optimizer.search()
        expected = [
            Instruction("LOAD_CONST", 1),
            Instruction("UNPACK_SEQUENCE", 2),
            Instruction("POP_TOP", None),
            Instruction("STORE_FAST", 0),
            Instruction("LOAD_FAST", 0),
            Instruction("RETURN_VALUE", None),
        ]
        self.assertEqual(ret.instructions, expected)
