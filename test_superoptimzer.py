import unittest

from superoptimizer import Superoptimizer, Program, Instruction, Opcode


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
            Instruction(Opcode.LOAD_CONST, 1),
            Instruction(Opcode.UNPACK_SEQUENCE, 2),
            Instruction(Opcode.STORE_FAST, 1),
            Instruction(Opcode.STORE_FAST, 0),
            Instruction(Opcode.LOAD_FAST, 0),
            Instruction(Opcode.RETURN_VALUE, None),
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
            Instruction(Opcode.LOAD_CONST, 1),
            Instruction(Opcode.UNPACK_SEQUENCE, 2),
            Instruction(Opcode.POP_TOP, None),
            Instruction(Opcode.STORE_FAST, 0),
            Instruction(Opcode.LOAD_FAST, 0),
            Instruction(Opcode.RETURN_VALUE, None),
        ]
        self.assertEqual(ret.instructions, expected)
