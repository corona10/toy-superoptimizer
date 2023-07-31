"""
Toy superoptimizer for minimum Python Bytecode Spec
"""

import copy
import dis
import enum
import itertools

from typing import Generator

import tqdm


class MemoryState:
    def __init__(self, stack, co_consts, co_varnames, ret):
        self.stack = stack
        self.co_consts = co_consts
        self.co_varnames = co_varnames
        self.ret = ret

    def __eq__(self, other) -> bool:
        if type(other) != type(self):
            return False
        if self.stack != other.stack:
            return False
        if self.co_consts != other.co_consts:
            return False
        if self.co_varnames != other.co_varnames:
            return False
        if self.ret != other.ret:
            return False
        return True

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"stack: {self.stack}, co_consts: {self.co_consts}, co_varnames: {self.co_varnames}, ret: {self.ret}"


# TODO: Autogen from opcode.opmap
class Opcode(enum.IntEnum):
    POP_TOP = (1,)
    RETURN_VALUE = (83,)
    UNPACK_SEQUENCE = (92,)
    SWAP = (99,)
    LOAD_CONST = (100,)
    COMPARE_OP = (107,)
    POP_JUMP_FORWARD_IF_FALSE = (114,)
    BINARY_OP = (122,)
    LOAD_FAST = (124,)
    STORE_FAST = (125,)
    RESUME = (151,)
    POP_JUMP_BACKWARD_IF_TRUE = (176,)


class Instruction:
    def __init__(self, op_code: Opcode, arg: int | None):
        self.op_code = op_code
        self.arg = arg

    def cost(self) -> int:
        if self.arg is None:
            return 0
        return 1

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"({self.op_code.name}, {self.arg})"

    def __eq__(self, other) -> bool:
        if type(self) != type(other):
            return False
        return self.op_code == other.op_code and self.arg == other.arg

    @staticmethod
    def ops():
        return (
            Opcode.POP_TOP,
            Opcode.RETURN_VALUE,
            Opcode.UNPACK_SEQUENCE,
            Opcode.SWAP,
            Opcode.LOAD_CONST,
            Opcode.COMPARE_OP,
            Opcode.POP_JUMP_FORWARD_IF_FALSE,
            Opcode.BINARY_OP,
            Opcode.LOAD_FAST,
            Opcode.STORE_FAST,
            Opcode.POP_JUMP_BACKWARD_IF_TRUE,
        )


class Program:
    def __init__(self, instructions: list[Instruction], co_consts, co_varnames):
        self.instructions = []
        for inst in instructions:
            if inst.op_code == Opcode.RESUME:
                # Filter out RESUME
                continue
            self.instructions.append(inst)
        self.co_consts = co_consts  # [c for c in f.__code__.co_consts]
        self.co_varnames = co_varnames  # [c for c in f.__code__.co_varnames]

    @classmethod
    def from_function(cls, f):
        instructions = []
        for inst in dis.Bytecode(f):
            if inst.opname == Opcode.RESUME:
                # Filter out RESUME
                continue
            opcode = Opcode(inst.opcode)
            instruction = Instruction(opcode, inst.arg)
            instructions.append(instruction)
        return cls(
            instructions,
            copy.deepcopy(f.__code__.co_consts),
            copy.deepcopy(f.__code__.co_varnames),
        )

    def cost(self) -> int:
        ret = 0
        for inst in self.instructions:
            ret += inst.cost()
        return ret

    def __len__(self) -> int:
        return len(self.instructions)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return str(self.instructions)


class VM:
    def __init__(self, program: Program):
        self.intructions = [inst for inst in program.instructions]
        self.stack = []
        self.co_consts = [c for c in program.co_consts]
        self.co_varnames = [c for c in program.co_varnames]
        self.ret = None
        self.pc = 0

    def _compare_op(self, left, right, op):
        match op:
            case 0:
                # Py_LT
                return left < right
            case 1:
                # Py_LE
                return left <= right
            case 2:
                # Py_EQ
                return left == right
            case 3:
                # Py_NE
                return left != right
            case 4:
                # Py_GT
                return left > right
            case 5:
                # Py_GE
                return left >= right
            case _:
                raise RuntimeError(f"Unsupported compare op: {op}")

    def _binary_op(self, left, right, op):
        match op:
            case 0:
                # NB_ADD
                return left + right
            case 1:
                # NB_AND
                return left & right
            case 2:
                # NB_FLOOR_DIVIDE
                return left // right
            case 3:
                # NB_LSHIFT
                return left << right
            case 4:
                # NB_MATRIX_MULTIPLY
                return left @ right
            case 5:
                # NB_MULTIPLY
                return left * right
            case 6:
                # NB_REMAINDER
                return left % right
            case 7:
                # NB_OR
                return left | right
            case 8:
                # NB_POWER
                return left**right
            case 9:
                # NB_RSHIFT
                return left >> right
            case 10:
                # NB_SUBTRACT
                return left - right
            case 11:
                # NB_TRUE_DIVIDE
                return left / right
            case 12:
                # NB_XOR
                return left ^ right
            case 13:
                # NB_INPLACE_ADD
                left += right
                return left
            case 14:
                # NB_INPLACE_AND
                left &= right
                return left
            case 15:
                # NB_INPLACE_FLOOR_DIVIDE
                left //= right
                return left
            case 16:
                # NB_INPLACE_LSHIFT
                left <<= right
                return left
            case 17:
                # NB_INPLACE_MATRIX_MULTIPLY
                left @= right
                return left
            case 18:
                # NB_INPLACE_MULTIPLY
                left *= right
                return left
            case 19:
                # NB_INPLACE_REMAINDER
                left %= right
                return left
            case 20:
                # NB_INPLACE_OR
                left |= right
                return left
            case 21:
                # NB_INPLACE_POWER
                left **= right
                return left
            case 22:
                # NB_INPLACE_RSHIFT
                left <<= right
                return left
            case 23:
                # NB_INPLACE_SUBTRACT
                left -= right
                return left
            case 24:
                # NB_INPLACE_TRUE_DIVIDE
                left /= right
                return left
            case 25:
                # NB_INPLACE_XOR
                left ^= right
                return left
            case _:
                raise RuntimeError(f"Unsupported binary op: {op}")

    def _dispatch(self):
        self.pc += 1

    def run(self) -> MemoryState:
        while True:
            inst = self.intructions[self.pc]
            match inst.op_code:
                case Opcode.RESUME:
                    self._dispatch()
                case Opcode.LOAD_CONST:
                    # https://docs.python.org/3.11/library/dis.html#opcode-LOAD_CONST
                    consti = inst.arg
                    self.stack.append(self.co_consts[consti])
                    self._dispatch()
                case Opcode.COMPARE_OP:
                    # https://docs.python.org/3.11/library/dis.html#opcode-COMPARE_OP
                    oparg = inst.arg
                    right = self.stack.pop()
                    left = self.stack[-1]
                    res = self._compare_op(left, right, oparg)
                    self.stack[-1] = res
                    self._dispatch()
                case Opcode.BINARY_OP:
                    # https://docs.python.org/3.11/library/dis.html#opcode-BINARY_OP
                    rhs = self.stack.pop()
                    lhs = self.stack[-1]
                    assert 0 <= oparg
                    res = self._binary_op(lhs, rhs, oparg)
                    self.stack[-1] = res
                    self._dispatch()
                case Opcode.UNPACK_SEQUENCE:
                    # https://docs.python.org/3.11/library/dis.html#opcode-UNPACK_SEQUENCE
                    count = inst.arg
                    assert len(self.stack[-1]) == count
                    self.stack.extend(self.stack.pop()[: -count - 1 : -1])
                    self._dispatch()
                case Opcode.STORE_FAST:
                    # https://docs.python.org/3.11/library/dis.html#opcode-STORE_FAST
                    var_num = inst.arg
                    top = self.stack.pop()
                    self.co_varnames[var_num] = top
                    self._dispatch()
                case Opcode.LOAD_FAST:
                    # https://docs.python.org/3.11/library/dis.html#opcode-LOAD_FAST
                    var_num = inst.arg
                    self.stack.append(self.co_varnames[var_num])
                    self._dispatch()
                case Opcode.SWAP:
                    # https://docs.python.org/3.11/library/dis.html#opcode-SWAP
                    i = inst.arg
                    self.stack[-i], self.stack[-1] = self.stack[-1], self.stack[-i]
                    self._dispatch()
                case Opcode.POP_TOP:
                    # https://docs.python.org/3.11/library/dis.html#opcode-POP_TOP
                    self.stack.pop()
                    self._dispatch()
                case Opcode.RETURN_VALUE:
                    # https://docs.python.org/3.11/library/dis.html#opcode-RETURN_VALUE
                    self.ret = self.stack[-1]
                    break
                case Opcode.POP_JUMP_FORWARD_IF_FALSE:
                    # https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_FORWARD_IF_FALSE
                    oparg = inst.arg
                    cond = self.stack.pop()
                    if not cond:
                        self.pc += oparg
                    self._dispatch()
                case Opcode.POP_JUMP_BACKWARD_IF_TRUE:
                    # # https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_BACKWARD_IF_TRUE
                    oparg = inst.arg
                    cond = self.stack.pop()
                    if cond:
                        self.pc -= oparg
                    self._dispatch()
                case _:
                    raise RuntimeError(f"Unsupported opcodes: {inst.opname}")
        memory_state = MemoryState(
            self.stack, self.co_consts, self.co_varnames, self.ret
        )
        return memory_state


class Superoptimizer:
    def __init__(self, program: Program):
        self.program = program

    def generate_programs(self) -> Generator[Program, None, None]:
        for length in range(1, len(self.program) + 1):
            for instructions in itertools.product(Instruction.ops(), repeat=length):
                arg_sets = []
                for inst in instructions:
                    match inst:
                        case Opcode.LOAD_CONST:
                            arg_sets.append(
                                [
                                    tuple([val])
                                    for val in range(len(self.program.co_consts))
                                ]
                            )
                        case Opcode.COMPARE_OP:
                            arg_sets.append([tuple([val]) for val in range(7)])
                        case Opcode.BINARY_OP:
                            arg_sets.append([tuple([val]) for val in range(26)])
                        case Opcode.UNPACK_SEQUENCE:
                            arg_sets.append(
                                [tuple([val]) for val in range(len(instructions))]
                            )
                        case Opcode.STORE_FAST:
                            arg_sets.append(
                                [
                                    tuple([val])
                                    for val in range(len(self.program.co_varnames))
                                ]
                            )
                        case Opcode.LOAD_FAST:
                            arg_sets.append(
                                [
                                    tuple([val])
                                    for val in range(len(self.program.co_varnames))
                                ]
                            )
                        case Opcode.SWAP:
                            arg_sets.append(
                                [tuple([val]) for val in range(len(instructions))]
                            )
                        case Opcode.POP_TOP:
                            arg_sets.append([(None,)])
                        case Opcode.RETURN_VALUE:
                            arg_sets.append([(None,)])
                        case Opcode.POP_JUMP_FORWARD_IF_FALSE:
                            arg_sets.append(
                                [tuple([val]) for val in range(len(instructions))]
                            )
                        case Opcode.POP_JUMP_BACKWARD_IF_TRUE:
                            arg_sets.append(
                                [tuple([val]) for val in range(len(instructions))]
                            )

                for arg_set in itertools.product(*arg_sets):
                    virtual_instructions = []
                    for inst, args in zip(instructions, arg_set):
                        virtual_instruction = Instruction(inst, *args)
                        virtual_instructions.append(virtual_instruction)

                    generated_program = Program(
                        virtual_instructions,
                        copy.deepcopy(self.program.co_consts),
                        copy.deepcopy(self.program.co_varnames),
                    )
                    yield generated_program

    def search(self) -> Program:
        origin_vm = VM(self.program)
        origin_state = origin_vm.run()
        print("========================================================")
        print(f"Original State: {origin_state} / Program: {self.program}")
        optimized_prog, optimized_state = self.program, None
        for prog in tqdm.tqdm(self.generate_programs()):
            if len(optimized_prog) < len(prog):
                break
            vm = VM(prog)
            try:
                possible_state = vm.run()
                if (
                    possible_state == origin_state
                    and len(prog) <= len(optimized_prog)
                    and prog.cost() < optimized_prog.cost()
                ):
                    optimized_prog, optimized_state = prog, possible_state
                    break
            except Exception:
                continue

        if optimized_state is not None:
            print(
                f"Found -> Optimized State: {optimized_state}, / Program: {optimized_prog}"
            )
        else:
            print("Not found!")
        print("========================================================")
        return optimized_prog


if __name__ == "__main__":

    def f():
        a, b = 3, 5
        a, b = b, a
        return a

    program = Program.from_function(f)
    optimizer = Superoptimizer(program)
    optimizer.search()

    def g():
        a, a = 3, 5
        return a

    program = Program.from_function(g)
    optimizer = Superoptimizer(program)
    optimizer.search()
