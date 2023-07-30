"""
Toy superoptimizer for minimum Python Bytecode Spec
"""

import copy
import dis
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


class VirtualInstruction:
    def __init__(self, op_name: str, arg: int | None):
        self.opname = op_name
        self.arg = arg

    def cost(self) -> int:
        if self.arg is None:
            return 0
        return 1

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'({self.opname}, {self.arg})'

    @staticmethod
    def ops():
        return ("LOAD_CONST", "UNPACK_SEQUENCE", "STORE_FAST", "LOAD_FAST", "SWAP", "POP_TOP", "RETURN_VALUE")


class Program:
    def __init__(self, instructions: list[VirtualInstruction], co_consts, co_varnames):
        self.instructions = []
        for inst in instructions:
            if inst.opname == "RESUME":
                # Filter out RESUME
                continue
            self.instructions.append(inst)
        self.co_consts = co_consts # [c for c in f.__code__.co_consts]
        self.co_varnames = co_varnames # [c for c in f.__code__.co_varnames]

    @classmethod
    def from_function(cls, f):
        instructions = []
        for inst in dis.Bytecode(f):
            if inst.opname == "RESUME":
                # Filter out RESUME
                continue
            instruction = VirtualInstruction(inst.opname, inst.arg)
            instructions.append(instruction)
        return cls(instructions, copy.deepcopy(f.__code__.co_consts), copy.deepcopy(f.__code__.co_varnames))

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

    def run(self) -> MemoryState:
        while True:
            inst = self.intructions[self.pc]
            match inst.opname:
                case "RESUME":
                    self.pc+=1
                case "LOAD_CONST":
                    # https://docs.python.org/3.11/library/dis.html#opcode-LOAD_CONST
                    consti = inst.arg
                    self.stack.append(self.co_consts[consti])
                    self.pc+=1
                case "UNPACK_SEQUENCE":
                    # https://docs.python.org/3.13/library/dis.html#opcode-UNPACK_SEQUENCE
                    count = inst.arg
                    assert(len(self.stack[-1]) == count)
                    self.stack.extend(self.stack.pop()[:-count-1:-1])
                    self.pc+=1
                case "STORE_FAST":
                    # https://docs.python.org/3.11/library/dis.html#opcode-STORE_FAST
                    var_num = inst.arg
                    top = self.stack.pop()
                    self.co_varnames[var_num] = top
                    self.pc+=1
                case "LOAD_FAST":
                    # https://docs.python.org/3.11/library/dis.html#opcode-LOAD_FAST
                    var_num = inst.arg
                    self.stack.append(self.co_varnames[var_num])
                    self.pc+=1
                case "SWAP":
                    # https://docs.python.org/3.11/library/dis.html#opcode-SWAP
                    i = inst.arg
                    self.stack[-i], self.stack[-1] = self.stack[-1], self.stack[-i]
                    self.pc+=1
                case "POP_TOP":
                    self.stack.pop()
                    self.pc+=1
                case "RETURN_VALUE":
                    self.ret = self.stack[-1]
                    break
                case _:
                    raise RuntimeError(f'Unsupported opcodes: {inst.opname}')
        memory_state = MemoryState(self.stack, self.co_consts, self.co_varnames, self.ret)
        return memory_state


class Superoptimizer:

    def __init__(self, program: Program):
        self.program = program

    def generate_programs(self) -> Generator[Program, None, None]:
        for length in range(1, len(self.program) + 1):
            for instructions in itertools.product(VirtualInstruction.ops(), repeat=length):
                arg_sets = []
                for inst in instructions:
                    match inst:
                        case "LOAD_CONST":
                            arg_sets.append([tuple([val]) for val in range(len(self.program.co_consts))])
                        case "UNPACK_SEQUENCE":
                            arg_sets.append([tuple([val]) for val in range(len(instructions))])
                        case "STORE_FAST":
                            arg_sets.append([tuple([val]) for val in range(len(self.program.co_varnames))])
                        case "LOAD_FAST":
                            arg_sets.append([tuple([val]) for val in range(len(self.program.co_varnames))])
                        case "SWAP":
                            arg_sets.append([tuple([val]) for val in range(len(instructions))])
                        case "POP_TOP":
                            arg_sets.append([(None,)])
                        case "RETURN_VALUE":
                            arg_sets.append([(None,)])

                for arg_set in itertools.product(*arg_sets):
                    virtual_instructions = []
                    for inst, args in zip(instructions, arg_set):
                        virtual_instruction = VirtualInstruction(inst, *args)
                        virtual_instructions.append(virtual_instruction)

                    generated_program = Program(virtual_instructions,
                                                copy.deepcopy(self.program.co_consts),
                                                copy.deepcopy(self.program.co_varnames))
                    yield generated_program

    def search(self) -> Program:
        origin_vm = VM(self.program)
        origin_state = origin_vm.run()
        print("========================================================")
        print(f"Original State: {origin_state} / Program: {self.program}")
        optimized_prog, optimized_state  = self.program, None
        for prog in tqdm.tqdm(self.generate_programs()):
            if len(optimized_prog) < len(prog):
                break
            vm = VM(prog)
            try:
                possible_state = vm.run()
                if possible_state == origin_state and len(prog) <= len(optimized_prog) and prog.cost() < optimized_prog.cost():
                    optimized_prog, optimized_state = prog, possible_state
                    break
            except Exception:
                continue

        if optimized_state is not None:
            print(f'Found -> Optimized State: {optimized_state}, / Program: {optimized_prog}')
        else:
            print('Not found!')
        print("========================================================")
        return optimized_prog


if __name__ == '__main__':
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
