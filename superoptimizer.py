import dis
import itertools

import tqdm


class MemoryState:
    def __init__(self, stack, co_consts, co_varnames, ret):
        self.stack = stack
        self.co_consts = co_consts
        self.co_varnames = co_varnames
        self.ret = ret

    def __eq__(self, other):
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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"stack: {self.stack}, co_consts: {self.co_consts}, co_varnames: {self.co_varnames}, ret: {self.ret}"


class VirtualInstruction:
    def __init__(self, op_name, arg):
        self.opname = op_name
        self.arg = arg

    @staticmethod
    def ops():
        return ("RESUME", "LOAD_CONST", "UNPACK_SEQUENCE", "STORE_FAST", "LOAD_FAST", "SWAP", "POP_TOP", "RETURN_VALUE")


class Program:
    def __init__(self, f):
        self.instructions = []
        self.co_consts = [c for c in f.__code__.co_consts]
        self.co_varnames = [c for c in f.__code__.co_varnames]

        for inst in dis.Bytecode(f):
            instruction = VirtualInstruction(inst.opname, inst.arg)
            self.instructions.append(instruction)

    def __len__(self):
        return len(self.instructions)


class VM:
    def __init__(self, program):
        self.intructions = [inst for inst in program.instructions]
        self.stack = []
        self.co_consts = [c for c in program.co_consts]
        self.co_varnames = [c for c in program.co_varnames]
        self.ret = None

    def run(self) -> MemoryState:
        for inst in self.intructions:
            match inst.opname:
                case "RESUME":
                    pass
                case "LOAD_CONST":
                    # https://docs.python.org/3.13/library/dis.html#opcode-LOAD_CONST
                    consti = inst.arg
                    self.stack.append(self.co_consts[consti])
                case "UNPACK_SEQUENCE":
                    # https://docs.python.org/3.13/library/dis.html#opcode-UNPACK_SEQUENCE
                    count = inst.arg
                    self.stack.extend(self.stack.pop()[:-count-1:-1])
                case "STORE_FAST":
                    # https://docs.python.org/3.13/library/dis.html#opcode-STORE_FAST
                    var_num = inst.arg
                    top = self.stack.pop()
                    self.co_varnames[var_num] = top
                case "LOAD_FAST":
                    # https://docs.python.org/3.13/library/dis.html#opcode-LOAD_FAST
                    var_num = inst.arg
                    self.stack.append(self.co_varnames[var_num])
                case "SWAP":
                    # https://docs.python.org/3.13/library/dis.html#opcode-SWAP
                    i = inst.arg
                    self.stack[-i], self.stack[-1] = self.stack[-1], self.stack[-i]
                case "POP_TOP":
                    self.stack.pop()
                case "RETURN_VALUE":
                    self.ret = self.stack[-1]
                case _:
                    raise RuntimeError(f'Unsupported opcodes: {inst.opname}')
        memory_state = MemoryState(self.stack, self.co_consts, self.co_varnames, self.ret)
        return memory_state


class Superoptimizer:

    def __init__(self, program: Program):
        self.program = program

    def generate_programs(self):
        for length in tqdm.tqdm(range(1, len(self.program))):
            for prog in itertools.product(VirtualInstruction.ops(), repeat=length):
                pass

    def search(self):
        origin_vm = VM(self.program)
        origin_state = origin_vm.run()
        min_length = len(self.program)
        print(f"Original State: {origin_state}")
        """
        for prog in self.generate_programs(f):
            vm = VM(prog)
            possible_state = vm.run()
        """


if __name__ == '__main__':
    def f():
        a, a = 3, 5
        return a

    program = Program(f)
    optimizer = Superoptimizer(program)
    optimizer.search()
