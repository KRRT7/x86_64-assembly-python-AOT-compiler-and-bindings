from __future__ import annotations
from x86_64_assembly_bindings import Block, Instruction, MemorySize, OffsetRegister, Register

# type aliases
Reg = Register
Ins = Instruction

# registers
rsp = Reg("rsp")
rbp = Reg("rbp")


class Var:
    def __init__(self, stack_frame:StackFrame, name:str, size:MemorySize, py_type:type = int):
        self.name = name
        self.size = size
        self.type = py_type
        self.stack_frame = stack_frame

    def cast(self, lines:list[Instruction|Block], py_type:type = int) -> Register:
        if py_type == float:
            lines.push(Ins("cvtsi2sd", fpr:=Reg.request_float(lines=lines, offset=self.stack_frame.stack_offset), self.get()))
            return fpr
        elif py_type == int:
            lines.push(Ins("cvttsd2si", r:=Reg.request_64(lines=lines, offset=self.stack_frame.stack_offset), self.get()))
            return r

    def get(self) -> OffsetRegister:
        return self.stack_frame[self.name]

class StackFrame:
    def __init__(self):
        self.variables:list[Var] = []

    @property
    def size(self):
        return len(self.variables)
    
    @property
    def stack_offset(self):
        offset = 0
        for v in self.variables:
            offset += v.size.value//8
        return offset
    

    def alloca(self, name:str, size:MemorySize = MemorySize.QWORD, py_type:type = int) -> Instruction:
        self.variables.append(Var(self, name, size, py_type))
        return Ins("sub", rsp, size.value//8)

    def pop(self) -> Instruction|None:
        if self.stack_offset != 0:
            return Ins("add", rsp, self.stack_offset)
        return None

    def __contains__(self, key:str) -> bool:
        for v  in self.variables:
            if v.name == key:
                return True
        
        return False

    def __getitem__(self, key:str) -> OffsetRegister:
        offset = 0
        for v in self.variables:
            offset += v.size.value//8
            if v.name == key:
                return OffsetRegister(rbp, offset, True)
        raise KeyError(f"Variable \"{key}\" not found in stack frame.")

    def getvar(self, key:str) -> Var:
        for v in self.variables:
            if v.name == key:
                return v
        raise KeyError(f"Variable \"{key}\" not found in stack frame.")

class Stack:
    def __init__(self):
        self.stack = [StackFrame()]
        self.cursor = -1
        self.push()
        self.__origin = True

    def get_is_origin(self):
        "This returns true only on the first ever call."
        so = self.__origin
        self.__origin = False
        return so

    @property
    def current(self) -> StackFrame:
        return self.stack[self.cursor]

    def alloca(self, name:str, size:MemorySize = MemorySize.QWORD, py_type:type = int) -> Instruction:
        return self.current.alloca(name, size, py_type)

    def push(self):
        self.stack.append(StackFrame())
        self.cursor+=1

    def pop(self) -> Instruction|None:
        r = self.current.pop()
        self.cursor-=1
        return r

    def __contains__(self, key:str) -> bool:
        for frame in self.stack:
            if key in frame:return True
        return False

    def __getitem__(self, key:str) -> OffsetRegister:
        for frame in reversed(self.stack[0:self.cursor+1]):
            
            if key in frame:
                return frame[key]
        raise KeyError(f"Variable \"{key}\" not found in function stack.")

    def getvar(self, key:str) -> Var:
        for frame in reversed(self.stack[0:self.cursor+1]):
            
            if key in frame:
                return frame.getvar(key)
        raise KeyError(f"Variable \"{key}\" not found in function stack.")