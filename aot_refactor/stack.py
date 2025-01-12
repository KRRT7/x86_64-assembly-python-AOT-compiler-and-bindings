from __future__ import annotations
from collections import OrderedDict
from aot_refactor.type_imports import *
from aot_refactor.variable import Variable

# >> TODO : Create a separate stack for keeping track of block scope << #

    # Maybe rename Stack/StackFrame to VariableStack/VariableStackFrame

class StackFrame:
    def __init__(self):
        self.frame_size = 0
        self.variables: OrderedDict[str, Variable] = OrderedDict()

    def __contains__(self, key: str):
        return key in self.variables
    
    def __getitem__(self, key: str) -> Variable:
        return self.variables[key]
    
    def allocate(self, name: str, python_type: type, size: MemorySize = MemorySize.QWORD) -> LinesType:
        self.frame_size += size.value // 8
        self.variables[name] = Variable(name, python_type, OffsetRegister(Reg("rbp"), self.frame_size, True, meta_tags={python_type}), size)
        return [Ins("sub", Reg("rsp"), size.value // 8)]
    
    def allocate_variable(self, variable: Variable) -> LinesType:
        self.frame_size += variable.size.value // 8
        self.variables[variable.name] = variable
        return [Ins("sub", Reg("rsp"), variable.size.value // 8)]
    
    def free(self) -> LinesType | None:
        if self.frame_size != 0:
            return [Ins("add", Reg("rsp"), self.frame_size)]
        return None
    
class Stack:
    def __init__(self):
        self.stack: list[StackFrame] = [StackFrame()]

    @property
    def current(self) -> StackFrame:
        return self.stack[-1]
    
    def allocate(self, name: str, python_type: type, size: MemorySize = MemorySize.QWORD) -> LinesType:
        return self.current.allocate(name, python_type, size)
    
    def push(self):
        self.stack.append(StackFrame())

    def pop(self) -> LinesType | None:
        lines: LinesType | None = self.current.free()
        self.stack.pop()
        return lines
    
    def __contains__(self, key: str) -> bool:
        for frame in self.stack:
            if key in frame:
                return True
        return False
    
    def __getitem__(self, key: str) -> Variable:
        for frame in reversed(self.stack):
            if key in frame:
                return frame[key]
        raise KeyError(f'Variable "{key}" not found in Stack.')
    
