from __future__ import annotations
from collections import OrderedDict
from aot.type_imports import *
from aot.variable import Value, Variable

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
    
    def allocate(self, name: str, python_type: type, size:MemorySize|None = None) -> None:
        if python_type in {float, int}:
            size: MemorySize = MemorySize.QWORD
            self.frame_size += size.value // 8
            self.variables[name] = Variable(name, python_type, OffsetRegister(Reg("rbp"), self.frame_size, True, meta_tags={python_type, "variable"}), size)
        elif python_type is bool:
            size: MemorySize = MemorySize.BYTE
            self.frame_size += size.value // 8
            self.variables[name] = Variable(name, python_type, OffsetRegister(Reg("rbp"), self.frame_size, True, meta_tags={python_type, "variable"}), size)
        elif isinstance(python_type, Array):
            raise TypeError(f"Allocate Array type using allocate_value then allocate_from_value")
        else:
            raise TypeError(f"Failed to allocate type {python_type.__name__ if hasattr(python_type, "__name__") else type(python_type).__name__}")
    
    def allocate_from_value(self, name:str, value:Value):
        if value.python_type in {float, int}:
            self.variables[name] = Variable.from_value(name, value)
        elif value.python_type is bool:
            self.variables[name] = Variable.from_value(name, value)
        elif isinstance(value.python_type, Array):
            self.variables[name] = Variable.from_value(name, value)
        else:
            raise TypeError(f"Failed to allocate type {value.python_type.__name__ if hasattr(value.python_type, "__name__") else type(value.python_type).__name__}")
        
    def allocate_value(self, python_type: type, size:MemorySize|None = None) -> Value:

        get_register = lambda frame_size:OffsetRegister(
            Reg("rbp"),
            frame_size,
            True,
            meta_tags={python_type, "variable", "value"}
        )

        if python_type in {float, int}:
            size: MemorySize = MemorySize.QWORD
            self.frame_size += size.value // 8
            return Value(python_type, get_register(self.frame_size), size)
        elif python_type is bool:
            size: MemorySize = MemorySize.BYTE
            self.frame_size += size.value // 8
            return Value(python_type, get_register(self.frame_size), size)
        elif isinstance(python_type, Array):
            self.frame_size += python_type.size // 8
            return Value(python_type, get_register(self.frame_size), python_type.type_size)
        else:
            raise TypeError(f"Failed to allocate type {python_type.__name__ if hasattr(python_type, "__name__") else type(python_type).__name__}")
    
    def allocate_variable(self, variable: Variable) -> LinesType:
        self.frame_size += variable.size.value // 8
        self.variables[variable.name] = variable
    
    def free(self) -> LinesType:
        if self.frame_size != 0:
            return [Ins("add", Reg("rsp"), self.frame_size)]
        return []
    
class Stack:
    def __init__(self):
        self.stack: list[StackFrame] = [StackFrame()]

    @property
    def current(self) -> StackFrame:
        return self.stack[-1]
    
    def allocate(self, name: str, python_type: type, size:MemorySize|None = None) -> None:
        return self.current.allocate(name, python_type, size)
    
    def allocate_from_value(self, name:str, value:Value):
        return self.current.allocate_from_value(name, value)
    
    def allocate_value(self, python_type: type, size:MemorySize|None = None) -> Value:
        return self.current.allocate_value(python_type, size)
    
    def push(self):
        self.stack.append(StackFrame())

    def pop(self) -> None:
        self.stack.pop()
        return
    
    def free(self) -> LinesType | None:
        lines: LinesType | None = self.current.free()
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
    
