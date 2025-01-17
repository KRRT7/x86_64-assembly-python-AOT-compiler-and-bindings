from __future__ import annotations
import copy
from dataclasses import dataclass
from enum import Enum, EnumMeta
from pathlib import Path
import subprocess
import sys
from typing import Any, Literal
import os
import ctypes
import platform

current_os = platform.system()

@dataclass
class PtrType:
    value:type

class MemorySize(Enum):
    BYTE = 8
    WORD = 16
    DWORD = 32
    QWORD = 64
    DQWORD = 128

    def __hash__(self):
        return hash(self.value)

    def to_ctype(self, signed: bool = False, py_type: type = int):
        if py_type is None:
            return None
        
        if isinstance(py_type, PtrType):
            return {
                int:  [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int64) ],
                float:[None,                            ctypes.POINTER(ctypes.c_double)],
                bool: [ctypes.POINTER(ctypes.c_bool),   ctypes.POINTER(ctypes.c_bool)  ]
            }[py_type.value][signed]

        return {
            self.BYTE: {
                int:[ctypes.c_ubyte, ctypes.c_byte],
                bool:[ctypes.c_bool,ctypes.c_bool]
            },
            self.WORD: {
                int:[ctypes.c_ushort, ctypes.c_short]
            },
            self.DWORD: {
                int:[ctypes.c_uint32, ctypes.c_int32],
                float:[None, ctypes.c_float]
            },
            self.QWORD: {
                int:[ctypes.c_uint64, ctypes.c_int64],
                float:[None, ctypes.c_double]
            },
            self.DQWORD: {
                int:[ctypes.c_uint64, ctypes.c_int64],
                float:[None, ctypes.c_double]
            },
        }[self][py_type][signed]

    def __eq__(self, other: MemorySize):
        if not isinstance(other, MemorySize):
            return False
        return self.value == other.value

    def __gt__(self, other: Register):
        if not isinstance(other, MemorySize):
            return False
        return self.value > other.value

    def __lt__(self, other: Register):
        if not isinstance(other, MemorySize):
            return False
        return self.value < other.value

    def __repr__(self):
        return f"{self.name}({self.value})"

    def __str__(self):
        return repr(self)

    @property
    def grow(self):
        match self:
            case self.BYTE:
                return self.WORD
            case self.WORD:
                return self.DWORD
            case self.DWORD:
                return self.QWORD
            case self.QWORD:
                return self.DQWORD
            case self.DQWORD:
                return None

    @property
    def shrink(self):
        match self:
            case self.BYTE:
                return None
            case self.WORD:
                return self.BYTE
            case self.DWORD:
                return self.WORD
            case self.QWORD:
                return self.DWORD
            case self.DQWORD:
                return self.QWORD

    @property
    def sec_data_write(self):
        match self:
            case self.BYTE:
                return "db"
            case self.WORD:
                return "dw"
            case self.DWORD:
                return "dd"
            case self.QWORD | self.DQWORD:
                return "dq"

    @property
    def sec_bss_write(self):
        match self:
            case self.BYTE:
                return "resb"
            case self.WORD:
                return "resw"
            case self.DWORD:
                return "resd"
            case self.QWORD | self.DQWORD:
                return "resq"

MemoryVariableValueType = dict[str, tuple[MemorySize | str, list | int]] | tuple[MemorySize, tuple[MemorySize | str, list | int]]
class Memory:
    def __init__(
        self,
        text_inclusions: list[str] | None = None,
        **memory: dict[str, tuple[MemorySize | str, list[Any] | int]],
    ):
        self.data = {}
        self.bss = {}
        self.variables:dict[str, StackVariable | MemoryVariableValueType] = {}
        self.text_inclusions = [] if text_inclusions is None else text_inclusions
        for label, val in memory.items():
            self[label] = val

    def __getitem__(self, key: str) -> StackVariable:
        return self.variables[key]
    
    def __setitem__(self, key: str, value: StackVariable | MemoryVariableValueType) -> StackVariable:
            val_new = (
                value if isinstance(value[0], MemorySize) else (MemorySize[value[0]], value[1])
            )
            if isinstance(val_new[1], int):
                self.bss[key] = val_new
            elif isinstance(val_new[1], list):
                self.data[key] = val_new

            self.variables[key] = StackVariable(key, *val_new)
    
    def __contains__(self, key:str):
        return key in self.variables

    def __str__(self):
        return (
            ("section .data\n    " if self.data else "")
            + "\n    ".join(
                f"{label} {size.sec_data_write} " + ", ".join(str(a) for a in arguments)
                for label, (size, arguments) in self.data.items()
            )
            + "\n"
            + ("section .bss\n    " if self.bss else "")
            + "\n    ".join(
                f"{label} {size.sec_bss_write} {arguments}"
                for label, (size, arguments) in self.bss.items()
            )
            + "\nsection .text\n    "
            + "\n    ".join(self.text_inclusions)
        )

    def write(self) -> str:
        return str(self)

    def __call__(self, recorder: Program | None = None):
        (recorder if recorder else Program.CURRENT).append(self)
        return self

class Program:
    CURRENT: Program | None = None

    @property
    def current_function(self) -> Function:
        return self.function_stack[-1]

    def __init__(self, name: str | None = None):
        self.name = name
        self.memory = Memory()
        self.lines: list[Instruction | Memory | str] = [self.memory]
        self.functions: dict[str, Function] = {}
        self.__ctypes_lib: ctypes.CDLL | None = None
        Program.CURRENT = self
        self.compiled = False
        self.linked = False
        self.function_stack: list[Function] = []

    def append(self, component: Instruction | Memory):
        self.lines.append(component)

    def write(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return (
            "\n".join(f"global {fun}" for fun in self.functions)
            + "\n"
            + "\n".join(
                i
                if isinstance(i, str)
                else f"{'    ' if isinstance(i, Instruction) else ''}{i.write()}"
                for i in self.lines
            )
        )

    def save(self, path: str):
        path:Path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir()
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(self.write())

    def comment(self, text: str):
        self.append(f"; {text}")

    def append_line(self, line: str):
        self.append(line)

    def new_line(self):
        self.append("")

    def exit_program(self):
        self.append("    mov rax, 60")
        self.append("    mov rdi, 0")
        self.append("    syscall")

    def compile(
        self,
        program: str | None = None,
        save: bool = True,
        arguments_: dict[str, any]|None = None,
        flags:list[str]|None = None
    ):
        program = self.name if program is None else program
        arguments_ = arguments_ if arguments_ else {}
        flags = flags if flags else []
        if program is None:
            raise RuntimeError(
                'You must specify a program name either in the "program" argument of the "compile" function, by setting the "name" attribute of your "Program" instance or by specifying it as the "name" argument when creating your "Program" instance.'
            )

        if save:
            self.save(f"{program}.asm")
        args = {
            "-f": "elf64" if current_os == "Linux" else "win64",
            "-o": f'"{program}.o"',
        }
        args.update({f"-{k}": (str(v) if v is not None else None) for k, v in arguments_.items()})
        flags = [f"-{fl}" for fl in flags]
        command = (
            "yasm "
            + " ".join(flags) + " "
            + " ".join([(f"{k} {v}" if v is not None else k) for k, v in args.items()])
            + f' "{program}.asm"'
        )

        os.system(command)
        self.compiled = True

    def link(
        self,
        output: str | None = None,
        programs: set[Program | str] | None = None,
        args: dict[str, any | None] | None = None,
        lib_paths: set[str] | None = None,
        libs: set[str] | None = None,
        script: str | None = None,
        output_extension: str = "",
    ):
        output = self.name if output is None else output

        programs = set() if programs is None else programs
        if self.name is not None:
            programs.add(self.name)

        if not programs:
            raise RuntimeError(
                'The "programs" argument cannot be empty unless the "Program" instance\'s "name" attribute is set.'
            )

        if output is None:
            raise RuntimeError(
                'You must specify a program name either by passing it as the "output" argument when calling the link function, by setting the "name" attribute of your "Program" instance or by specifying it as the "name" argument when creating your "Program" instance.'
            )

        output = f"{output}{output_extension}"

        out_file = f'-o "{output}"'
        o_files = '"' + ' "'.join([f'{f}.o"' for f in programs])
        script = "" if script is None else f'-T "{script}"'
        lib_paths = (
            ""
            if lib_paths is None
            else '"-L' + ' "-L'.join([f'{p}"' for p in lib_paths])
        )
        libs = "" if libs is None else '"-l' + ' "-l'.join([f'{l}"' for l in libs])
        args: str = (
            ""
            if args is None
            else " ".join(
                [f"-{k}" + ("" if v is None else f' "{v}"') for k, v in args.items()]
            )
        )

        command = f"ld {args} {out_file} {script} {o_files} {lib_paths} {libs}"

        os.system(command)

        self.linked = True

    def call(
        self, function_name: str, *arguments: list[any], library: str | None = None, force_lookup: bool = False
    ) -> any:
        library = library if library else self.name
        if not library:
            raise RuntimeError(
                'Either the "library" argument of the call function or "Program" instance\'s "name" attribute need to be specified to call an assembled function from python.'
            )
        library = f"./{library}.so" if current_os == "Linux" else f"./{library}.dll"
        if force_lookup or not self.__ctypes_lib or self.__ctypes_lib._name != library:
            self.__ctypes_lib = ctypes.CDLL(library)
        func: Function = self.functions[function_name]
        cfunc: ctypes._NamedFuncPointer = self.__ctypes_lib[function_name]
        cfunc.argtypes = func.ctypes_arguments
        cfunc.restype = func.ctypes_restype

        return cfunc(*arguments)

    def run(
        self,
        *args: list[any],
        compile_args: dict | None = None,
        link_args: dict | None = None,
        skip_compile: bool = False,
        skip_link: bool = False,
    ):
        if self.name is None:
            raise RuntimeError(
                'The "name" attribute of the "Program" instance must be specified to run the program.'
            )

        compile_args = {} if compile_args is None else compile_args
        link_args = {} if link_args is None else link_args

        if not skip_compile:
            self.compile(**compile_args)
        if not skip_link:
            self.link(**link_args)

        args: str = " ".join([f"'{a}'" for a in args])

        os.system(f"./{self.name} {args}")


Program()  # create the current program


class Block:
    block_counter = 0

    def __init__(self, label: str | None = None, prefix: str = "", suffix: str = ""):
        self.label: str = label if label else f"block{Block.block_counter}"
        self.prefix: str = prefix
        self.suffix: str = suffix
        if label is None:
            Block.block_counter += 1

    @property
    def name(self) -> str:
        return self.label

    def __str__(self):
        return f"{self.prefix}{self.label}{self.suffix}"

    def write(self):
        return f"{self}:"

    def __call__(self, recorder: Program | None = None):
        (recorder if recorder else Program.CURRENT).append(self)
        return self





RDT = RegisterDataType = tuple[str, MemorySize, Literal[0] | Literal[1]]
#                       [name, size, position (0 = upper bytes, 1 = lower bytes)]

MemS = MemorySize


class RegisterData(Enum):
    """
    This enum defines all the sizes and other shared properties of all registers.
    """

    # main registers
    ah: RegisterDataType = ("ah", MemorySize.BYTE, 0)
    al: RegisterDataType = ("al", MemorySize.BYTE, 1)
    dx: RegisterDataType = ("dx", MemorySize.WORD, 0)
    ax: RegisterDataType = ("ax", MemorySize.WORD, 1)
    edx: RegisterDataType = ("edx", MemorySize.DWORD, 0)
    eax: RegisterDataType = ("eax", MemorySize.DWORD, 1)
    rdx: RegisterDataType = ("rdx", MemorySize.QWORD, 0)
    rax: RegisterDataType = ("rax", MemorySize.QWORD, 1)

    rcx: RegisterDataType = ("rcx", MemorySize.QWORD, 0)
    ecx: RegisterDataType = ("ecx", MemorySize.DWORD, 0)
    cx: RegisterDataType = ("cx", MemorySize.WORD, 0)
    ch: RegisterDataType = ("ch", MemorySize.BYTE, 0)
    cl: RegisterDataType = ("cl", MemorySize.BYTE, 1)

    dh: RegisterDataType = ("dh", MemorySize.BYTE, 0)
    dl: RegisterDataType = ("dl", MemorySize.BYTE, 1)

    rbx: RegisterDataType = ("rbx", MemorySize.QWORD, 0)
    ebx: RegisterDataType = ("ebx", MemorySize.DWORD, 0)
    bx: RegisterDataType = ("bx", MemorySize.WORD, 0)
    bh: RegisterDataType = ("bh", MemorySize.BYTE, 0)
    bl: RegisterDataType = ("bl", MemorySize.BYTE, 1)

    rsp: RegisterDataType = ("rsp", MemorySize.QWORD, 0)
    esp: RegisterDataType = ("esp", MemorySize.DWORD, 0)
    sp: RegisterDataType = ("sp", MemorySize.WORD, 0)
    spl: RegisterDataType = ("spl", MemorySize.BYTE, 1)

    rbp: RegisterDataType = ("rbp", MemorySize.QWORD, 0)
    ebp: RegisterDataType = ("ebp", MemorySize.DWORD, 0)
    bp: RegisterDataType = ("bp", MemorySize.WORD, 0)
    bpl: RegisterDataType = ("bpl", MemorySize.BYTE, 1)

    # other 64
    rdi: RegisterDataType = ("rdi", MemorySize.QWORD, 0)
    rsi: RegisterDataType = ("rsi", MemorySize.QWORD, 0)
    r8: RegisterDataType = ("r8", MemorySize.QWORD, 0)
    r9: RegisterDataType = ("r9", MemorySize.QWORD, 0)
    r10: RegisterDataType = ("r10", MemorySize.QWORD, 0)
    r11: RegisterDataType = ("r11", MemorySize.QWORD, 0)

    # other 32
    esi: RegisterDataType = ("esi", MemorySize.DWORD, 0)
    edi: RegisterDataType = ("edi", MemorySize.DWORD, 0)
    r8d: RegisterDataType = ("r8d", MemorySize.DWORD, 0)
    r9d: RegisterDataType = ("r9d", MemorySize.DWORD, 0)
    r10d: RegisterDataType = ("r10d", MemorySize.DWORD, 0)
    r11d: RegisterDataType = ("r11d", MemorySize.DWORD, 0)

    # other 16
    si: RegisterDataType = ("si", MemorySize.WORD, 0)
    di: RegisterDataType = ("di", MemorySize.WORD, 0)
    r8w: RegisterDataType = ("r8w", MemorySize.WORD, 0)
    r9w: RegisterDataType = ("r9w", MemorySize.WORD, 0)
    r10w: RegisterDataType = ("r10w", MemorySize.WORD, 0)
    r11w: RegisterDataType = ("r11w", MemorySize.WORD, 0)

    # other 8
    sil: RegisterDataType = ("sil", MemorySize.BYTE, 1)
    dil: RegisterDataType = ("dil", MemorySize.BYTE, 1)
    r8b: RegisterDataType = ("r8b", MemorySize.BYTE, 0)
    r9b: RegisterDataType = ("r9b", MemorySize.BYTE, 0)
    r10b: RegisterDataType = ("r10b", MemorySize.BYTE, 0)
    r11b: RegisterDataType = ("r11b", MemorySize.BYTE, 0)

    r12: RegisterDataType = ("r12", MemS.QWORD, 0)
    r12d: RegisterDataType = ("r12d", MemS.DWORD, 0)
    r12w: RegisterDataType = ("r12w", MemS.WORD, 0)
    r12b: RegisterDataType = ("r12b", MemS.BYTE, 0)

    r13: RegisterDataType = ("r13", MemS.QWORD, 0)
    r13d: RegisterDataType = ("r13d", MemS.DWORD, 0)
    r13w: RegisterDataType = ("r13w", MemS.WORD, 0)
    r13b: RegisterDataType = ("r13b", MemS.BYTE, 0)

    r14: RegisterDataType = ("r14", MemS.QWORD, 0)
    r14d: RegisterDataType = ("r14d", MemS.DWORD, 0)
    r14w: RegisterDataType = ("r14w", MemS.WORD, 0)
    r14b: RegisterDataType = ("r14b", MemS.BYTE, 0)

    r15: RegisterDataType = ("r15", MemS.QWORD, 0)
    r15d: RegisterDataType = ("r15d", MemS.DWORD, 0)
    r15w: RegisterDataType = ("r15w", MemS.WORD, 0)
    r15b: RegisterDataType = ("r15b", MemS.BYTE, 0)

    xmm0: RDT = ("xmm0", MemS.DQWORD, 0)
    xmm1: RDT = ("xmm1", MemS.DQWORD, 0)
    xmm2: RDT = ("xmm2", MemS.DQWORD, 0)
    xmm3: RDT = ("xmm3", MemS.DQWORD, 0)
    xmm4: RDT = ("xmm4", MemS.DQWORD, 0)
    xmm5: RDT = ("xmm5", MemS.DQWORD, 0)
    xmm6: RDT = ("xmm6", MemS.DQWORD, 0)
    xmm7: RDT = ("xmm7", MemS.DQWORD, 0)
    xmm8: RDT = ("xmm8", MemS.DQWORD, 0)
    xmm9: RDT = ("xmm9", MemS.DQWORD, 0)
    xmm10: RDT = ("xmm10", MemS.DQWORD, 0)
    xmm11: RDT = ("xmm11", MemS.DQWORD, 0)
    xmm12: RDT = ("xmm12", MemS.DQWORD, 0)
    xmm13: RDT = ("xmm13", MemS.DQWORD, 0)
    xmm14: RDT = ("xmm14", MemS.DQWORD, 0)
    xmm15: RDT = ("xmm15", MemS.DQWORD, 0)

    @classmethod
    def from_size(cls, size: MemorySize) -> tuple[RegisterData, RegisterData]:
        RD = RegisterData
        match size:
            case MemorySize.BYTE:
                return RD.ah, RD.al
            case MemorySize.WORD:
                return RD.dx, RD.ax
            case MemorySize.DWORD:
                return RD.edx, RD.eax
            case MemorySize.QWORD:
                return RD.rdx, RD.rax

    def cast_to(self, size: MemorySize) -> RegisterData:
        RD = RegisterData
        rname = self.name[:3]
        match size:
            case MemorySize.BYTE:
                return RegD[f"{rname}b"]
            case MemorySize.WORD:
                return RegD[f"{rname}w"]
            case MemorySize.DWORD:
                return RegD[f"{rname}d"]
            case MemorySize.QWORD:
                return RegD[f"{rname}"]

    @property
    def is_callee_saved(self) -> bool:
        return bool(self.get_callee_saved())

    def get_callee_saved(self) -> RegisterData:
        if current_os == "Linux":
            match self:
                case self.rbx:
                    return self.rbx
                case self.r12 | self.r12d | self.r12w | self.r12b:
                    return self.r12
                case self.r13 | self.r13d | self.r13w | self.r13b:
                    return self.r13
                case self.r14 | self.r14d | self.r14w | self.r14b:
                    return self.r14
                case self.r15 | self.r15d | self.r15w | self.r15b:
                    return self.r15
                case _:
                    return None
        else:
            match self:
                case self.r12 | self.r12d | self.r12w | self.r12b:
                    return self.r12
                case self.r13 | self.r13d | self.r13w | self.r13b:
                    return self.r13
                case self.r14 | self.r14d | self.r14w | self.r14b:
                    return self.r14
                case self.r15 | self.r15d | self.r15w | self.r15b:
                    return self.r15
                case (
                    self.xmm6
                    | self.xmm7
                    | self.xmm8
                    | self.xmm9
                    | self.xmm10
                    | self.xmm11
                    | self.xmm12
                    | self.xmm13
                    | self.xmm14
                    | self.xmm15
                ):
                    return self
                case _:
                    return None

    @property
    def register_name(self) -> str:
        return self.value[0]

    @property
    def size(self) -> MemorySize:
        return self.value[1]

    @property
    def position(self) -> Literal[0] | Literal[1]:
        """
        return 0 if upper bytes, return 1 if lower bytes
        """
        return self.value[2]


RegD = RegisterData

get_scratch_reg_list = (
    lambda s: [RegisterData[f"r{r}{s}"] for r in reversed(range(10, 16))]
    if current_os == "Linux"
    else [RegisterData[f"r{r}{s}"] for r in reversed(range(12, 16))]
)


class Register:
    data: RegisterData

    available_64: list[RegisterData] = get_scratch_reg_list("")
    available_32: list[RegisterData] = get_scratch_reg_list("d")
    available_16: list[RegisterData] = get_scratch_reg_list("w")
    available_8: list[RegisterData] = get_scratch_reg_list("b")

    available_float: list[RegisterData] = (
        [RegisterData[f"xmm{n}"] for n in reversed(range(8, 16))]
        if current_os == "Linux"
        else [RegisterData[f"xmm{n}"] for n in reversed(range(4, 16))]
    )

    all_scratch_registers: set[RegisterData] = set(
        get_scratch_reg_list("")
        + get_scratch_reg_list("d")
        + get_scratch_reg_list("w")
        + get_scratch_reg_list("b")
        + (
            [RegisterData[f"xmm{n}"] for n in reversed(range(8, 16))]
            if current_os == "Linux"
            else [RegisterData[f"xmm{n}"] for n in reversed(range(4, 16))]
        )
    )

    stack_memory_in_use: int = 0

    def __init__(self, register: str | RegisterData, meta_tags: set | None = None):
        self.data = RegisterData[register] if isinstance(register, str) else register
        self.meta_tags = meta_tags if meta_tags else set()

    def __hash__(self):
        return hash(self.data)

    @property
    def is_scratch(self) -> bool:
        return self.data in self.all_scratch_registers

    def cast_to(self, size: MemorySize) -> Register:
        return Register(self.data.cast_to(size))

    @classmethod
    def free_all(cls, lines: list[Instruction] | None = None):
        "frees all scratch registers"
        if lines is None:
            raise ValueError(
                "lines must be provided in order to free 64 bit stack memory.  Stack memory is being used because there was no more available 64 bit registers."
            )
        if cls.stack_memory_in_use > 0:
            lines.append(Instruction("add", cls("rsp"), cls.stack_memory_in_use))
        cls.available_64 = get_scratch_reg_list("")
        cls.available_32 = get_scratch_reg_list("d")
        cls.available_16 = get_scratch_reg_list("w")
        cls.available_8 = get_scratch_reg_list("b")
        cls.available_float = (
            [RegisterData[f"xmm{n}"] for n in reversed(range(8, 16))]
            if current_os == "Linux"
            else [RegisterData[f"xmm{n}"] for n in reversed(range(4, 16))]
        )

    def free(self, lines: list[Instruction] = None):
        rname = self.data.name[:3]
        if rname == "xmm":
            self.available_float.append(self.data)
        elif RegisterData[f"{rname}"] not in self.available_64:
            self.available_64.append(RegisterData[f"{rname}"])
            self.available_32.append(RegisterData[f"{rname}d"])
            self.available_16.append(RegisterData[f"{rname}w"])
            self.available_8.append(RegisterData[f"{rname}b"])

    @property
    def is_float(self) -> bool:
        return self.name.startswith("xmm")

    @classmethod
    def __request_wrapper(
        cls,
        reg_list: list[RegisterData],
        size: int | Literal["float"],
        specific: RegisterData | str | None = None,
    ) -> Register:
        try:
            if specific is None:
                reg: RegisterData = reg_list.pop()
            elif isinstance(specific, str):
                reg_list.remove(r := RegisterData[specific])
                reg: RegisterData = r
            elif isinstance(specific, RegisterData):
                reg_list.remove(r := specific)
                reg: RegisterData = r
            if size != "float":
                r_pref_name = reg.register_name[:3]
                if size != 64:
                    cls.available_64.remove(RegisterData[f"{r_pref_name}"])
                if size != 32:
                    cls.available_32.remove(RegisterData[f"{r_pref_name}d"])
                if size != 16:
                    cls.available_16.remove(RegisterData[f"{r_pref_name}w"])
                if size != 8:
                    cls.available_8.remove(RegisterData[f"{r_pref_name}b"])
                if c_reg := reg.get_callee_saved():
                    Program.CURRENT.current_function.push_callee_saved(cls(c_reg))
            return cls(reg)
        except IndexError as _:
            raise RuntimeError(f"Ran out of {size} bit scratch registers.")

    @classmethod
    def request_float(
        cls,
        specific: RegisterData | str | None = None,
        lines: list[Instruction] = None,
        offset: int = 0,
    ) -> Register:
        try:
            return cls.__request_wrapper(
                cls.available_float, "float", specific=specific
            )
        except RuntimeError as _:
            # create stack memory
            if lines is None:
                raise ValueError(
                    "lines must be provided in order to request stack memory.  Stack memory is being used because there are no more available floating point registers."
                )
            lines.append(Instruction("sub", cls("rsp"), 8))
            cls.stack_memory_in_use += 8
            return OffsetRegister(cls("rbp"), offset + cls.stack_memory_in_use, True)

    @classmethod
    def request_64(
        cls,
        specific: RegisterData | str | None = None,
        lines: list[Instruction] = None,
        offset: int = 0,
    ) -> Register:
        try:
            return cls.__request_wrapper(cls.available_64, 64, specific=specific)
        except RuntimeError as _:
            # create stack memory
            if lines is None:
                raise ValueError(
                    "lines must be provided in order to request stack memory.  Stack memory is being used because there are no more available 64 bit registers."
                )
            lines.append(Instruction("sub", cls("rsp"), 8))
            cls.stack_memory_in_use += 8
            return OffsetRegister(cls("rbp"), offset + cls.stack_memory_in_use, True, override_size=MemorySize.QWORD)

    @classmethod
    def request_32(cls, specific: RegisterData | str | None = None,
        lines: list[Instruction] = None,
        offset: int = 0
    ) -> Register:
        try:
            return cls.__request_wrapper(cls.available_32, 32, specific=specific)
        except RuntimeError as _:
            # create stack memory
            if lines is None:
                raise ValueError(
                    "lines must be provided in order to request stack memory.  Stack memory is being used because there are no more available 32 bit registers."
                )
            lines.append(Instruction("sub", cls("rsp"), 4))
            cls.stack_memory_in_use += 4
            return OffsetRegister(cls("rbp"), offset + cls.stack_memory_in_use, True, override_size=MemorySize.DWORD)

    @classmethod
    def request_16(cls, specific: RegisterData | str | None = None,
        lines: list[Instruction] = None,
        offset: int = 0
    ) -> Register:
        try:
            return cls.__request_wrapper(cls.available_16, 16, specific=specific)
        except RuntimeError as _:
            # create stack memory
            if lines is None:
                raise ValueError(
                    "lines must be provided in order to request stack memory.  Stack memory is being used because there are no more available 16 bit registers."
                )
            lines.append(Instruction("sub", cls("rsp"), 2))
            cls.stack_memory_in_use += 2
            return OffsetRegister(cls("rbp"), offset + cls.stack_memory_in_use, True, override_size=MemorySize.WORD)

    @classmethod
    def request_8(cls, specific: RegisterData | str | None = None,
        lines: list[Instruction] = None,
        offset: int = 0
    ) -> Register:
        try:
            return cls.__request_wrapper(cls.available_8, 8, specific=specific)
        except RuntimeError as _:
            # create stack memory
            if lines is None:
                raise ValueError(
                    "lines must be provided in order to request stack memory.  Stack memory is being used because there are no more available 8 bit registers."
                )
            lines.append(Instruction("sub", cls("rsp"), 1))
            cls.stack_memory_in_use += 1
            return OffsetRegister(cls("rbp"), offset + cls.stack_memory_in_use, True, override_size=MemorySize.BYTE)

    @property
    def name(self) -> str:
        return self.data.register_name

    @property
    def size(self) -> MemorySize:
        return self.data.size

    @property
    def position(self) -> Literal[0, 1]:
        """
        return 0 if upper bytes, return 1 if lower bytes
        """
        return self.data.position

    def __eq__(self, other: Register):
        return self.data == other.data

    def __gt__(self, other: Register):
        return self.size.value > other.size.value

    def __lt__(self, other: Register):
        return self.size.value > other.size.value

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"({self.name}[{self.size}] : {'lower' if self.position else 'upper'})"


InstructionDataType = tuple[
    str,
    list[list[MemorySize | type | int]],
    list[MemorySize | type | int | str | tuple[str, str] | None | Block],
]
#                          [name, argument permutations (literal integers identify a wildcard size match group), return memory (None means for all permutations, value of str of num means use the same size as that index in the permutation; if str is reg name it loads into that specific reg; val of None means unknown)]


class OffsetRegister(Register):
    def __init__(
        self,
        register: Register,
        offset: str | int | tuple[function, list],
        negative: bool = False,
        ptr: bool = False,
        rel: bool = False,
        override_size: MemorySize | None = None,
        meta_tags: set[str] = None,
    ):
        """
        offset can take in either str, int or a tuple with a function and arguments to be passed to said function in the format (def function(arg1, arg2, arg3), [arg1, arg2, arg3])
        """
        self.register = register
        self.offset = offset
        self.negative = negative
        self.ptr = ptr
        self.override_size = override_size
        self.meta_tags = meta_tags if meta_tags else set()
        self.is_rel = rel

    @property
    def name(self):
        return self.register.name
    
    def __hash__(self):
        return hash(f"{hash(self.register)}{self.meta_tags}{self.offset}{self.negative}{self.ptr}")

    @property
    def size(self):
        return self.override_size if self.override_size else self.register.size

    @property
    def position(self):
        return self.register.position

    @property
    def data(self):
        return self.register.data
    
    def __getitem__(self, index:Any) -> OffsetRegister:
        ret = copy.copy(self)
        if isinstance(index, str) and index != "":
            ret.offset = index.replace("{}", str(ret.offset))
        else:
            ret.offset = f"{ret.offset} + {index}"
        return ret

    def __str__(self) -> str:
        if any(isinstance(self.offset, t) for t in [int, str]):
            if self.offset:
                offset = self.offset
            else:
                offset = ""
        elif isinstance(self.offset, tuple):
            offset = self.offset[0](*self.offset[1])
        else:
            offset = self.offset
        return (
            f"{self.size.name}{' ptr ' if self.ptr else ''}[{'rel ' if self.is_rel else ''}{self.name}"
            + (("-" if self.negative else "+") if self.offset else "")
            + f"{offset}]"
        )


class StackVariable:
    def __init__(self, name: str, size: MemorySize, value: list | int = None):
        self.name = name
        self.size = size
        self.value = value
        self.empty = isinstance(self.value, int)

    @property
    def rel(self) -> OffsetStackVariable:
        return OffsetStackVariable(self, rel=True)

    def write(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.size.name}[{self.name}]"

    def __getitem__(self, offset: str) -> StackVariable | OffsetStackVariable:
        return OffsetStackVariable(self, offset)

    def declare(self):
        return (
            f"{self.name} {self.size.sec_bss_write if self.empty else self.size.sec_data_write} "
            + ", ".join(str(a) for a in self.value)
        )
    
    def __hash__(self):
        return hash(f"{self.name}{self.size}{self.value}{self.empty}")


class OffsetStackVariable(StackVariable):
    def __init__(self, variable: StackVariable, offset: str = "", negative: bool = False, rel: bool = False,):
        self.variable = variable
        self.offset = offset
        self.negative = negative
        self.is_rel = rel

    @property
    def rel(self) -> OffsetStackVariable:
        ret = copy.copy(self)
        ret.is_rel = True
        return ret

    @property
    def name(self):
        return self.variable.name

    @property
    def size(self):
        return self.variable.size

    @property
    def value(self):
        return self.variable.value

    def __str__(self) -> str:
        return (
            f"{self.size.name}[{'rel ' if self.is_rel else ''}{self.name}"
            + (("-" if self.negative else "+") if self.offset else "")
            + f"{self.offset}]"
        )
    
    def __hash__(self):
        return hash(f"{hash(self.variable)}{self.offset}{self.negative}")


# redefine enum meta to handle builtin enum names
class InstructionDataEnumMeta(EnumMeta):
    def __getitem__(cls, name):
        return super().__getitem__(
            f"{name}_" if name in {"and", "or", "not", "int"} else name
        )


class InstructionData(Enum, metaclass=InstructionDataEnumMeta):
    # this class's enums contains sizes, number of args etc to validate instructions
    mov: InstructionDataType = ("mov", [[0, 1], [0, int], [0, str]], [0, 0, 0])
    movsx: InstructionDataType = ("movsx", [[0, 1], [0, int], [0, str]], [0, 0, 0])
    movzx: InstructionDataType = ("movzx", [[0, 1], [0, int], [0, str]], [0, 0, 0])
    add: InstructionDataType = (
        "add",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    sub: InstructionDataType = (
        "sub",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )

    mul: InstructionDataType = ("mul", [[0], [int], [str]], ["0", None, None])
    div: InstructionDataType = ("div", [[0]], [("0", "0")])

    imul: InstructionDataType = (
        "imul",
        [
            [0, 0, 0],
            [0, 0, int],
            [0, 0, str],
            [0, 0],
            [0, int],
            [0, str],
            [0],
            [int],
            [str],
        ],
        ["0", "0", "0", "0", "0", "0", "0", None, None],
    )
    idiv: InstructionDataType = ("idiv", [[0]], [("0", "0")])

    inc: InstructionDataType = ("inc", [[0]], [0])
    dec: InstructionDataType = ("dec", [[0]], [0])
    syscall: InstructionDataType = ("syscall", [[]], [])
    ret: InstructionDataType = ("ret", [[]], [])
    cdq: InstructionDataType = ("cdq", [[]], [])
    cqo: InstructionDataType = ("cqo", [[]], [])

    push: InstructionDataType = ("push", [[MemorySize.QWORD]], [None])
    pop: InstructionDataType = ("pop", [[MemorySize.QWORD]], [0])

    cmp: InstructionDataType = ("cmp", [[0, 0], [0, int], [0, str]], [0, 0, 0])
    test: InstructionDataType = ("test", [[0, 0], [0, int], [0, str]], [0, 0, 0])

    and_: InstructionDataType = ("and", [[0, 0], [0, int], [0, str]], [0, 0, 0])
    or_: InstructionDataType = ("or", [[0, 0], [0, int], [0, str]], [0, 0, 0])
    xor: InstructionDataType = ("xor", [[0, 0], [0, int], [0, str]], [0, 0, 0])
    not_: InstructionDataType = ("not", [[0]], [0])
    neg: InstructionDataType = ("neg", [[0]], [0])
    shl: InstructionDataType = ("shl", [[0, 0], [0, int], [0, str]], [0, 0, 0])
    shr: InstructionDataType = ("shr", [[0, 0], [0, int], [0, str]], [0, 0, 0])
    sar: InstructionDataType = ("sar", [[0, 0], [0, int], [0, str]], [0, 0, 0])
    rol: InstructionDataType = ("rol", [[0, 0], [0, int], [0, str]], [0, 0, 0])
    ror: InstructionDataType = ("ror", [[0, 0], [0, int], [0, str]], [0, 0, 0])

    call: InstructionDataType = ("call", [[Block]], [Block])
    jmp: InstructionDataType = ("jmp", [[Block]], [Block])
    loop: InstructionDataType = ("loop", [[Block]], [Block])
    jne: InstructionDataType = ("jne", [[Block]], [Block])
    jle: InstructionDataType = ("jle", [[Block]], [Block])
    jl: InstructionDataType = ("jl", [[Block]], [Block])
    jge: InstructionDataType = ("jge", [[Block]], [Block])
    jg: InstructionDataType = ("jg", [[Block]], [Block])
    je: InstructionDataType = ("je", [[Block]], [Block])
    jz: InstructionDataType = ("jz", [[Block]], [Block])
    jnz: InstructionDataType = ("jnz", [[Block]], [Block])
    jns: InstructionDataType = ("jns", [[Block]], [Block])
    js: InstructionDataType = ("js", [[Block]], [Block])

    lea: InstructionDataType = ("lea", [[0, 1]], [0])

    nop: InstructionDataType = ("nop", [[]], [])
    clc: InstructionDataType = ("clc", [[]], [])
    stc: InstructionDataType = ("stc", [[]], [])
    cld: InstructionDataType = ("cld", [[]], [])
    std: InstructionDataType = ("std", [[]], [])
    rep: InstructionDataType = ("rep", [[]], [])
    int_: InstructionDataType = ("int", [[int], [str]], [None, None])

    sete: InstructionDataType = ("sete", [[0]], [0])
    setz: InstructionDataType = ("setz", [[0]], [0])
    setne: InstructionDataType = ("setne", [[0]], [0])
    setnz: InstructionDataType = ("setnz", [[0]], [0])
    setg: InstructionDataType = ("setg", [[0]], [0])
    setnle: InstructionDataType = ("setnle", [[0]], [0])
    setge: InstructionDataType = ("setge", [[0]], [0])
    setnl: InstructionDataType = ("setnl", [[0]], [0])
    setl: InstructionDataType = ("setl", [[0]], [0])
    setnge: InstructionDataType = ("setnge", [[0]], [0])
    setle: InstructionDataType = ("setle", [[0]], [0])
    setng: InstructionDataType = ("setng", [[0]], [0])
    seta: InstructionDataType = ("seta", [[0]], [0])
    setnbe: InstructionDataType = ("setnbe", [[0]], [0])
    setae: InstructionDataType = ("setae", [[0]], [0])
    setnb: InstructionDataType = ("setnb", [[0]], [0])
    setnc: InstructionDataType = ("setnc", [[0]], [0])
    setb: InstructionDataType = ("setb", [[0]], [0])
    setnae: InstructionDataType = ("setnae", [[0]], [0])
    setc: InstructionDataType = ("setc", [[0]], [0])
    setbe: InstructionDataType = ("setbe", [[0]], [0])
    setna: InstructionDataType = ("setna", [[0]], [0])
    sets: InstructionDataType = ("sets", [[0]], [0])
    setns: InstructionDataType = ("setns", [[0]], [0])
    seto: InstructionDataType = ("seto", [[0]], [0])
    setno: InstructionDataType = ("setno", [[0]], [0])
    setp: InstructionDataType = ("setp", [[0]], [0])
    setpe: InstructionDataType = ("setpe", [[0]], [0])
    setnp: InstructionDataType = ("setnp", [[0]], [0])
    setpo: InstructionDataType = ("setpo", [[0]], [0])

    # floating point operations:
    fld: InstructionDataType = ("fld", [[0]], [None])
    fadd: InstructionDataType = ("fadd", [[]], [])
    fmul: InstructionDataType = ("fmul", [[]], [])
    fstp: InstructionDataType = ("fstp", [[MemorySize.QWORD]], [0])
    fistp: InstructionDataType = ("fistp", [[0]], [0])

    movq: InstructionDataType = ("movq", [[0, 1], [0, int], [0, str]], [0, 0, 0])
    movd: InstructionDataType = ("movd", [[0, 1], [0, int], [0, str]], [0, 0, 0])
    movapd: InstructionDataType = ("movapd", [[0, 1], [0, int], [0, str]], [0, 0, 0])
    movsd: InstructionDataType = ("movsd", [[0, 1], [0, int], [0, str]], [0, 0, 0])
    movmskpd: InstructionDataType = ("movmskpd", [[0,1]], [0])
    pextrb: InstructionDataType = ("pextrb", [[0,1,2]], [0])


    # int to float and vice versa

    cvtsi2sd: InstructionDataType = ("cvtsi2sd", [[0, 1]], [0])
    cvtsi2ss: InstructionDataType = ("cvtsi2ss", [[0, 0]], [0])
    cvttsd2si: InstructionDataType = ("cvttsd2si", [[0, 0]], [0])
    cvtsd2si: InstructionDataType = ("cvtsd2si", [[0, 1]], [0])

    # float operations

    addps: InstructionDataType = (
        "addps",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    addsd: InstructionDataType = (
        "addsd",
        [[0, 1], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    paddq: InstructionDataType = (
        "paddq",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    subsd: InstructionDataType = (
        "subsd",
        [[0, 1], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    subpd: InstructionDataType = (
        "subpd",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    mulsd: InstructionDataType = (
        "mulsd",
        [[0, 1], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    mulpd: InstructionDataType = (
        "mulpd",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    divsd: InstructionDataType = (
        "divsd",
        [[0, 1], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    divpd: InstructionDataType = (
        "divpd",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    minsd: InstructionDataType = (
        "minsd",
        [[0, 1], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    minpd: InstructionDataType = (
        "minpd",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    maxsd: InstructionDataType = (
        "maxsd",
        [[0, 1], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    maxpd: InstructionDataType = (
        "maxpd",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    sqrtsd: InstructionDataType = (
        "sqrtsd",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    sqrtpd: InstructionDataType = (
        "sqrtpd",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    andpd: InstructionDataType = (
        "andpd",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    orpd: InstructionDataType = (
        "orpd",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    xorpd: InstructionDataType = (
        "xorpd",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    pxor: InstructionDataType = (
        "pxor",
        [[0,0]],
        [0]
    )
    hsubpd: InstructionDataType = (
        "hsubpd",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    haddpd: InstructionDataType = (
        "haddpd",
        [[0, 0], [0, int], [0, str], [0], [int], [str]],
        [0, 0, 0, "0", None, None],
    )
    cmpsd: InstructionDataType = ("cmpsd", [[0, 0], [0, int], [0, str]], [0, 0, 0])
    cmppd: InstructionDataType = ("cmppd", [[0, 0], [0, int], [0, str]], [0, 0, 0])
    comisd: InstructionDataType = ("comisd", [[0, 0], [0, int], [0, str]], [0, 0, 0])
    ucomisd: InstructionDataType = ("ucomisd", [[0, 0], [0, int], [0, str]], [0, 0, 0])

    vfmadd213ss: InstructionDataType = ("vfmadd213ss", [[0, 0, 0]], [0])
    vfmadd213ps: InstructionDataType = ("vfmadd213ps", [[0, 0, 0]], [0])
    vfmadd132ss: InstructionDataType = ("vfmadd132ss", [[0, 0, 0]], [0])
    vfmadd132pd: InstructionDataType = ("vfmadd132pd", [[0, 0, 0]], [0])
    vfmadd213sd: InstructionDataType = ("vfmadd213sd", [[0, 0, 0]], [0])
    vfmadd231pd: InstructionDataType = ("vfmadd132pd", [[0, 0, 0]], [0])

    @classmethod
    def from_py_type(cls, name: str, py_type: type) -> InstructionData:
        aliases = {}
        match py_type.__name__:
            case "int":
                pass
            case "float":
                aliases = {
                    "cmp": "comisd",
                    "add": "addsd",
                    "sub": "subsd",
                    "mul": "mulsd",
                    "div": "divsd",
                    "imul": "mulsd",
                    "idiv": "divsd",
                    "sqrt": "sqrtsd",
                    "max": "maxsd",
                    "min": "minsd",
                }

        if name in aliases:
            return cls[aliases[name]]
        return cls[name]

    @property
    def instruction_name(self) -> str:
        return self.value[0]

    @property
    def arguments(
        self,
    ) -> list[MemorySize | type | int | str | tuple[str, str] | None | StackVariable]:
        return self.value[1]

    @property
    def ret_key(
        self,
    ) -> list[MemorySize | type | int | str | tuple[str, str] | None | StackVariable]:
        return self.value[2]


class Instruction:
    def __init__(
        self,
        instruction: str | InstructionData,
        *arguments: list[Register | str | int | StackVariable | Block],
    ):
        self.data = (
            InstructionData[instruction]
            if isinstance(instruction, str)
            else instruction
        )
        self.arguments = arguments
        self.err_msg = None
        self.__ret = None
        if not self:
            raise SyntaxError(f'Invalid instruction: "{self}".\nReason: {self.err_msg}')

    @property
    def name(self) -> str:
        return self.data.instruction_name

    def __str__(self):
        return f"{self.name} " + ", ".join(str(a) for a in self.arguments)

    def write(self):
        if not self:
            raise SyntaxError(f'Invalid instruction: "{self}".\nReason: {self.err_msg}')
        return str(self)

    def __bool__(self) -> bool:
        """
        This is where the instruction arguments are validated.
        """
        # >> TODO Refactor argument validation << #
        return True

    def __get_ret(self, r_ind: int):
        index = self.data.ret_key[r_ind]

        if isinstance(index, int):
            if hasattr(self.arguments[index], "size"):
                return self.arguments[index]
        elif isinstance(index, str):
            int_ind = int(index)
            if hasattr(self.arguments[int_ind], "size"):
                return Register(RegisterData.from_size(self.arguments[int_ind].size))
        elif isinstance(index, tuple):
            ind_1, _ = tuple(int(i) for i in index)
            if hasattr(self.arguments[ind_1], "size"):
                if self.data.instruction_name in {"div", "idiv"}:
                    return tuple(
                        Register(r)
                        for r in RegisterData.from_size(
                            self.arguments[ind_1].size.shrink
                        )
                    )
                return tuple(
                    Register(r)
                    for r in RegisterData.from_size(self.arguments[ind_1].size)
                )
        return None

    def __call__(self, recorder: Program | None = None):
        if not self:
            raise SyntaxError(f'Invalid instruction: "{self}".\nReason: {self.err_msg}')
        (recorder if recorder else Program.CURRENT).append(self)
        return self.__ret


class Function(Block):
    # None argument gets casted to 64 bit and pushed/popped to the stack
    def __init__(
        self,
        arguments: list[Register | None],
        signed_args: set[int] | None = None,
        return_register: Register | None = None,
        return_signed: bool = False,
        label: str | None = None,
        ret_py_type: type = int,
        arguments_py_type: list[type] = None,
    ):
        Program.CURRENT.function_stack.append(self)
        super().__init__(label)
        self.arguments = []
        self.stack_offset = -8
        self.signed_args = {} if signed_args is None else signed_args
        self.return_register = return_register
        self.ret_py_type = ret_py_type
        self.ctypes_restype = (
            self.return_register.size.to_ctype(return_signed, self.ret_py_type)
            if self.return_register
            else None
        )
        self.ctypes_arguments = []
        self.arguments_py_type: list[type] = (
            arguments_py_type if arguments_py_type else []
        )
        for a_n, arg in enumerate(arguments):
            if arg is None:
                self.stack_offset += 8
                self.arguments.append(
                    OffsetRegister(Register("rsp"), self.stack_offset)
                )
                self.ctypes_arguments.append(
                    MemorySize.QWORD.to_ctype(
                        a_n in self.signed_args,
                        self.arguments_py_type[a_n]
                        if a_n < len(self.arguments_py_type)
                        else int,
                    )
                )
            else:
                self.arguments.append(arg)
                self.ctypes_arguments.append(
                    arg.size.to_ctype(
                        a_n in self.signed_args,
                        self.arguments_py_type[a_n]
                        if a_n < len(self.arguments_py_type)
                        else int,
                    )
                )
        self.callee_saved_regs: list[Register] = []

    def push_callee_saved(self, reg: Register):
        if reg not in self.callee_saved_regs:
            self.callee_saved_regs.append(reg)

    def end_definition(self):
        Program.CURRENT.function_stack.pop()

    def __str__(self):
        return f"{self.label}"

    def write(self):
        return (
            f"{self}:"
            + "\n    push rbp"
            + "\n    mov rbp, rsp"
            + (
                "".join([f"\n    push {reg}" for reg in self.callee_saved_regs])
                if self.callee_saved_regs
                else ""
            )
        )

    def __call__(self, recorder: Program | None = None):
        (recorder if recorder else Program.CURRENT).append(self)
        (recorder if recorder else Program.CURRENT).functions[self.label] = self
        return self

    def ret(self):
        for reg in reversed(self.callee_saved_regs):
            Instruction("pop", reg)()
        Instruction("mov", Register("rsp"), Register("rbp"))()
        Instruction("pop", Register("rbp"))()
        Instruction("ret")()

    def call(self, *arguments: list) -> Register | None:
        if len(
            list(filter(lambda a: not isinstance(a, OffsetRegister), self.arguments))
        ) != len(self.arguments):
            Instruction("sub", Register("rsp"), 32)()
        for a_n, arg in reversed(list(enumerate(self.arguments))):
            d_ = (
                "d"
                if hasattr(arguments[a_n], "size")
                and arg.size == MemorySize.QWORD
                and arguments[a_n].size == MemorySize.DWORD
                else ""
            )

            if isinstance(arg, Register):
                if arg > arguments[a_n]:
                    Instruction(
                        ("movsx" if a_n in self.signed_args else "movzx") + d_,
                        arg,
                        arguments[a_n],
                    )()
                else:
                    Instruction("mov", arg, arguments[a_n])()

            elif isinstance(arg, OffsetRegister):
                if arg > arguments[a_n]:
                    Instruction(
                        ("movsx" if a_n in self.signed_args else "movzx") + d_,
                        arg,
                        arguments[a_n],
                    )()
                else:
                    Instruction("mov", arg, arguments[a_n])()

        Instruction("call", self)()
        return self.return_register


if __name__ == "__main__":
    Reg = Register
    RegD = RegisterData
    Ins = Instruction
    InsD = InstructionData

    ah = Reg("ah")
    al = Reg("al")
    dx = Reg("dx")
    ax = Reg("ax")
    edx = Reg("edx")
    eax = Reg("eax")
    rdx = Reg("rdx")
    rax = Reg("rax")
    rdi = Reg("rdi")
    rsi = Reg("rsi")

    Program.CURRENT.name = "test"

    Program.CURRENT.comment("Function start:")

    func_add_a_b = Function([rdi, rsi], return_register=rax, label="add_a_b")()

    Program.CURRENT.new_line()

    f_ret = Ins("mov", rax, func_add_a_b.arguments[0])()

    Ins("add", f_ret, func_add_a_b.arguments[1])()

    Program.CURRENT.new_line()

    func_add_a_b.ret()

    Program.CURRENT.compile()
    Program.CURRENT.link(args={"shared": None}, output_extension=".so")

    total = 0
    for _ in range(1, 101):
        total = Program.CURRENT.call("add_a_b", total, 2)
        print(total)
        print("+")
    print(f" = {total}")
    # prints 7
