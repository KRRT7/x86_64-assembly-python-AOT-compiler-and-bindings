from x86_64_assembly_bindings import (
    Program,
    Function,
    Register,
    Instruction,
)


def test_add_a_b_function():
    Reg = Register
    Ins = Instruction

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
    print(Program.CURRENT)
    Program.CURRENT.compile()
    Program.CURRENT.link(args={"shared": None}, output_extension=".so")

    total = 0
    for _ in range(1, 101):
        total = Program.CURRENT.call("add_a_b", total, 2)
    assert total == 7
