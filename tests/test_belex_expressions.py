r"""
By Dylon Edwards
"""

import numpy as np

import pytest

from open_belex.expressions import SymbolTable, Variable, VariableAccess


def test_VariableAccess_init():
    foo = Variable("foo")

    # Index Lists
    # -----------

    expr = VariableAccess(foo)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    expr = VariableAccess(foo, indices=[])
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = [0]
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert indices == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = [1, 3, 7, 8, 14]
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert indices == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = list(range(16))
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert indices == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = ["0"]
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = ["1", "3", "7", "8", "E"]
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert [1, 3, 7, 8, 14] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = ["1", 5, "E", 15]
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert [1, 5, 14, 15] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = [f"{i:x}" for i in range(16)]
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    with pytest.raises(ValueError):
        indices = [-1]
        VariableAccess(foo, indices=indices)

    with pytest.raises(ValueError):
        indices = [-2]
        VariableAccess(foo, indices=indices)

    with pytest.raises(ValueError):
        indices = [16]
        VariableAccess(foo, indices=indices)

    with pytest.raises(ValueError):
        indices = [17]
        VariableAccess(foo, indices=indices)

    # Index Sets
    # ----------

    expr = VariableAccess(foo, indices=set())
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = {0}
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = {1, 3, 7, 8, 14}
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert [1, 3, 7, 8, 14] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = set(range(16))
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = {"0"}
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = {"1", "7", "3", "E", "8"}
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert [1, 3, 7, 8, 14] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = {"1", 5, "E", 15}
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert [1, 5, 14, 15] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = {f"{i:x}" for i in range(16)}
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    # Section Masks
    # -------------

    # Note: Not including any index means include all indices
    section_mask = [False] * 16
    expr = VariableAccess(foo, indices=section_mask)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    section_mask = [False] * 16
    section_mask[0] = True
    expr = VariableAccess(foo, indices=section_mask)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    section_mask = [False] * 16
    section_mask[1] = True
    section_mask[5] = True
    section_mask[14] = True
    expr = VariableAccess(foo, indices=section_mask)
    assert foo == expr.var
    assert [1, 5, 14] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    section_mask = [True] * 16
    expr = VariableAccess(foo, indices=section_mask)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    with pytest.raises(ValueError):
        section_mask = [False]
        VariableAccess(foo, indices=section_mask)

    with pytest.raises(ValueError):
        section_mask = [False] * 15
        VariableAccess(foo, indices=section_mask)

    with pytest.raises(ValueError):
        section_mask = [False] * 17
        VariableAccess(foo, indices=section_mask)

    with pytest.raises(ValueError):
        section_mask = [False] * 18
        VariableAccess(foo, indices=section_mask)

    # Index Iterator
    # --------------

    indices = iter([])
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = iter([0])
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = iter([0, 1, 5, 12])
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert [0, 1, 5, 12] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = iter(range(16))
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = iter(["0"])
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = iter(["0", "1", "5", "C"])
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert [0, 1, 5, 12] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = iter([f"{i:x}" for i in range(16)])
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    # Section Mask Iterator
    # ---------------------

    section_mask = iter([False] * 16)
    expr = VariableAccess(foo, indices=section_mask)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    section_mask = [False] * 16
    section_mask[0] = True
    section_mask = iter(section_mask)
    expr = VariableAccess(foo, indices=section_mask)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    section_mask = [False] * 16
    section_mask[3] = True
    section_mask[5] = True
    section_mask[9] = True
    section_mask = iter(section_mask)
    expr = VariableAccess(foo, indices=section_mask)
    assert foo == expr.var
    assert [3, 5, 9] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    section_mask = iter([True] * 16)
    expr = VariableAccess(foo, indices=section_mask)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    # Section Mappings
    # ----------------

    # Note: Not including any index means include all indices
    mapping = {}
    expr = VariableAccess(foo, indices=mapping)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    # Note: Not including any index means include all indices
    mapping = {index: False for index in range(16)}
    expr = VariableAccess(foo, indices=mapping)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    mapping = {0: True}
    expr = VariableAccess(foo, indices=mapping)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    mapping = {0: True, 1: False, 2: True}
    expr = VariableAccess(foo, indices=mapping)
    assert foo == expr.var
    assert [0, 2] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    mapping = {index: False for index in range(16)}
    mapping[3] = True
    mapping[9] = True
    mapping[12] = True
    expr = VariableAccess(foo, indices=mapping)
    assert foo == expr.var
    assert [3, 9, 12] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    mapping = {index: True for index in range(16)}
    expr = VariableAccess(foo, indices=mapping)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    mapping = {"0": True}
    expr = VariableAccess(foo, indices=mapping)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    mapping = {"0": False, 9: True, "A": True}
    expr = VariableAccess(foo, indices=mapping)
    assert foo == expr.var
    assert [9, 10] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    mapping = {f"{i:x}": True for i in range(16)}
    expr = VariableAccess(foo, indices=mapping)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    with pytest.raises(ValueError):
        mapping = {-1: True}
        VariableAccess(foo, indices=mapping)

    with pytest.raises(ValueError):
        mapping = {-2: True}
        VariableAccess(foo, indices=mapping)

    with pytest.raises(ValueError):
        mapping = {16: True}
        VariableAccess(foo, indices=mapping)

    with pytest.raises(ValueError):
        mapping = {17: True}
        VariableAccess(foo, indices=mapping)

    with pytest.raises(ValueError):
        mapping = {0: 0}
        VariableAccess(foo, indices=mapping)

    with pytest.raises(ValueError):
        mapping = {0: "True"}
        VariableAccess(foo, indices=mapping)

    # Index Range
    # -----------

    rng = range(0)
    expr = VariableAccess(foo, indices=rng)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    rng = range(1)
    expr = VariableAccess(foo, indices=rng)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    rng = range(3)
    expr = VariableAccess(foo, indices=rng)
    assert foo == expr.var
    assert [0, 1, 2] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    rng = range(16)
    expr = VariableAccess(foo, indices=rng)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    # Index Slice
    # -----------

    slc = slice(0)
    expr = VariableAccess(foo, indices=slc)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    slc = slice(1)
    expr = VariableAccess(foo, indices=slc)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    slc = slice(3)
    expr = VariableAccess(foo, indices=slc)
    assert foo == expr.var
    assert [0, 1, 2] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    slc = slice(16)
    expr = VariableAccess(foo, indices=slc)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    slc = slice(0, 1)
    expr = VariableAccess(foo, indices=slc)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    slc = slice(3, 6)
    expr = VariableAccess(foo, indices=slc)
    assert foo == expr.var
    assert [3, 4, 5] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    slc = slice(3, 8, 2)
    expr = VariableAccess(foo, indices=slc)
    assert foo == expr.var
    assert [3, 5, 7] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    slc = slice(0, 16)
    expr = VariableAccess(foo, indices=slc)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    with pytest.raises(ValueError):
        slc = slice(17)
        VariableAccess(foo, indices=slc)

    with pytest.raises(ValueError):
        slc = slice(0, -2, -1)
        VariableAccess(foo, indices=slc)

    # Index Literal
    # -------------

    index = 0
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = 3
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [3] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    with pytest.raises(ValueError):
        index = -1
        VariableAccess(foo, indices=index)

    with pytest.raises(ValueError):
        index = 16
        VariableAccess(foo, indices=index)

    # Hex Literal
    # -----------

    index = "0"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "3"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [3] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "F"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [15] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "048C"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [0, 4, 8, 12] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "ABCDEF"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [10, 11, 12, 13, 14, 15] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0123456789abcdef"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "CCC"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [12, 12, 12] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "321"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [3, 2, 1] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0x0"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0x1"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0x2"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [1] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0xF"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [0, 1, 2, 3] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0x0F"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [0, 1, 2, 3] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0x10"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [4] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0xF0"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [4, 5, 6, 7] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0xF3"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [0, 1, 4, 5, 6, 7] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0x001"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [0] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0xABC"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [2, 3, 4, 5, 7, 9, 11] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0x0000"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0xF000"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [12, 13, 14, 15] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0xF0F0"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [4, 5, 6, 7, 12, 13, 14, 15] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = "0xFFFF"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    # Don't do this even though it is a valid hex value in the range [0,16)
    index = "0x00FFFF"
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert list(range(16)) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    with pytest.raises(ValueError):
        index = "0x"
        VariableAccess(foo, indices=index)

    with pytest.raises(ValueError):
        index = "0x10000"
        VariableAccess(foo, indices=index)

    with pytest.raises(ValueError):
        index = "0x*$"
        VariableAccess(foo, indices=index)

    with pytest.raises(ValueError):
        index = "Z"
        VariableAccess(foo, indices=index)

    with pytest.raises(ValueError):
        index = "^"
        VariableAccess(foo, indices=index)

    # Numpy Arrays
    # ------------

    index = np.int8(1)
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [1] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = np.int16(2)
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [2] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = np.int32(3)
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [3] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = np.int64(4)
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [4] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = np.uint8(1)
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [1] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = np.uint16(2)
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [2] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = np.uint32(3)
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [3] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    index = np.uint64(4)
    expr = VariableAccess(foo, indices=index)
    assert foo == expr.var
    assert [4] == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()

    indices = np.linspace(1, 1+12, 4).astype(int)
    expr = VariableAccess(foo, indices=indices)
    assert foo == expr.var
    assert list(indices) == expr.indices
    assert expr is not expr.clone()
    assert expr == expr.clone()


def test_SymbolTable_get_temp():
    tbl = SymbolTable()
    assert Variable("t_0", is_temp=True) == tbl.get_temp()
    assert Variable("t_1", is_temp=True) == tbl.get_temp()
    assert Variable("t_2", is_temp=True) == tbl.get_temp()


def test_SymbolTable_add_symbol():
    tbl = SymbolTable()
    assert Variable("foo") == tbl.variable_for_symstr("foo")
    assert Variable("bar") == tbl.variable_for_symstr("bar")


def test_SymbolTable_exist():
    tbl = SymbolTable()

    tbl.variable_for_symstr("foo")
    tbl.variable_for_symstr("bar")
    tbl.get_temp()

    assert tbl.exist("foo")
    assert tbl.exist("bar")
    assert tbl.exist("t_0")
    assert not tbl.exist("baz")
    assert not tbl.exist("t_1")


def test_SymbolTable_getitem():
    tbl = SymbolTable()

    foo = tbl.variable_for_symstr("foo")
    bar = tbl.variable_for_symstr("bar")
    t_0 = tbl.get_temp()

    assert foo == tbl["foo"]
    assert bar == tbl["bar"]
    assert t_0 == tbl["t_0"]

    with pytest.raises(KeyError):
        tbl["baz"]
