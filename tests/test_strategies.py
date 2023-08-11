r"""By Brian Beckman.

Copyright 2019 - 2023 GSI Technology, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The tests in this file let us experiment with strategies that generate
tests for other functions and classes in the project. These "test"
don't do a lot of asserting; they're meant for printing examples.
"""

import hypothesis
import hypothesis.strategies as st
from hypothesis import given

import open_belex.bleir.types as BLEIR
from open_belex.common.subset import *
from open_belex_tests.strategies import *


# I used to think the following strategies failed to produce big numbers.
# However, hypothesis behaves differently when running slowly under the
# debugger, as with breakpoints on "debug_me." When running slowly, hypothesis
# produces fewer samples and "simpler" samples, presumably closer to the
# shrinkage targets. A better way to see what's going on is just to print. Run
# pytest with the -s option to see the printout.


big_int_strategy = st.integers(
    min_value=0,
    max_value=((2**64) - 1))


huge_int_strategy = st.integers(
    min_value=0,
    max_value=((2**2048) - 1))


@given(big_int=big_int_strategy)
def test_big_int_range(big_int):
    print(big_int)


@given(huge_int=huge_int_strategy)
def test_huge_int_range(huge_int):
    print(huge_int)


@hypothesis.settings(deadline=None)
@given(huge_bytes=st.binary(max_size=2048))
def test_huge_binary(huge_bytes):
    l0 = len(huge_bytes)
    print({"len(huge_bytes)": l0})


@given(shape=st.builds(
    lambda x: (x,),
    st.integers(min_value=0, max_value=20)))
def test_shape(shape):
    l0 = len(shape)
    print({"shape": shape, "len(shape)": l0})


# This strategy lets us generate longer sequences of bigger numbers, for
# testing "Subset.py" for markers.

# arrays(
#     dtype=numpy.int16,
#     elements=st.integers(min_value=0, max_value=NUM_PLATS_PER_APUC-1),
#     unique=True,
#     shape=st.builds(lambda x: (x,), st.integers(min_value=0, max_value=100))
#     # shape=(20,)
# ))

@given(an_array=marker_biggish_array_strategy)
def test_biggish_array(an_array):
    l0 = len(an_array)
    hex_ = Subset(max=NUM_PLATS_PER_APUC - 1, user_input=an_array).big_endian_hex
    print({"len": l0, "markers": hex_})


#   ___ _    ___ ___ ___   ___ _            _            _
#  | _ ) |  | __|_ _| _ \ / __| |_ _ _ __ _| |_ ___ __ _(_)___ ___
#  | _ \ |__| _| | ||   / \__ \  _| '_/ _` |  _/ -_) _` | / -_|_-<
#  |___/____|___|___|_|_\ |___/\__|_| \__,_|\__\___\__, |_\___/__/
#                                                  |___/


bleir_src_expr_strategy = st.sampled_from(BLEIR.SRC_EXPR)
bleir_bit_expr_strategy = st.sampled_from(BLEIR.BIT_EXPR)
bleir_assign_op_strategy = st.sampled_from(BLEIR.ASSIGN_OP)
bleir_register_nym_strategy = st.characters


@given(src=bleir_src_expr_strategy,
       bit=bleir_bit_expr_strategy,
       assign_op=bleir_assign_op_strategy,)
def test_bleir_expr_enums(src, bit, assign_op):

    foo = BLEIR.SRC_EXPR.find_by_value(src.value)
    assert foo is not None

    bar = BLEIR.BIT_EXPR.find_by_value(bit.value)
    assert bar is not None

    found_op = False
    for op in list(BLEIR.ASSIGN_OP):
        if op.value == assign_op.value:
            found_op = True
    assert found_op

    baz = BLEIR.RN_REG('lvr')
    qux = BLEIR.SB[baz]

    stop_here = 42
