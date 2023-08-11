r"""By Dylon Edwards

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
"""

from open_belex.bleir.analyzers import LiveParameterMarker
from open_belex.bleir.optimizers import (DoubleNegativeResolver,
                                         UnusedParameterRemover)
from open_belex.bleir.types import (RL, RN_REG, SB, SM_REG, SRC_EXPR, UNARY_OP,
                                    UNARY_SRC, Fragment, assign, masked,
                                    statement)
from open_belex.bleir.walkables import BLEIRWalker


def test_double_negative_resolver():
    walker = BLEIRWalker()
    optimizer = DoubleNegativeResolver()

    unary_src_00 = UNARY_SRC(expression=SRC_EXPR.GGL, operator=None)
    unary_src_01 = UNARY_SRC(expression=SRC_EXPR.GGL, operator=UNARY_OP.NEGATE)
    unary_src_10 = UNARY_SRC(expression=SRC_EXPR.INV_GGL, operator=None)
    unary_src_11 = UNARY_SRC(expression=SRC_EXPR.INV_GGL, operator=UNARY_OP.NEGATE)

    # Non-inverted <SRC>, no negation operation
    assert walker.walk(optimizer, unary_src_00) == unary_src_00

    # Non-inverted <SRC>, with negation operation
    assert walker.walk(optimizer, unary_src_01) == \
        UNARY_SRC(expression=SRC_EXPR.INV_GGL, operator=None)

    # Inverted <SRC>, no negation operation
    assert walker.walk(optimizer, unary_src_10) == unary_src_10

    # Inverted <SRC>, with negation operation
    assert walker.walk(optimizer, unary_src_11) == \
        UNARY_SRC(expression=SRC_EXPR.GGL, operator=None)


def test_unused_parameter_remover():
    walker = BLEIRWalker()
    live_parameter_marker = LiveParameterMarker()
    unused_parameter_remover = UnusedParameterRemover(live_parameter_marker)

    lvr_rp = RN_REG("lvr")
    rvr_rp = RN_REG("rvr")
    r2vr_rp = RN_REG("r2vr")

    msk_rp = SM_REG("msk")
    fs_rp = SM_REG("fs")

    fragment = Fragment(
        identifier="fragment",
        parameters=[lvr_rp, rvr_rp, r2vr_rp, msk_rp, fs_rp],
        operations=[
            statement(masked(msk_rp, assign(SB[lvr_rp], RL))),
        ])

    walker.walk(live_parameter_marker, fragment)
    assert walker.walk(unused_parameter_remover, fragment) == Fragment(
        identifier="fragment",
        parameters=[lvr_rp, msk_rp],
        operations=[
            statement(masked(msk_rp, assign(SB[lvr_rp], RL))),
        ])


# # Regression test
# def test_unused_parameter_remover_against_full_scts_to_marked():
#     # There was a bug in which the fragment's unused parameters were removed before the fragment
#     # caller call's, and the constructor of the fragment caller call zipped over the formal and
#     # actual parameters. Since there were fewer formal parameters than actual parameters, the zip
#     # did not iterate over all the actual parameters, causing live parameters to be removed instead
#     # of the dead ones.

#     lvr_vp = 1
#     rvr_vp = 2  # <<- this is a dead parameter
#     mvr_vp = 3

#     lsec_vp = 'FFFF'
#     rsec_vp = 'F00D'
#     msec_vp = 'BEEF'

#     blecci = BLECCI()
#     fragment_caller_call = blecci._bb_seq_op__full_scts_to_marked(
#         lvr=lvr_vp,
#         rvr=rvr_vp,
#         mvr=mvr_vp,
#         lsec=lsec_vp,
#         rsec=rsec_vp,
#         msec=msec_vp)

#     walker = BLEIRWalker()

#     allocate_registers = AllocateRegisters()
#     transformed_fragment_caller_call = walker.walk(allocate_registers, fragment_caller_call)

#     live_parameter_marker = LiveParameterMarker()
#     walker.walk(live_parameter_marker, fragment_caller_call)

#     unused_parameter_remover = UnusedParameterRemover(live_parameter_marker)
#     transformed_fragment_caller_call = walker.walk(unused_parameter_remover,
#                                                    transformed_fragment_caller_call)
#     transformed_fragment_caller = transformed_fragment_caller_call.caller
#     transformed_fragment = transformed_fragment_caller.fragment

#     assert len(transformed_fragment_caller_call.parameters) == 5
#     assert transformed_fragment_caller_call.parameters == (lvr_vp, mvr_vp,
#                                                            int(lsec_vp, 16),
#                                                            int(rsec_vp, 16),
#                                                            int(msec_vp, 16))

#     assert len(transformed_fragment_caller.registers) == 5
#     allocated_parameters = transformed_fragment_caller.register_map.keys()
#     allocated_identifiers = [register_parameter.identifier
#                              for register_parameter in allocated_parameters]
#     assert allocated_identifiers == ["lvr", "mvr", "lsec", "rsec", "msec"]

#     assert len(transformed_fragment.parameters) == 5
#     parameter_identifiers = [register_parameter.identifier
#                              for register_parameter in transformed_fragment.parameters]
#     assert parameter_identifiers == ["lvr", "mvr", "lsec", "rsec", "msec"]
