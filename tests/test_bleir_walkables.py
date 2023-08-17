r"""
By Dylon Edwards
"""

from typing import Any
from unittest.mock import MagicMock, call

import numpy as np

from open_belex.bleir.types import (ASSIGN_OP, ASSIGNMENT, BINARY_EXPR, BINOP,
                                    BIT_EXPR, BROADCAST, BROADCAST_EXPR, MASK,
                                    MASKED, READ, RL_EXPR, RN_REG,
                                    RSP2K_ASSIGNMENT, RSP2K_EXPR, RSP2K_RVALUE,
                                    RSP16_ASSIGNMENT, RSP16_EXPR, RSP16_RVALUE,
                                    RSP32K_ASSIGNMENT, RSP32K_EXPR,
                                    RSP32K_RVALUE, RSP256_ASSIGNMENT,
                                    RSP256_EXPR, RSP256_RVALUE, SB,
                                    SHIFTED_SM_REG, SM_REG, SPECIAL, SRC_EXPR,
                                    STATEMENT, UNARY_EXPR, UNARY_OP, UNARY_SB,
                                    UNARY_SRC, WRITE, Example, Fragment,
                                    FragmentCaller, FragmentCallerCall,
                                    MultiStatement, Snippet, ValueParameter)
from open_belex.bleir.walkables import (BLEIRListener, BLEIRTransformer,
                                        BLEIRVisitor, BLEIRWalker)
from open_belex.common.constants import NUM_PLATS_PER_APUC


def stub_visitor():
    visitor = BLEIRVisitor()
    visitor.visit_snippet = MagicMock()
    visitor.visit_example = MagicMock()
    visitor.visit_value_parameter = MagicMock()
    visitor.visit_fragment_caller_call = MagicMock()
    visitor.visit_fragment_caller = MagicMock()
    visitor.visit_allocated_register = MagicMock()
    visitor.visit_fragment = MagicMock()
    visitor.visit_actual_parameter = MagicMock()
    visitor.visit_parameter = MagicMock()
    visitor.visit_multi_statement = MagicMock()
    visitor.visit_statement = MagicMock()
    visitor.visit_masked = MagicMock()
    visitor.visit_mask = MagicMock()
    visitor.visit_shifted_sm_reg = MagicMock()
    visitor.visit_assignment = MagicMock()
    visitor.visit_read = MagicMock()
    visitor.visit_write = MagicMock()
    visitor.visit_broadcast = MagicMock()
    visitor.visit_rsp16_assignment = MagicMock()
    visitor.visit_rsp256_assignment = MagicMock()
    visitor.visit_rsp2k_assignment = MagicMock()
    visitor.visit_rsp32k_assignment = MagicMock()
    visitor.visit_binary_expr = MagicMock()
    visitor.visit_unary_expr = MagicMock()
    visitor.visit_unary_src = MagicMock()
    visitor.visit_unary_sb = MagicMock()
    visitor.visit_sb_expr = MagicMock()
    visitor.visit_rn_reg = MagicMock()
    visitor.visit_sm_reg = MagicMock()
    visitor.visit_rl_expr = MagicMock()
    visitor.visit_rsp16_expr = MagicMock()
    visitor.visit_rsp256_expr = MagicMock()
    visitor.visit_rsp2k_expr = MagicMock()
    visitor.visit_rsp32k_expr = MagicMock()
    visitor.visit_bit_expr = MagicMock()
    visitor.visit_src_expr = MagicMock()
    visitor.visit_broadcast_expr = MagicMock()
    visitor.visit_special = MagicMock()
    visitor.visit_assign_op = MagicMock()
    visitor.visit_binop = MagicMock()
    visitor.visit_unary_op = MagicMock()
    return visitor


def identity_fn(x: Any) -> Any:
    return x


def stub_transformer():
    transformer = BLEIRTransformer()
    transformer.transform_snippet = MagicMock()
    transformer.transform_snippet.side_effect = identity_fn
    transformer.transform_example = MagicMock()
    transformer.transform_example.side_effect = identity_fn
    transformer.transform_value_parameter = MagicMock()
    transformer.transform_value_parameter.side_effect = identity_fn
    transformer.transform_fragment_caller_call = MagicMock()
    transformer.transform_fragment_caller_call.side_effect = identity_fn
    transformer.transform_fragment_caller = MagicMock()
    transformer.transform_fragment_caller.side_effect = identity_fn
    transformer.transform_allocated_register = MagicMock()
    transformer.transform_allocated_register.side_effect = identity_fn
    transformer.transform_fragment = MagicMock()
    transformer.transform_fragment.side_effect = identity_fn
    transformer.transform_actual_parameter = MagicMock()
    transformer.transform_actual_parameter.side_effect = identity_fn
    transformer.transform_parameter = MagicMock()
    transformer.transform_parameter.side_effect = identity_fn
    transformer.transform_multi_statement = MagicMock()
    transformer.transform_multi_statement.side_effect = identity_fn
    transformer.transform_statement = MagicMock()
    transformer.transform_statement.side_effect = identity_fn
    transformer.transform_masked = MagicMock()
    transformer.transform_masked.side_effect = identity_fn
    transformer.transform_mask = MagicMock()
    transformer.transform_mask.side_effect = identity_fn
    transformer.transform_shifted_sm_reg = MagicMock()
    transformer.transform_shifted_sm_reg.side_effect = identity_fn
    transformer.transform_assignment = MagicMock()
    transformer.transform_assignment.side_effect = identity_fn
    transformer.transform_read = MagicMock()
    transformer.transform_read.side_effect = identity_fn
    transformer.transform_write = MagicMock()
    transformer.transform_write.side_effect = identity_fn
    transformer.transform_broadcast = MagicMock()
    transformer.transform_broadcast.side_effect = identity_fn
    transformer.transform_rsp16_assignment = MagicMock()
    transformer.transform_rsp16_assignment.side_effect = identity_fn
    transformer.transform_rsp256_assignment = MagicMock()
    transformer.transform_rsp256_assignment.side_effect = identity_fn
    transformer.transform_rsp2k_assignment = MagicMock()
    transformer.transform_rsp2k_assignment.side_effect = identity_fn
    transformer.transform_rsp32k_assignment = MagicMock()
    transformer.transform_rsp32k_assignment.side_effect = identity_fn
    transformer.transform_binary_expr = MagicMock()
    transformer.transform_binary_expr.side_effect = identity_fn
    transformer.transform_unary_expr = MagicMock()
    transformer.transform_unary_expr.side_effect = identity_fn
    transformer.transform_unary_src = MagicMock()
    transformer.transform_unary_src.side_effect = identity_fn
    transformer.transform_unary_sb = MagicMock()
    transformer.transform_unary_sb.side_effect = identity_fn
    transformer.transform_sb_expr = MagicMock()
    transformer.transform_sb_expr.side_effect = identity_fn
    transformer.transform_rn_reg = MagicMock()
    transformer.transform_rn_reg.side_effect = identity_fn
    transformer.transform_sm_reg = MagicMock()
    transformer.transform_sm_reg.side_effect = identity_fn
    transformer.transform_rl_expr = MagicMock()
    transformer.transform_rl_expr.side_effect = identity_fn
    transformer.transform_rsp16_expr = MagicMock()
    transformer.transform_rsp16_expr.side_effect = identity_fn
    transformer.transform_rsp256_expr = MagicMock()
    transformer.transform_rsp256_expr.side_effect = identity_fn
    transformer.transform_rsp2k_expr = MagicMock()
    transformer.transform_rsp2k_expr.side_effect = identity_fn
    transformer.transform_rsp32k_expr = MagicMock()
    transformer.transform_rsp32k_expr.side_effect = identity_fn
    transformer.transform_bit_expr = MagicMock()
    transformer.transform_bit_expr.side_effect = identity_fn
    transformer.transform_src_expr = MagicMock()
    transformer.transform_src_expr.side_effect = identity_fn
    transformer.transform_broadcast_expr = MagicMock()
    transformer.transform_broadcast_expr.side_effect = identity_fn
    transformer.transform_special = MagicMock()
    transformer.transform_special.side_effect = identity_fn
    transformer.transform_assign_op = MagicMock()
    transformer.transform_assign_op.side_effect = identity_fn
    transformer.transform_binop = MagicMock()
    transformer.transform_binop.side_effect = identity_fn
    transformer.transform_unary_op = MagicMock()
    transformer.transform_unary_op.side_effect = identity_fn
    return transformer


def stub_listener():
    listener = BLEIRListener()
    listener.enter_snippet = MagicMock()
    listener.exit_snippet = MagicMock()
    listener.enter_example = MagicMock()
    listener.exit_example = MagicMock()
    listener.enter_value_parameter = MagicMock()
    listener.exit_value_parameter = MagicMock()
    listener.enter_fragment_caller_call = MagicMock()
    listener.exit_fragment_caller_call = MagicMock()
    listener.enter_fragment_caller = MagicMock()
    listener.exit_fragment_caller = MagicMock()
    listener.enter_allocated_register = MagicMock()
    listener.exit_allocated_register = MagicMock()
    listener.enter_fragment = MagicMock()
    listener.exit_fragment = MagicMock()
    listener.enter_actual_parameter = MagicMock()
    listener.exit_actual_parameter = MagicMock()
    listener.enter_parameter = MagicMock()
    listener.exit_parameter = MagicMock()
    listener.enter_multi_statement = MagicMock()
    listener.exit_multi_statement = MagicMock()
    listener.enter_statement = MagicMock()
    listener.exit_statement = MagicMock()
    listener.enter_masked = MagicMock()
    listener.exit_masked = MagicMock()
    listener.enter_mask = MagicMock()
    listener.exit_mask = MagicMock()
    listener.enter_shifted_sm_reg = MagicMock()
    listener.exit_shifted_sm_reg = MagicMock()
    listener.enter_assignment = MagicMock()
    listener.exit_assignment = MagicMock()
    listener.enter_read = MagicMock()
    listener.exit_read = MagicMock()
    listener.enter_write = MagicMock()
    listener.exit_write = MagicMock()
    listener.enter_broadcast = MagicMock()
    listener.exit_broadcast = MagicMock()
    listener.enter_rsp16_assignment = MagicMock()
    listener.exit_rsp16_assignment = MagicMock()
    listener.enter_rsp256_assignment = MagicMock()
    listener.exit_rsp256_assignment = MagicMock()
    listener.enter_rsp2k_assignment = MagicMock()
    listener.exit_rsp2k_assignment = MagicMock()
    listener.enter_rsp32k_assignment = MagicMock()
    listener.exit_rsp32k_assignment = MagicMock()
    listener.enter_binary_expr = MagicMock()
    listener.exit_binary_expr = MagicMock()
    listener.enter_unary_expr = MagicMock()
    listener.exit_unary_expr = MagicMock()
    listener.enter_unary_src = MagicMock()
    listener.exit_unary_src = MagicMock()
    listener.enter_unary_sb = MagicMock()
    listener.exit_unary_sb = MagicMock()
    listener.enter_sb_expr = MagicMock()
    listener.exit_sb_expr = MagicMock()
    listener.enter_rn_reg = MagicMock()
    listener.exit_rn_reg = MagicMock()
    listener.enter_sm_reg = MagicMock()
    listener.exit_sm_reg = MagicMock()
    listener.enter_rl_expr = MagicMock()
    listener.exit_rl_expr = MagicMock()
    listener.enter_rsp16_expr = MagicMock()
    listener.exit_rsp16_expr = MagicMock()
    listener.enter_rsp256_expr = MagicMock()
    listener.exit_rsp256_expr = MagicMock()
    listener.enter_rsp2k_expr = MagicMock()
    listener.exit_rsp2k_expr = MagicMock()
    listener.enter_rsp32k_expr = MagicMock()
    listener.exit_rsp32k_expr = MagicMock()
    listener.enter_bit_expr = MagicMock()
    listener.exit_bit_expr = MagicMock()
    listener.enter_src_expr = MagicMock()
    listener.exit_src_expr = MagicMock()
    listener.enter_broadcast_expr = MagicMock()
    listener.exit_broadcast_expr = MagicMock()
    listener.enter_special = MagicMock()
    listener.exit_special = MagicMock()
    listener.enter_assign_op = MagicMock()
    listener.exit_assign_op = MagicMock()
    listener.enter_binop = MagicMock()
    listener.exit_binop = MagicMock()
    listener.enter_unary_op = MagicMock()
    listener.exit_unary_op = MagicMock()
    return listener


def test_bleir_listener_and_transformer():
    walker = BLEIRWalker()

    unary_op = UNARY_OP.NEGATE

    listener = stub_listener()
    walker.walk(listener, unary_op)
    listener.enter_unary_op.assert_called_once_with(unary_op)
    listener.exit_unary_op.assert_called_once_with(unary_op)

    visitor = stub_visitor()
    walker.walk(visitor, unary_op)
    visitor.visit_unary_op.assert_called_once_with(unary_op)

    transformer = stub_transformer()
    walker.walk(transformer, unary_op)
    transformer.transform_unary_op.assert_called_once_with(unary_op)

    binop = BINOP.AND

    listener = stub_listener()
    walker.walk(listener, binop)
    listener.enter_binop.assert_called_once_with(binop)
    listener.exit_binop.assert_called_once_with(binop)

    visitor = stub_visitor()
    walker.walk(visitor, binop)
    visitor.visit_binop.assert_called_once_with(binop)

    transformer = stub_transformer()
    walker.walk(transformer, binop)
    transformer.transform_binop.assert_called_once_with(binop)

    assign_op = ASSIGN_OP.EQ

    listener = stub_listener()
    walker.walk(listener, assign_op)
    listener.enter_assign_op.assert_called_once_with(assign_op)
    listener.exit_assign_op.assert_called_once_with(assign_op)

    visitor = stub_visitor()
    walker.walk(visitor, assign_op)
    visitor.visit_assign_op.assert_called_once_with(assign_op)

    transformer = stub_transformer()
    walker.walk(transformer, assign_op)
    transformer.transform_assign_op.assert_called_once_with(assign_op)

    special = SPECIAL.NOOP

    listener = stub_listener()
    walker.walk(listener, special)
    listener.enter_special.assert_called_once_with(special)
    listener.exit_special.assert_called_once_with(special)

    visitor = stub_visitor()
    walker.walk(visitor, special)
    visitor.visit_special.assert_called_once_with(special)

    transformer = stub_transformer()
    walker.walk(transformer, special)
    transformer.transform_special.assert_called_once_with(special)

    broadcast_expr = BROADCAST_EXPR.RSP16

    listener = stub_listener()
    walker.walk(listener, broadcast_expr)
    listener.enter_broadcast_expr.assert_called_once_with(broadcast_expr)
    listener.exit_broadcast_expr.assert_called_once_with(broadcast_expr)

    visitor = stub_visitor()
    walker.walk(visitor, broadcast_expr)
    visitor.visit_broadcast_expr.assert_called_once_with(broadcast_expr)

    transformer = stub_transformer()
    walker.walk(transformer, broadcast_expr)
    transformer.transform_broadcast_expr.assert_called_once_with(broadcast_expr)

    src_expr = SRC_EXPR.INV_SRL

    listener = stub_listener()
    walker.walk(listener, src_expr)
    listener.enter_src_expr.assert_called_once_with(src_expr)
    listener.exit_src_expr.assert_called_once_with(src_expr)

    visitor = stub_visitor()
    walker.walk(visitor, src_expr)
    visitor.visit_src_expr.assert_called_once_with(src_expr)

    transformer = stub_transformer()
    walker.walk(transformer, src_expr)
    transformer.transform_src_expr.assert_called_once_with(src_expr)

    bit_expr = BIT_EXPR.ONE

    listener = stub_listener()
    walker.walk(listener, bit_expr)
    listener.enter_bit_expr.assert_called_once_with(bit_expr)
    listener.exit_bit_expr.assert_called_once_with(bit_expr)

    visitor = stub_visitor()
    walker.walk(visitor, bit_expr)
    visitor.visit_bit_expr.assert_called_once_with(bit_expr)

    transformer = stub_transformer()
    walker.walk(transformer, bit_expr)
    transformer.transform_bit_expr.assert_called_once_with(bit_expr)

    rsp32k_expr = RSP32K_EXPR.RSP32K

    listener = stub_listener()
    walker.walk(listener, rsp32k_expr)
    listener.enter_rsp32k_expr.assert_called_once_with(rsp32k_expr)
    listener.exit_rsp32k_expr.assert_called_once_with(rsp32k_expr)

    visitor = stub_visitor()
    walker.walk(visitor, rsp32k_expr)
    visitor.visit_rsp32k_expr.assert_called_once_with(rsp32k_expr)

    transformer = stub_transformer()
    walker.walk(transformer, rsp32k_expr)
    transformer.transform_rsp32k_expr.assert_called_once_with(rsp32k_expr)

    rsp2k_expr = RSP2K_EXPR.RSP2K

    listener = stub_listener()
    walker.walk(listener, rsp2k_expr)
    listener.enter_rsp2k_expr.assert_called_once_with(rsp2k_expr)
    listener.exit_rsp2k_expr.assert_called_once_with(rsp2k_expr)

    visitor = stub_visitor()
    walker.walk(visitor, rsp2k_expr)
    visitor.visit_rsp2k_expr.assert_called_once_with(rsp2k_expr)

    transformer = stub_transformer()
    walker.walk(transformer, rsp2k_expr)
    transformer.transform_rsp2k_expr.assert_called_once_with(rsp2k_expr)

    rsp256_expr = RSP256_EXPR.RSP256

    listener = stub_listener()
    walker.walk(listener, rsp256_expr)
    listener.enter_rsp256_expr.assert_called_once_with(rsp256_expr)
    listener.exit_rsp256_expr.assert_called_once_with(rsp256_expr)

    visitor = stub_visitor()
    walker.walk(visitor, rsp256_expr)
    visitor.visit_rsp256_expr.assert_called_once_with(rsp256_expr)

    transformer = stub_transformer()
    walker.walk(transformer, rsp256_expr)
    transformer.transform_rsp256_expr.assert_called_once_with(rsp256_expr)

    rsp16_expr = RSP16_EXPR.RSP16

    listener = stub_listener()
    walker.walk(listener, rsp16_expr)
    listener.enter_rsp16_expr.assert_called_once_with(rsp16_expr)
    listener.exit_rsp16_expr.assert_called_once_with(rsp16_expr)

    visitor = stub_visitor()
    walker.walk(visitor, rsp16_expr)
    visitor.visit_rsp16_expr.assert_called_once_with(rsp16_expr)

    transformer = stub_transformer()
    walker.walk(transformer, rsp16_expr)
    transformer.transform_rsp16_expr.assert_called_once_with(rsp16_expr)

    rl_expr = RL_EXPR.RL

    listener = stub_listener()
    walker.walk(listener, rl_expr)
    listener.enter_rl_expr.assert_called_once_with(rl_expr)
    listener.exit_rl_expr.assert_called_once_with(rl_expr)

    visitor = stub_visitor()
    walker.walk(visitor, rl_expr)
    visitor.visit_rl_expr.assert_called_once_with(rl_expr)

    transformer = stub_transformer()
    walker.walk(transformer, rl_expr)
    transformer.transform_rl_expr.assert_called_once_with(rl_expr)

    fs_rp = SM_REG("fs")

    listener = stub_listener()
    walker.walk(listener, fs_rp)
    listener.enter_sm_reg.assert_called_once_with(fs_rp)
    listener.exit_sm_reg.assert_called_once_with(fs_rp)

    visitor = stub_visitor()
    walker.walk(visitor, fs_rp)
    visitor.visit_sm_reg.assert_called_once_with(fs_rp)

    transformer = stub_transformer()
    walker.walk(transformer, fs_rp)
    transformer.transform_sm_reg.assert_called_once_with(fs_rp)

    lvr_rp = RN_REG("lvr")

    listener = stub_listener()
    walker.walk(listener, lvr_rp)
    listener.enter_rn_reg.assert_called_once_with(lvr_rp)
    listener.exit_rn_reg.assert_called_once_with(lvr_rp)

    visitor = stub_visitor()
    walker.walk(visitor, lvr_rp)
    visitor.visit_rn_reg.assert_called_once_with(lvr_rp)

    transformer = stub_transformer()
    walker.walk(transformer, lvr_rp)
    transformer.transform_rn_reg.assert_called_once_with(lvr_rp)

    rvr_rp = RN_REG("rvr")
    r2vr_rp = RN_REG("r2vr")

    listener = stub_listener()
    walker.walk(listener, SB[lvr_rp])
    listener.enter_sb_expr.assert_called_once_with(SB[lvr_rp])
    listener.exit_sb_expr.assert_called_once_with(SB[lvr_rp])
    listener.enter_rn_reg.assert_called_once_with(lvr_rp)
    listener.exit_rn_reg.assert_called_once_with(lvr_rp)

    visitor = stub_visitor()
    walker.walk(visitor, SB[lvr_rp])
    visitor.visit_sb_expr.assert_called_once_with(SB[lvr_rp])
    visitor.visit_rn_reg.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, SB[lvr_rp])
    transformer.transform_sb_expr.assert_called_once_with(SB[lvr_rp])
    transformer.transform_rn_reg.assert_called_once_with(lvr_rp)

    listener = stub_listener()
    walker.walk(listener, SB[lvr_rp, rvr_rp])
    listener.enter_sb_expr.assert_called_once_with(SB[lvr_rp, rvr_rp])
    listener.exit_sb_expr.assert_called_once_with(SB[lvr_rp, rvr_rp])
    assert listener.enter_rn_reg.call_args_list == [
        call(lvr_rp),
        call(rvr_rp),
    ]
    assert listener.exit_rn_reg.call_args_list == [
        call(lvr_rp),
        call(rvr_rp),
    ]

    visitor = stub_visitor()
    walker.walk(visitor, SB[lvr_rp, rvr_rp])
    visitor.visit_sb_expr.assert_called_once_with(SB[lvr_rp, rvr_rp])
    visitor.visit_rn_reg.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, SB[lvr_rp, rvr_rp])
    transformer.transform_sb_expr.assert_called_once_with(SB[lvr_rp, rvr_rp])
    assert transformer.transform_rn_reg.call_args_list == [
        call(lvr_rp),
        call(rvr_rp),
    ]

    listener = stub_listener()
    walker.walk(listener, SB[lvr_rp, rvr_rp, r2vr_rp])
    listener.enter_sb_expr.assert_called_once_with(SB[lvr_rp, rvr_rp, r2vr_rp])
    listener.exit_sb_expr.assert_called_once_with(SB[lvr_rp, rvr_rp, r2vr_rp])
    assert listener.enter_rn_reg.call_args_list == [
        call(lvr_rp),
        call(rvr_rp),
        call(r2vr_rp),
    ]
    assert listener.exit_rn_reg.call_args_list == [
        call(lvr_rp),
        call(rvr_rp),
        call(r2vr_rp),
    ]

    visitor = stub_visitor()
    walker.walk(visitor, SB[lvr_rp, rvr_rp, r2vr_rp])
    visitor.visit_sb_expr.assert_called_once_with(SB[lvr_rp, rvr_rp, r2vr_rp])
    visitor.visit_rn_reg.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, SB[lvr_rp, rvr_rp, r2vr_rp])
    transformer.transform_sb_expr.assert_called_once_with(SB[lvr_rp, rvr_rp, r2vr_rp])
    assert transformer.transform_rn_reg.call_args_list == [
        call(lvr_rp),
        call(rvr_rp),
        call(r2vr_rp),
    ]

    unary_sb = UNARY_SB(
        expression=SB[lvr_rp],
        operator=UNARY_OP.NEGATE,
    )

    listener = stub_listener()
    walker.walk(listener, unary_sb)
    listener.enter_unary_sb.assert_called_once_with(unary_sb)
    listener.exit_unary_sb.assert_called_once_with(unary_sb)
    listener.enter_sb_expr.assert_called_once_with(SB[lvr_rp])
    listener.exit_sb_expr.assert_called_once_with(SB[lvr_rp])
    listener.enter_unary_op.assert_called_once_with(UNARY_OP.NEGATE)
    listener.exit_unary_op.assert_called_once_with(UNARY_OP.NEGATE)

    visitor = stub_visitor()
    walker.walk(visitor, unary_sb)
    visitor.visit_unary_sb.assert_called_once_with(unary_sb)
    visitor.visit_sb_expr.assert_not_called()
    visitor.visit_unary_op.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, unary_sb)
    transformer.transform_unary_sb.assert_called_once_with(unary_sb)
    transformer.transform_sb_expr.assert_called_once_with(SB[lvr_rp])
    transformer.transform_unary_op.assert_called_once_with(UNARY_OP.NEGATE)

    unary_sb = UNARY_SB(
        expression=SB[lvr_rp],
        operator=None,
    )

    listener = stub_listener()
    walker.walk(listener, unary_sb)
    listener.enter_unary_sb.assert_called_once_with(unary_sb)
    listener.exit_unary_sb.assert_called_once_with(unary_sb)
    listener.enter_sb_expr.assert_called_once_with(SB[lvr_rp])
    listener.exit_sb_expr.assert_called_once_with(SB[lvr_rp])

    visitor = stub_visitor()
    walker.walk(visitor, unary_sb)
    visitor.visit_unary_sb.assert_called_once_with(unary_sb)
    visitor.visit_sb_expr.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, unary_sb)
    transformer.transform_unary_sb.assert_called_once_with(unary_sb)
    transformer.transform_sb_expr.assert_called_once_with(SB[lvr_rp])

    unary_src = UNARY_SRC(
        expression=SRC_EXPR.GGL,
        operator=UNARY_OP.NEGATE,
    )

    listener = stub_listener()
    walker.walk(listener, unary_src)
    listener.enter_unary_src.assert_called_once_with(unary_src)
    listener.exit_unary_src.assert_called_once_with(unary_src)
    listener.enter_src_expr.assert_called_once_with(SRC_EXPR.GGL)
    listener.exit_src_expr.assert_called_once_with(SRC_EXPR.GGL)
    listener.enter_unary_op.assert_called_once_with(UNARY_OP.NEGATE)
    listener.exit_unary_op.assert_called_once_with(UNARY_OP.NEGATE)

    visitor = stub_visitor()
    walker.walk(visitor, unary_src)
    visitor.visit_unary_src.assert_called_once_with(unary_src)
    visitor.visit_src_expr.assert_not_called()
    visitor.visit_unary_op.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, unary_src)
    transformer.transform_unary_src.assert_called_once_with(unary_src)
    transformer.transform_src_expr.assert_called_once_with(SRC_EXPR.GGL)
    transformer.transform_unary_op.assert_called_once_with(UNARY_OP.NEGATE)

    unary_src = UNARY_SRC(
        expression=SRC_EXPR.GGL,
        operator=None,
    )

    listener = stub_listener()
    walker.walk(listener, unary_src)
    listener.enter_unary_src.assert_called_once_with(unary_src)
    listener.exit_unary_src.assert_called_once_with(unary_src)
    listener.enter_src_expr.assert_called_once_with(SRC_EXPR.GGL)
    listener.exit_src_expr.assert_called_once_with(SRC_EXPR.GGL)

    visitor = stub_visitor()
    walker.walk(visitor, unary_src)
    visitor.visit_unary_src.assert_called_once_with(unary_src)
    visitor.visit_src_expr.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, unary_src)
    transformer.transform_unary_src.assert_called_once_with(unary_src)
    transformer.transform_src_expr.assert_called_once_with(SRC_EXPR.GGL)

    unary_expr = UNARY_EXPR(
        expression=unary_sb,
    )

    listener = stub_listener()
    walker.walk(listener, unary_expr)
    listener.enter_unary_expr.assert_called_once_with(unary_expr)
    listener.exit_unary_expr.assert_called_once_with(unary_expr)
    listener.enter_unary_sb.assert_called_once_with(unary_sb)
    listener.exit_unary_sb.assert_called_once_with(unary_sb)
    listener.enter_unary_src.assert_not_called()
    listener.exit_unary_src.assert_not_called()
    listener.enter_bit_expr.assert_not_called()
    listener.exit_bit_expr.assert_not_called()

    visitor = stub_visitor()
    walker.walk(visitor, unary_expr)
    visitor.visit_unary_expr.assert_called_once_with(unary_expr)
    visitor.visit_unary_sb.assert_not_called()
    visitor.visit_unary_src.assert_not_called()
    visitor.visit_bit_expr.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, unary_expr)
    transformer.transform_unary_expr.assert_called_once_with(unary_expr)
    transformer.transform_unary_sb.assert_called_once_with(unary_sb)
    transformer.transform_unary_src.assert_not_called()
    transformer.transform_bit_expr.assert_not_called()

    unary_expr = UNARY_EXPR(
        expression=unary_src,
    )

    listener = stub_listener()
    walker.walk(listener, unary_expr)
    listener.enter_unary_expr.assert_called_once_with(unary_expr)
    listener.exit_unary_expr.assert_called_once_with(unary_expr)
    listener.enter_unary_sb.assert_not_called()
    listener.exit_unary_sb.assert_not_called()
    listener.enter_unary_src.assert_called_once_with(unary_src)
    listener.exit_unary_src.assert_called_once_with(unary_src)
    listener.enter_bit_expr.assert_not_called()
    listener.exit_bit_expr.assert_not_called()

    visitor = stub_visitor()
    walker.walk(visitor, unary_expr)
    visitor.visit_unary_expr.assert_called_once_with(unary_expr)
    visitor.visit_unary_sb.assert_not_called()
    visitor.visit_unary_src.assert_not_called()
    visitor.visit_bit_expr.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, unary_expr)
    transformer.transform_unary_expr.assert_called_once_with(unary_expr)
    transformer.transform_unary_sb.assert_not_called()
    transformer.transform_unary_src.assert_called_once_with(unary_src)
    transformer.transform_bit_expr.assert_not_called()

    unary_expr = UNARY_EXPR(
        expression=bit_expr,
    )

    listener = stub_listener()
    walker.walk(listener, unary_expr)
    listener.enter_unary_expr.assert_called_once_with(unary_expr)
    listener.exit_unary_expr.assert_called_once_with(unary_expr)
    listener.enter_unary_sb.assert_not_called()
    listener.exit_unary_sb.assert_not_called()
    listener.enter_unary_src.assert_not_called()
    listener.exit_unary_src.assert_not_called()
    listener.enter_bit_expr.assert_called_once_with(bit_expr)
    listener.exit_bit_expr.assert_called_once_with(bit_expr)

    visitor = stub_visitor()
    walker.walk(visitor, unary_expr)
    visitor.visit_unary_expr.assert_called_once_with(unary_expr)
    visitor.visit_unary_sb.assert_not_called()
    visitor.visit_unary_src.assert_not_called()
    visitor.visit_bit_expr.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, unary_expr)
    transformer.transform_unary_expr.assert_called_once_with(unary_expr)
    transformer.transform_unary_sb.assert_not_called()
    transformer.transform_unary_src.assert_not_called()
    transformer.transform_bit_expr.assert_called_once_with(bit_expr)

    binary_expr = BINARY_EXPR(
        operator=BINOP.XOR,
        left_operand=unary_sb,
        right_operand=unary_src,
    )

    listener = stub_listener()
    walker.walk(listener, binary_expr)
    listener.enter_binary_expr.assert_called_once_with(binary_expr)
    listener.exit_binary_expr.assert_called_once_with(binary_expr)
    listener.enter_binop.assert_called_once_with(BINOP.XOR)
    listener.exit_binop.assert_called_once_with(BINOP.XOR)
    listener.enter_unary_sb.assert_called_once_with(unary_sb)
    listener.exit_unary_sb.assert_called_once_with(unary_sb)
    listener.enter_unary_src.assert_called_once_with(unary_src)
    listener.exit_unary_src.assert_called_once_with(unary_src)

    visitor = stub_visitor()
    walker.walk(visitor, binary_expr)
    visitor.visit_binary_expr.assert_called_once_with(binary_expr)
    visitor.visit_binop.assert_not_called()
    visitor.visit_unary_sb.assert_not_called()
    visitor.visit_unary_src.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, binary_expr)
    transformer.transform_binary_expr.assert_called_once_with(binary_expr)
    transformer.transform_binop.assert_called_once_with(BINOP.XOR)
    transformer.transform_unary_sb.assert_called_once_with(unary_sb)
    transformer.transform_unary_src.assert_called_once_with(unary_src)

    rsp32k_assignment = RSP32K_ASSIGNMENT(rvalue=RSP32K_RVALUE.RSP2K)

    listener = stub_listener()
    walker.walk(listener, rsp32k_assignment)
    listener.enter_rsp32k_assignment.assert_called_once_with(rsp32k_assignment)
    listener.exit_rsp32k_assignment.assert_called_once_with(rsp32k_assignment)

    visitor = stub_visitor()
    walker.walk(visitor, rsp32k_assignment)
    visitor.visit_rsp32k_assignment.assert_called_once_with(rsp32k_assignment)

    transformer = stub_transformer()
    walker.walk(transformer, rsp32k_assignment)
    transformer.transform_rsp32k_assignment.assert_called_once_with(rsp32k_assignment)

    rsp2k_assignment = RSP2K_ASSIGNMENT(rvalue=RSP2K_RVALUE.RSP256)

    listener = stub_listener()
    walker.walk(listener, rsp2k_assignment)
    listener.enter_rsp2k_assignment.assert_called_once_with(rsp2k_assignment)
    listener.exit_rsp2k_assignment.assert_called_once_with(rsp2k_assignment)

    visitor = stub_visitor()
    walker.walk(visitor, rsp2k_assignment)
    visitor.visit_rsp2k_assignment.assert_called_once_with(rsp2k_assignment)

    transformer = stub_transformer()
    walker.walk(transformer, rsp2k_assignment)
    transformer.transform_rsp2k_assignment.assert_called_once_with(rsp2k_assignment)

    rsp2k_assignment = RSP2K_ASSIGNMENT(rvalue=RSP2K_RVALUE.RSP32K)

    listener = stub_listener()
    walker.walk(listener, rsp2k_assignment)
    listener.enter_rsp2k_assignment.assert_called_once_with(rsp2k_assignment)
    listener.exit_rsp2k_assignment.assert_called_once_with(rsp2k_assignment)

    visitor = stub_visitor()
    walker.walk(visitor, rsp2k_assignment)
    visitor.visit_rsp2k_assignment.assert_called_once_with(rsp2k_assignment)

    transformer = stub_transformer()
    walker.walk(transformer, rsp2k_assignment)
    transformer.transform_rsp2k_assignment.assert_called_once_with(rsp2k_assignment)

    rsp256_assignment = RSP256_ASSIGNMENT(rvalue=RSP256_RVALUE.RSP16)

    listener = stub_listener()
    walker.walk(listener, rsp256_assignment)
    listener.enter_rsp256_assignment.assert_called_once_with(rsp256_assignment)
    listener.exit_rsp256_assignment.assert_called_once_with(rsp256_assignment)

    visitor = stub_visitor()
    walker.walk(visitor, rsp256_assignment)
    visitor.visit_rsp256_assignment.assert_called_once_with(rsp256_assignment)

    transformer = stub_transformer()
    walker.walk(transformer, rsp256_assignment)
    transformer.transform_rsp256_assignment.assert_called_once_with(rsp256_assignment)

    rsp256_assignment = RSP256_ASSIGNMENT(rvalue=RSP256_RVALUE.RSP2K)

    listener = stub_listener()
    walker.walk(listener, rsp256_assignment)
    listener.enter_rsp256_assignment.assert_called_once_with(rsp256_assignment)
    listener.exit_rsp256_assignment.assert_called_once_with(rsp256_assignment)

    visitor = stub_visitor()
    walker.walk(visitor, rsp256_assignment)
    visitor.visit_rsp256_assignment.assert_called_once_with(rsp256_assignment)

    transformer = stub_transformer()
    walker.walk(transformer, rsp256_assignment)
    transformer.transform_rsp256_assignment.assert_called_once_with(rsp256_assignment)

    rsp16_assignment = RSP16_ASSIGNMENT(rvalue=RSP16_RVALUE.RSP256)

    listener = stub_listener()
    walker.walk(listener, rsp16_assignment)
    listener.enter_rsp16_assignment.assert_called_once_with(rsp16_assignment)
    listener.exit_rsp16_assignment.assert_called_once_with(rsp16_assignment)

    visitor = stub_visitor()
    walker.walk(visitor, rsp16_assignment)
    visitor.visit_rsp16_assignment.assert_called_once_with(rsp16_assignment)

    transformer = stub_transformer()
    walker.walk(transformer, rsp16_assignment)
    transformer.transform_rsp16_assignment.assert_called_once_with(rsp16_assignment)

    broadcast = BROADCAST(
        lvalue=BROADCAST_EXPR.RSP16,
    )

    listener = stub_listener()
    walker.walk(listener, broadcast)
    listener.enter_broadcast.assert_called_once_with(broadcast)
    listener.exit_broadcast.assert_called_once_with(broadcast)
    listener.enter_broadcast_expr.assert_called_once_with(BROADCAST_EXPR.RSP16)
    listener.exit_broadcast_expr.assert_called_once_with(BROADCAST_EXPR.RSP16)

    visitor = stub_visitor()
    walker.walk(visitor, broadcast)
    visitor.visit_broadcast.assert_called_once_with(broadcast)
    visitor.visit_broadcast_expr.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, broadcast)
    transformer.transform_broadcast.assert_called_once_with(broadcast)
    transformer.transform_broadcast_expr.assert_called_once_with(BROADCAST_EXPR.RSP16)

    sb_expr = SB[lvr_rp]

    write = WRITE(
        operator=ASSIGN_OP.COND_EQ,
        lvalue=sb_expr,
        rvalue=unary_src,
    )

    listener = stub_listener()
    walker.walk(listener, write)
    listener.enter_write.assert_called_once_with(write)
    listener.exit_write.assert_called_once_with(write)
    listener.enter_assign_op.assert_called_once_with(ASSIGN_OP.COND_EQ)
    listener.exit_assign_op.assert_called_once_with(ASSIGN_OP.COND_EQ)
    listener.enter_sb_expr.assert_called_once_with(sb_expr)
    listener.exit_sb_expr.assert_called_once_with(sb_expr)
    listener.enter_unary_src.assert_called_once_with(unary_src)
    listener.exit_unary_src.assert_called_once_with(unary_src)

    visitor = stub_visitor()
    walker.walk(visitor, write)
    visitor.visit_write.assert_called_once_with(write)
    visitor.visit_assign_op.assert_not_called()
    visitor.visit_sb_expr.assert_not_called()
    visitor.visit_unary_src.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, write)
    transformer.transform_write.assert_called_once_with(write)
    transformer.transform_assign_op.assert_called_once_with(ASSIGN_OP.COND_EQ)
    transformer.transform_sb_expr.assert_called_once_with(sb_expr)
    transformer.transform_unary_src.assert_called_once_with(unary_src)

    read = READ(
        operator=ASSIGN_OP.OR_EQ,
        rvalue=unary_expr,
    )

    listener = stub_listener()
    walker.walk(listener, read)
    listener.enter_read.assert_called_once_with(read)
    listener.exit_read.assert_called_once_with(read)
    listener.enter_assign_op.assert_called_once_with(ASSIGN_OP.OR_EQ)
    listener.exit_assign_op.assert_called_once_with(ASSIGN_OP.OR_EQ)
    listener.enter_unary_expr.assert_called_once_with(unary_expr)
    listener.exit_unary_expr.assert_called_once_with(unary_expr)
    listener.enter_binary_expr.assert_not_called()
    listener.exit_binary_expr.assert_not_called()

    visitor = stub_visitor()
    walker.walk(visitor, read)
    visitor.visit_read.assert_called_once_with(read)
    visitor.visit_assign_op.assert_not_called()
    visitor.visit_unary_expr.assert_not_called()
    visitor.visit_binary_expr.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, read)
    transformer.transform_read.assert_called_once_with(read)
    transformer.transform_assign_op.assert_called_once_with(ASSIGN_OP.OR_EQ)
    transformer.transform_unary_expr.assert_called_once_with(unary_expr)
    transformer.transform_binary_expr.assert_not_called()

    read = READ(
        operator=ASSIGN_OP.OR_EQ,
        rvalue=binary_expr,
    )

    listener = stub_listener()
    walker.walk(listener, read)
    listener.enter_read.assert_called_once_with(read)
    listener.exit_read.assert_called_once_with(read)
    listener.enter_assign_op.assert_called_once_with(ASSIGN_OP.OR_EQ)
    listener.exit_assign_op.assert_called_once_with(ASSIGN_OP.OR_EQ)
    listener.enter_unary_expr.assert_not_called()
    listener.exit_unary_expr.assert_not_called()
    listener.enter_binary_expr.assert_called_once_with(binary_expr)
    listener.exit_binary_expr.assert_called_once_with(binary_expr)

    visitor = stub_visitor()
    walker.walk(visitor, read)
    visitor.visit_read.assert_called_once_with(read)
    visitor.visit_assign_op.assert_not_called()
    visitor.visit_unary_expr.assert_not_called()
    visitor.visit_binary_expr.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, read)
    transformer.transform_read.assert_called_once_with(read)
    transformer.transform_assign_op.assert_called_once_with(ASSIGN_OP.OR_EQ)
    transformer.transform_unary_expr.assert_not_called()
    transformer.transform_binary_expr.assert_called_once_with(binary_expr)

    assignment = ASSIGNMENT(
        operation=read,
    )

    listener = stub_listener()
    walker.walk(listener, assignment)
    listener.enter_assignment.assert_called_once_with(assignment)
    listener.exit_assignment.assert_called_once_with(assignment)
    listener.enter_read.assert_called_once_with(read)
    listener.exit_read.assert_called_once_with(read)
    listener.enter_write.assert_not_called()
    listener.exit_write.assert_not_called()
    listener.enter_broadcast.assert_not_called()
    listener.exit_broadcast.assert_not_called()
    listener.enter_rsp256_assignment.assert_not_called()
    listener.exit_rsp256_assignment.assert_not_called()
    listener.enter_rsp2k_assignment.assert_not_called()
    listener.exit_rsp2k_assignment.assert_not_called()
    listener.enter_rsp32k_assignment.assert_not_called()
    listener.exit_rsp32k_assignment.assert_not_called()

    visitor = stub_visitor()
    walker.walk(visitor, assignment)
    visitor.visit_assignment.assert_called_once_with(assignment)
    visitor.visit_read.assert_not_called()
    visitor.visit_write.assert_not_called()
    visitor.visit_broadcast.assert_not_called()
    visitor.visit_rsp256_assignment.assert_not_called()
    visitor.visit_rsp2k_assignment.assert_not_called()
    visitor.visit_rsp32k_assignment.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, assignment)
    transformer.transform_assignment.assert_called_once_with(assignment)
    transformer.transform_read.assert_called_once_with(read)
    transformer.transform_write.assert_not_called()
    transformer.transform_broadcast.assert_not_called()
    transformer.transform_rsp256_assignment.assert_not_called()
    transformer.transform_rsp2k_assignment.assert_not_called()
    transformer.transform_rsp32k_assignment.assert_not_called()

    assignment = ASSIGNMENT(
        operation=write,
    )

    listener = stub_listener()
    walker.walk(listener, assignment)
    listener.enter_assignment.assert_called_once_with(assignment)
    listener.exit_assignment.assert_called_once_with(assignment)
    listener.enter_read.assert_not_called()
    listener.exit_read.assert_not_called()
    listener.enter_write.assert_called_once_with(write)
    listener.exit_write.assert_called_once_with(write)
    listener.enter_broadcast.assert_not_called()
    listener.exit_broadcast.assert_not_called()
    listener.enter_rsp256_assignment.assert_not_called()
    listener.exit_rsp256_assignment.assert_not_called()
    listener.enter_rsp2k_assignment.assert_not_called()
    listener.exit_rsp2k_assignment.assert_not_called()
    listener.enter_rsp32k_assignment.assert_not_called()
    listener.exit_rsp32k_assignment.assert_not_called()

    visitor = stub_visitor()
    walker.walk(visitor, assignment)
    visitor.visit_assignment.assert_called_once_with(assignment)
    visitor.visit_read.assert_not_called()
    visitor.visit_write.assert_not_called()
    visitor.visit_broadcast.assert_not_called()
    visitor.visit_rsp256_assignment.assert_not_called()
    visitor.visit_rsp2k_assignment.assert_not_called()
    visitor.visit_rsp32k_assignment.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, assignment)
    transformer.transform_assignment.assert_called_once_with(assignment)
    transformer.transform_read.assert_not_called()
    transformer.transform_write.assert_called_once_with(write)
    transformer.transform_broadcast.assert_not_called()
    transformer.transform_rsp256_assignment.assert_not_called()
    transformer.transform_rsp2k_assignment.assert_not_called()
    transformer.transform_rsp32k_assignment.assert_not_called()

    assignment = ASSIGNMENT(
        operation=broadcast,
    )

    listener = stub_listener()
    walker.walk(listener, assignment)
    listener.enter_assignment.assert_called_once_with(assignment)
    listener.exit_assignment.assert_called_once_with(assignment)
    listener.enter_read.assert_not_called()
    listener.exit_read.assert_not_called()
    listener.enter_write.assert_not_called()
    listener.exit_write.assert_not_called()
    listener.enter_broadcast.assert_called_once_with(broadcast)
    listener.exit_broadcast.assert_called_once_with(broadcast)
    listener.enter_rsp256_assignment.assert_not_called()
    listener.exit_rsp256_assignment.assert_not_called()
    listener.enter_rsp2k_assignment.assert_not_called()
    listener.exit_rsp2k_assignment.assert_not_called()
    listener.enter_rsp32k_assignment.assert_not_called()
    listener.exit_rsp32k_assignment.assert_not_called()

    visitor = stub_visitor()
    walker.walk(visitor, assignment)
    visitor.visit_assignment.assert_called_once_with(assignment)
    visitor.visit_read.assert_not_called()
    visitor.visit_write.assert_not_called()
    visitor.visit_broadcast.assert_not_called()
    visitor.visit_rsp256_assignment.assert_not_called()
    visitor.visit_rsp2k_assignment.assert_not_called()
    visitor.visit_rsp32k_assignment.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, assignment)
    transformer.transform_assignment.assert_called_once_with(assignment)
    transformer.transform_read.assert_not_called()
    transformer.transform_write.assert_not_called()
    transformer.transform_broadcast.assert_called_once_with(broadcast)
    transformer.transform_rsp256_assignment.assert_not_called()
    transformer.transform_rsp2k_assignment.assert_not_called()
    transformer.transform_rsp32k_assignment.assert_not_called()

    assignment = ASSIGNMENT(
        operation=rsp256_assignment,
    )

    listener = stub_listener()
    walker.walk(listener, assignment)
    listener.enter_assignment.assert_called_once_with(assignment)
    listener.exit_assignment.assert_called_once_with(assignment)
    listener.enter_read.assert_not_called()
    listener.exit_read.assert_not_called()
    listener.enter_write.assert_not_called()
    listener.exit_write.assert_not_called()
    listener.enter_broadcast.assert_not_called()
    listener.exit_broadcast.assert_not_called()
    listener.enter_rsp256_assignment.assert_called_once_with(rsp256_assignment)
    listener.exit_rsp256_assignment.assert_called_once_with(rsp256_assignment)
    listener.enter_rsp2k_assignment.assert_not_called()
    listener.exit_rsp2k_assignment.assert_not_called()
    listener.enter_rsp32k_assignment.assert_not_called()
    listener.exit_rsp32k_assignment.assert_not_called()

    visitor = stub_visitor()
    walker.walk(visitor, assignment)
    visitor.visit_assignment.assert_called_once_with(assignment)
    visitor.visit_read.assert_not_called()
    visitor.visit_write.assert_not_called()
    visitor.visit_broadcast.assert_not_called()
    visitor.visit_rsp256_assignment.assert_not_called()
    visitor.visit_rsp2k_assignment.assert_not_called()
    visitor.visit_rsp32k_assignment.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, assignment)
    transformer.transform_assignment.assert_called_once_with(assignment)
    transformer.transform_read.assert_not_called()
    transformer.transform_write.assert_not_called()
    transformer.transform_broadcast.assert_not_called()
    transformer.transform_rsp256_assignment.assert_called_once_with(rsp256_assignment)
    transformer.transform_rsp2k_assignment.assert_not_called()
    transformer.transform_rsp32k_assignment.assert_not_called()

    assignment = ASSIGNMENT(
        operation=rsp2k_assignment,
    )

    listener = stub_listener()
    walker.walk(listener, assignment)
    listener.enter_assignment.assert_called_once_with(assignment)
    listener.exit_assignment.assert_called_once_with(assignment)
    listener.enter_read.assert_not_called()
    listener.exit_read.assert_not_called()
    listener.enter_write.assert_not_called()
    listener.exit_write.assert_not_called()
    listener.enter_broadcast.assert_not_called()
    listener.exit_broadcast.assert_not_called()
    listener.enter_rsp256_assignment.assert_not_called()
    listener.exit_rsp256_assignment.assert_not_called()
    listener.enter_rsp2k_assignment.assert_called_once_with(rsp2k_assignment)
    listener.exit_rsp2k_assignment.assert_called_once_with(rsp2k_assignment)
    listener.enter_rsp32k_assignment.assert_not_called()
    listener.exit_rsp32k_assignment.assert_not_called()

    visitor = stub_visitor()
    walker.walk(visitor, assignment)
    visitor.visit_assignment.assert_called_once_with(assignment)
    visitor.visit_read.assert_not_called()
    visitor.visit_write.assert_not_called()
    visitor.visit_broadcast.assert_not_called()
    visitor.visit_rsp256_assignment.assert_not_called()
    visitor.visit_rsp2k_assignment.assert_not_called()
    visitor.visit_rsp32k_assignment.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, assignment)
    transformer.transform_assignment.assert_called_once_with(assignment)
    transformer.transform_read.assert_not_called()
    transformer.transform_write.assert_not_called()
    transformer.transform_broadcast.assert_not_called()
    transformer.transform_rsp256_assignment.assert_not_called()
    transformer.transform_rsp2k_assignment.assert_called_once_with(rsp2k_assignment)
    transformer.transform_rsp32k_assignment.assert_not_called()

    assignment = ASSIGNMENT(
        operation=rsp32k_assignment,
    )

    listener = stub_listener()
    walker.walk(listener, assignment)
    listener.enter_assignment.assert_called_once_with(assignment)
    listener.exit_assignment.assert_called_once_with(assignment)
    listener.enter_read.assert_not_called()
    listener.exit_read.assert_not_called()
    listener.enter_write.assert_not_called()
    listener.exit_write.assert_not_called()
    listener.enter_broadcast.assert_not_called()
    listener.exit_broadcast.assert_not_called()
    listener.enter_rsp256_assignment.assert_not_called()
    listener.exit_rsp256_assignment.assert_not_called()
    listener.enter_rsp2k_assignment.assert_not_called()
    listener.exit_rsp2k_assignment.assert_not_called()
    listener.enter_rsp32k_assignment.assert_called_once_with(rsp32k_assignment)
    listener.exit_rsp32k_assignment.assert_called_once_with(rsp32k_assignment)

    visitor = stub_visitor()
    walker.walk(visitor, assignment)
    visitor.visit_assignment.assert_called_once_with(assignment)
    visitor.visit_read.assert_not_called()
    visitor.visit_write.assert_not_called()
    visitor.visit_broadcast.assert_not_called()
    visitor.visit_rsp256_assignment.assert_not_called()
    visitor.visit_rsp2k_assignment.assert_not_called()
    visitor.visit_rsp32k_assignment.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, assignment)
    transformer.transform_assignment.assert_called_once_with(assignment)
    transformer.transform_read.assert_not_called()
    transformer.transform_write.assert_not_called()
    transformer.transform_broadcast.assert_not_called()
    transformer.transform_rsp256_assignment.assert_not_called()
    transformer.transform_rsp2k_assignment.assert_not_called()
    transformer.transform_rsp32k_assignment.assert_called_once_with(rsp32k_assignment)

    shifted_sm_reg = SHIFTED_SM_REG(
        register=fs_rp,
        num_bits=4
    )

    listener = stub_listener()
    walker.walk(listener, shifted_sm_reg)
    listener.enter_shifted_sm_reg.assert_called_once_with(shifted_sm_reg)
    listener.exit_shifted_sm_reg.assert_called_once_with(shifted_sm_reg)
    listener.enter_sm_reg.assert_called_once_with(fs_rp)
    listener.exit_sm_reg.assert_called_once_with(fs_rp)

    visitor = stub_visitor()
    walker.walk(visitor, shifted_sm_reg)
    visitor.visit_shifted_sm_reg.assert_called_once_with(shifted_sm_reg)
    visitor.visit_sm_reg.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, shifted_sm_reg)
    transformer.transform_shifted_sm_reg.assert_called_once_with(shifted_sm_reg)
    transformer.transform_sm_reg.assert_called_once_with(fs_rp)

    mask = MASK(
        expression=fs_rp,
        operator=UNARY_OP.NEGATE,
    )

    listener = stub_listener()
    walker.walk(listener, mask)
    listener.enter_mask.assert_called_once_with(mask)
    listener.exit_mask.assert_called_once_with(mask)
    listener.enter_shifted_sm_reg.assert_not_called()
    listener.exit_shifted_sm_reg.assert_not_called()
    listener.enter_sm_reg.assert_called_once_with(fs_rp)
    listener.exit_sm_reg.assert_called_once_with(fs_rp)
    listener.enter_unary_op.assert_called_once_with(UNARY_OP.NEGATE)
    listener.exit_unary_op.assert_called_once_with(UNARY_OP.NEGATE)

    visitor = stub_visitor()
    walker.walk(visitor, mask)
    visitor.visit_mask.assert_called_once_with(mask)
    visitor.visit_shifted_sm_reg.assert_not_called()
    visitor.visit_sm_reg.assert_not_called()
    visitor.visit_unary_op.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, mask)
    transformer.transform_mask.assert_called_once_with(mask)
    transformer.transform_shifted_sm_reg.assert_not_called()
    transformer.transform_sm_reg.assert_called_once_with(fs_rp)
    transformer.transform_unary_op.assert_called_once_with(UNARY_OP.NEGATE)

    mask = MASK(
        expression=fs_rp,
        operator=None,
    )

    listener = stub_listener()
    walker.walk(listener, mask)
    listener.enter_mask.assert_called_once_with(mask)
    listener.exit_mask.assert_called_once_with(mask)
    listener.enter_shifted_sm_reg.assert_not_called()
    listener.exit_shifted_sm_reg.assert_not_called()
    listener.enter_sm_reg.assert_called_once_with(fs_rp)
    listener.exit_sm_reg.assert_called_once_with(fs_rp)

    visitor = stub_visitor()
    walker.walk(visitor, mask)
    visitor.visit_mask.assert_called_once_with(mask)
    visitor.visit_shifted_sm_reg.assert_not_called()
    visitor.visit_sm_reg.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, mask)
    transformer.transform_mask.assert_called_once_with(mask)
    transformer.transform_shifted_sm_reg.assert_not_called()
    transformer.transform_sm_reg.assert_called_once_with(fs_rp)

    mask = MASK(
        expression=shifted_sm_reg,
        operator=UNARY_OP.NEGATE,
    )

    listener = stub_listener()
    walker.walk(listener, mask)
    listener.enter_mask.assert_called_once_with(mask)
    listener.exit_mask.assert_called_once_with(mask)
    listener.enter_shifted_sm_reg.assert_called_once_with(shifted_sm_reg)
    listener.exit_shifted_sm_reg.assert_called_once_with(shifted_sm_reg)
    listener.enter_sm_reg.assert_called_once_with(fs_rp)
    listener.exit_sm_reg.assert_called_once_with(fs_rp)
    listener.enter_unary_op.assert_called_once_with(UNARY_OP.NEGATE)
    listener.exit_unary_op.assert_called_once_with(UNARY_OP.NEGATE)

    visitor = stub_visitor()
    walker.walk(visitor, mask)
    visitor.visit_mask.assert_called_once_with(mask)
    visitor.visit_shifted_sm_reg.assert_not_called()
    visitor.visit_sm_reg.assert_not_called()
    visitor.visit_unary_op.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, mask)
    transformer.transform_mask.assert_called_once_with(mask)
    transformer.transform_shifted_sm_reg.assert_called_once_with(shifted_sm_reg)
    transformer.transform_sm_reg.assert_called_once_with(fs_rp)
    transformer.transform_unary_op.assert_called_once_with(UNARY_OP.NEGATE)

    mask = MASK(
        expression=shifted_sm_reg,
        operator=None,
    )

    listener = stub_listener()
    walker.walk(listener, mask)
    listener.enter_mask.assert_called_once_with(mask)
    listener.exit_mask.assert_called_once_with(mask)
    listener.enter_shifted_sm_reg.assert_called_once_with(shifted_sm_reg)
    listener.exit_shifted_sm_reg.assert_called_once_with(shifted_sm_reg)
    listener.enter_sm_reg.assert_called_once_with(fs_rp)
    listener.exit_sm_reg.assert_called_once_with(fs_rp)

    visitor = stub_visitor()
    walker.walk(visitor, mask)
    visitor.visit_mask.assert_called_once_with(mask)
    visitor.visit_shifted_sm_reg.assert_not_called()
    visitor.visit_sm_reg.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, mask)
    transformer.transform_mask.assert_called_once_with(mask)
    transformer.transform_shifted_sm_reg.assert_called_once_with(shifted_sm_reg)
    transformer.transform_sm_reg.assert_called_once_with(fs_rp)

    masked = MASKED(
        mask=mask,
        assignment=assignment,
    )

    listener = stub_listener()
    walker.walk(listener, masked)
    listener.enter_masked.assert_called_once_with(masked)
    listener.exit_masked.assert_called_once_with(masked)
    listener.enter_mask.assert_called_once_with(mask)
    listener.exit_mask.assert_called_once_with(mask)
    listener.enter_assignment.assert_called_once_with(assignment)
    listener.exit_assignment.assert_called_once_with(assignment)

    visitor = stub_visitor()
    walker.walk(visitor, masked)
    visitor.visit_masked.assert_called_once_with(masked)
    visitor.visit_mask.assert_not_called()
    visitor.visit_assignment.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, masked)
    transformer.transform_masked.assert_called_once_with(masked)
    transformer.transform_mask.assert_called_once_with(mask)
    transformer.transform_assignment.assert_called_once_with(assignment)

    statement_1 = STATEMENT(
        operation=masked,
    )

    listener = stub_listener()
    walker.walk(listener, statement_1)
    listener.enter_statement.assert_called_once_with(statement_1)
    listener.exit_statement.assert_called_once_with(statement_1)
    listener.enter_masked.assert_called_once_with(masked)
    listener.exit_masked.assert_called_once_with(masked)
    listener.enter_special.assert_not_called()
    listener.exit_special.assert_not_called()

    visitor = stub_visitor()
    walker.walk(visitor, statement_1)
    visitor.visit_statement.assert_called_once_with(statement_1)
    visitor.visit_masked.assert_not_called()
    visitor.visit_special.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, statement_1)
    transformer.transform_statement.assert_called_once_with(statement_1)
    transformer.transform_masked.assert_called_once_with(masked)
    transformer.transform_special.assert_not_called()

    statement_2 = STATEMENT(
        operation=SPECIAL.RSP_END,
    )

    listener = stub_listener()
    walker.walk(listener, statement_2)
    listener.enter_statement.assert_called_once_with(statement_2)
    listener.exit_statement.assert_called_once_with(statement_2)
    listener.enter_masked.assert_not_called()
    listener.exit_masked.assert_not_called()
    listener.enter_special.assert_called_once_with(SPECIAL.RSP_END)
    listener.exit_special.assert_called_once_with(SPECIAL.RSP_END)

    visitor = stub_visitor()
    walker.walk(visitor, statement_2)
    visitor.visit_statement.assert_called_once_with(statement_2)
    visitor.visit_masked.assert_not_called()
    visitor.visit_special.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, statement_2)
    transformer.transform_statement.assert_called_once_with(statement_2)
    transformer.transform_masked.assert_not_called()
    transformer.transform_special.assert_called_once_with(SPECIAL.RSP_END)

    multi_statement = MultiStatement(
        statements=[statement_1]
    )

    listener = stub_listener()
    walker.walk(listener, multi_statement)
    listener.enter_multi_statement.assert_called_once_with(multi_statement)
    listener.exit_multi_statement.assert_called_once_with(multi_statement)
    assert listener.enter_statement.call_args_list == [
        call(statement_1),
    ]
    assert listener.exit_statement.call_args_list == [
        call(statement_1),
    ]

    visitor = stub_visitor()
    walker.walk(visitor, multi_statement)
    visitor.visit_multi_statement.assert_called_once_with(multi_statement)
    visitor.visit_statement.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, multi_statement)
    transformer.transform_multi_statement.assert_called_once_with(multi_statement)
    assert transformer.transform_statement.call_args_list == [
        call(statement_1),
    ]

    multi_statement = MultiStatement(
        statements=[statement_1, statement_2]
    )

    listener = stub_listener()
    walker.walk(listener, multi_statement)
    listener.enter_multi_statement.assert_called_once_with(multi_statement)
    listener.exit_multi_statement.assert_called_once_with(multi_statement)
    assert listener.enter_statement.call_args_list == [
        call(statement_1),
        call(statement_2),
    ]
    assert listener.exit_statement.call_args_list == [
        call(statement_1),
        call(statement_2),
    ]

    visitor = stub_visitor()
    walker.walk(visitor, multi_statement)
    visitor.visit_multi_statement.assert_called_once_with(multi_statement)
    visitor.visit_statement.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, multi_statement)
    transformer.transform_multi_statement.assert_called_once_with(multi_statement)
    assert transformer.transform_statement.call_args_list == [
        call(statement_1),
        call(statement_2),
    ]

    multi_statement = MultiStatement(
        statements=[statement_1, statement_2, statement_1]
    )

    listener = stub_listener()
    walker.walk(listener, multi_statement)
    listener.enter_multi_statement.assert_called_once_with(multi_statement)
    listener.exit_multi_statement.assert_called_once_with(multi_statement)
    assert listener.enter_statement.call_args_list == [
        call(statement_1),
        call(statement_2),
        call(statement_1),
    ]
    assert listener.exit_statement.call_args_list == [
        call(statement_1),
        call(statement_2),
        call(statement_1),
    ]

    visitor = stub_visitor()
    walker.walk(visitor, multi_statement)
    visitor.visit_multi_statement.assert_called_once_with(multi_statement)
    visitor.visit_statement.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, multi_statement)
    transformer.transform_multi_statement.assert_called_once_with(multi_statement)
    assert transformer.transform_statement.call_args_list == [
        call(statement_1),
        call(statement_2),
        call(statement_1),
    ]

    multi_statement = MultiStatement(
        statements=[statement_1, statement_2, statement_1, statement_2]
    )

    listener = stub_listener()
    walker.walk(listener, multi_statement)
    listener.enter_multi_statement.assert_called_once_with(multi_statement)
    listener.exit_multi_statement.assert_called_once_with(multi_statement)
    assert listener.enter_statement.call_args_list == [
        call(statement_1),
        call(statement_2),
        call(statement_1),
        call(statement_2),
    ]
    assert listener.exit_statement.call_args_list == [
        call(statement_1),
        call(statement_2),
        call(statement_1),
        call(statement_2),
    ]

    visitor = stub_visitor()
    walker.walk(visitor, multi_statement)
    visitor.visit_multi_statement.assert_called_once_with(multi_statement)
    visitor.visit_statement.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, multi_statement)
    transformer.transform_multi_statement.assert_called_once_with(multi_statement)
    assert transformer.transform_statement.call_args_list == [
        call(statement_1),
        call(statement_2),
        call(statement_1),
        call(statement_2),
    ]

    operation = statement_1

    listener = stub_listener()
    walker.walk(listener, operation)
    listener.enter_multi_statement.assert_not_called()
    listener.exit_multi_statement.assert_not_called()
    assert listener.enter_statement.call_args_list == [
        call(statement_1),
    ]
    assert listener.exit_statement.call_args_list == [
        call(statement_1),
    ]

    visitor = stub_visitor()
    walker.walk(visitor, operation)
    visitor.visit_multi_statement.assert_not_called()
    assert visitor.visit_statement.call_args_list == [
        call(statement_1),
    ]

    transformer = stub_transformer()
    walker.walk(transformer, operation)
    transformer.transform_multi_statement.assert_not_called()
    assert transformer.transform_statement.call_args_list == [
        call(statement_1),
    ]

    operation = multi_statement

    listener = stub_listener()
    walker.walk(listener, operation)
    listener.enter_multi_statement.assert_called_once_with(multi_statement)
    listener.exit_multi_statement.assert_called_once_with(multi_statement)
    assert listener.enter_statement.call_args_list == [
        call(statement_1),
        call(statement_2),
        call(statement_1),
        call(statement_2),
    ]
    assert listener.exit_statement.call_args_list == [
        call(statement_1),
        call(statement_2),
        call(statement_1),
        call(statement_2),
    ]

    visitor = stub_visitor()
    walker.walk(visitor, operation)
    visitor.visit_multi_statement.assert_called_once_with(multi_statement)
    visitor.visit_statement.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, operation)
    transformer.transform_multi_statement.assert_called_once_with(multi_statement)
    assert transformer.transform_statement.call_args_list == [
        call(statement_1),
        call(statement_2),
        call(statement_1),
        call(statement_2),
    ]

    formal_parameter = lvr_rp

    listener = stub_listener()
    walker.walk(listener, formal_parameter)
    listener.enter_rn_reg.assert_called_once_with(lvr_rp)
    listener.exit_rn_reg.assert_called_once_with(lvr_rp)
    listener.enter_sm_reg.assert_not_called()
    listener.exit_sm_reg.assert_not_called()

    visitor = stub_visitor()
    walker.walk(visitor, formal_parameter)
    visitor.visit_rn_reg.assert_called_once_with(lvr_rp)
    visitor.visit_sm_reg.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, formal_parameter)
    transformer.transform_rn_reg.assert_called_once_with(lvr_rp)
    transformer.transform_sm_reg.assert_not_called()

    formal_parameter = fs_rp

    listener = stub_listener()
    walker.walk(listener, formal_parameter)
    listener.enter_rn_reg.assert_not_called()
    listener.exit_rn_reg.assert_not_called()
    listener.enter_sm_reg.assert_called_once_with(fs_rp)
    listener.exit_sm_reg.assert_called_once_with(fs_rp)

    visitor = stub_visitor()
    walker.walk(visitor, formal_parameter)
    visitor.visit_rn_reg.assert_not_called()
    visitor.visit_sm_reg.assert_called_once_with(fs_rp)

    transformer = stub_transformer()
    walker.walk(transformer, formal_parameter)
    transformer.transform_rn_reg.assert_not_called()
    transformer.transform_sm_reg.assert_called_once_with(fs_rp)

    actual_parameter = 0xF00D

    listener = stub_listener()
    walker.walk(listener, actual_parameter)
    listener.enter_parameter.assert_not_called()
    listener.exit_parameter.assert_not_called()

    visitor = stub_visitor()
    walker.walk(visitor, actual_parameter)
    visitor.visit_parameter.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, actual_parameter)
    transformer.transform_parameter.assert_not_called()

    fragment_1 = Fragment(
        identifier="foo",
        parameters=[lvr_rp, fs_rp],
        operations=[statement_1, statement_2],
    )

    listener = stub_listener()
    walker.walk(listener, fragment_1)
    listener.enter_fragment.assert_called_once_with(fragment_1)
    listener.exit_fragment.assert_called_once_with(fragment_1)

    visitor = stub_visitor()
    walker.walk(visitor, fragment_1)
    visitor.visit_fragment.assert_called_once_with(fragment_1)

    transformer = stub_transformer()
    walker.walk(transformer, fragment_1)
    transformer.transform_fragment.assert_called_once_with(fragment_1)

    fragment_2 = Fragment(
        identifier="bar",
        parameters=[lvr_rp, rvr_rp, fs_rp],
        operations=[multi_statement],
    )

    listener = stub_listener()
    walker.walk(listener, fragment_2)
    listener.enter_fragment.assert_called_once_with(fragment_2)
    listener.exit_fragment.assert_called_once_with(fragment_2)

    visitor = stub_visitor()
    walker.walk(visitor, fragment_2)
    visitor.visit_fragment.assert_called_once_with(fragment_2)

    transformer = stub_transformer()
    walker.walk(transformer, fragment_2)
    transformer.transform_fragment.assert_called_once_with(fragment_2)

    fragment_1_caller = FragmentCaller(
        fragment=fragment_1,
    )

    listener = stub_listener()
    walker.walk(listener, fragment_1_caller)
    listener.enter_fragment_caller.assert_called_once_with(fragment_1_caller)
    listener.exit_fragment_caller.assert_called_once_with(fragment_1_caller)
    listener.enter_fragment.assert_called_once_with(fragment_1)
    listener.exit_fragment.assert_called_once_with(fragment_1)

    visitor = stub_visitor()
    walker.walk(visitor, fragment_1_caller)
    visitor.visit_fragment_caller.assert_called_once_with(fragment_1_caller)
    visitor.visit_fragment.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, fragment_1_caller)
    transformer.transform_fragment_caller.assert_called_once_with(fragment_1_caller)
    transformer.transform_fragment.assert_called_once_with(fragment_1)

    fragment_1_caller_call = FragmentCallerCall(
        caller=fragment_1_caller,
        parameters=[0xFFFF],
    )

    listener = stub_listener()
    walker.walk(listener, fragment_1_caller_call)
    listener.enter_fragment_caller_call.assert_called_once_with(fragment_1_caller_call)
    listener.exit_fragment_caller_call.assert_called_once_with(fragment_1_caller_call)
    listener.enter_fragment_caller.assert_called_once_with(fragment_1_caller)
    listener.exit_fragment_caller.assert_called_once_with(fragment_1_caller)

    visitor = stub_visitor()
    walker.walk(visitor, fragment_1_caller_call)
    visitor.visit_fragment_caller_call.assert_called_once_with(fragment_1_caller_call)
    visitor.visit_fragment_caller.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, fragment_1_caller_call)
    transformer.transform_fragment_caller_call.assert_called_once_with(fragment_1_caller_call)
    transformer.transform_fragment_caller.assert_called_once_with(fragment_1_caller)

    fragment_2_caller = FragmentCaller(
        fragment=fragment_2,
    )

    fragment_2_caller_call = FragmentCallerCall(
        caller=fragment_2_caller,
        parameters=[0xF00D, 0x1234, 0xFFFF],
    )

    value_parameter = ValueParameter(
        identifier="out",
        row_number=0,
        value=np.repeat(0x53AD, NUM_PLATS_PER_APUC).astype(np.uint16))

    example = Example(expected_value=value_parameter)

    snippet = Snippet(
        name="test_foo_bar",
        examples=[example],
        calls=[fragment_1_caller_call, fragment_2_caller_call],
    )

    listener = stub_listener()
    walker.walk(listener, snippet)
    listener.enter_snippet.assert_called_once_with(snippet)
    listener.exit_snippet.assert_called_once_with(snippet)
    listener.enter_example.assert_called_once_with(example)
    listener.exit_example.assert_called_once_with(example)
    listener.enter_value_parameter.assert_called_once_with(value_parameter)
    listener.exit_value_parameter.assert_called_once_with(value_parameter)
    assert listener.enter_fragment_caller_call.call_args_list == [
        call(fragment_1_caller_call),
        call(fragment_2_caller_call),
    ]
    assert listener.exit_fragment_caller_call.call_args_list == [
        call(fragment_1_caller_call),
        call(fragment_2_caller_call),
    ]

    visitor = stub_visitor()
    walker.walk(visitor, snippet)
    visitor.visit_snippet.assert_called_once_with(snippet)
    visitor.visit_example.assert_not_called()
    visitor.visit_value_parameter.assert_not_called()
    visitor.visit_fragment_caller_call.assert_not_called()

    transformer = stub_transformer()
    walker.walk(transformer, snippet)
    transformer.transform_snippet.assert_called_once_with(snippet)
    transformer.transform_example.assert_called_once_with(example)
    transformer.transform_value_parameter.assert_called_once_with(value_parameter)
    assert transformer.transform_fragment_caller_call.call_args_list == [
        call(fragment_1_caller_call),
        call(fragment_2_caller_call),
    ]
