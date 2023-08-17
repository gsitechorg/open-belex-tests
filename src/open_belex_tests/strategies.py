r"""
By Brian Beckman and Dylon Edwards.
"""

import pickle
from collections import deque
from dataclasses import dataclass, field
from hashlib import md5
from typing import (Any, Callable, Dict, List, Optional, Sequence, Set, Tuple,
                    Union)

import numpy

import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

import open_belex.bleir.boilerplate as boilerplate
import open_belex.bleir.types as BLEIR
from open_belex.bleir.analyzers import (MAX_FRAGMENT_INSTRUCTIONS,
                                        SBGroupAnalyzer)
from open_belex.bleir.syntactic_validators import (FRAGMENT_ID_PATTERN,
                                                   REGISTER_PARAM_ID_PATTERN)
from open_belex.bleir.walkables import BLEIRWalker
from open_belex.common.constants import *
from open_belex.common.constants import NPLATS
from open_belex.common.register_arenas import (NUM_RN_REGS, RnRegArena,
                                               SmRegArena)
from open_belex.common.subset import *

from open_belex_tests.alias_method import AliasMethod

#   _            _ _
#  | |_  _____ _(_) |_
#  | ' \/ -_) \ / |  _|
#  |_||_\___/_\_\_|\__|


def power_of_2_strategy_maker(exponent):
    return (
        st.integers(
            min_value=0,
            max_value=2 ** exponent - 1),
        st.integers(
            min_value=-2 ** (exponent - 1),
            max_value=2 ** (exponent - 1) - 1),
    )


u4_strategy, s4_strategy = power_of_2_strategy_maker(4)
u8_strategy, s8_strategy = power_of_2_strategy_maker(8)
u16_strategy, s16_strategy = power_of_2_strategy_maker(16)
u32_strategy, s32_strategy = power_of_2_strategy_maker(32)
u64_strategy, s64_strategy = power_of_2_strategy_maker(64)


def exact_power_of_2_lt_2_to_the_n_maker(n):
    result = st.builds(
        lambda n: 2**n,
        st.integers(
            min_value=0,
            max_value=n-1))
    return result


p2_lt_2_4_strategy = exact_power_of_2_lt_2_to_the_n_maker(4)
p2_lt_2_8_strategy = exact_power_of_2_lt_2_to_the_n_maker(8)
p2_lt_2_16_strategy = exact_power_of_2_lt_2_to_the_n_maker(16)
p2_lt_2_32_strategy = exact_power_of_2_lt_2_to_the_n_maker(32)
p2_lt_2_64_strategy = exact_power_of_2_lt_2_to_the_n_maker(64)


#                 _
#   _ __  __ _ __| |__ ___
#  | '  \/ _` (_-< / /(_-<
#  |_|_|_\__,_/__/_\_\/__/

# NOT guaranteed unique (not sets)

sixteen_strategy = st.integers(
    min_value=0,
    max_value=NSECTIONS - 1)

sixteen_bit_strategy = st.integers(
    min_value=0,
    max_value=((2 ** NSECTIONS) - 1))

mask_hex_strategy = st.builds(
    lambda k: f'{k:04X}',
    sixteen_bit_strategy)

mask_bitstring_strategy = st.builds(
    lambda k: bin(k)[2:].zfill(NSECTIONS),
    sixteen_bit_strategy)

sections_smallish_array_strategy = arrays(
    dtype=numpy.int16,
    elements=st.integers(min_value=0, max_value=NSECTIONS - 1),
    unique=False,  # NOTA BENE: Defaults to False; here we emphasize.
    shape=st.builds(lambda x: (x,),
                    st.integers(min_value=0, max_value=NSECTIONS)))

section_mask_strategy = st.builds(
    lambda x: Mask(user_input=x),
    sections_smallish_array_strategy)


#                  _
#   _ __  __ _ _ _| |_____ _ _ ___
#  | '  \/ _` | '_| / / -_) '_(_-<
#  |_|_|_\__,_|_| |_\_\___|_| /__/

# NOT guaranteed unique (not sets)

marker_single_index_strategy = st.integers(
    min_value=0,
    max_value=NPLATS - 1)

marker_bits_strategy = st.integers(
    min_value=0,
    max_value=((2 ** NPLATS) - 1))

marker_hex_strategy = st.builds(
    lambda k: f'{k:X}',
    marker_bits_strategy)

marker_bitstring_strategy = st.builds(
    lambda k: bin(k)[2:].zfill(NPLATS),
    marker_bits_strategy)

marker_biggish_array_strategy = arrays(
    dtype=numpy.int16,
    elements=st.integers(min_value=0, max_value=NPLATS - 1),
    unique=False,
    shape=st.builds(lambda x: (x,),
                    st.integers(min_value=0, max_value=NPLATS)))

#              _   _
#   ___ ___ __| |_(_)___ _ _  ___
#  (_-</ -_) _|  _| / _ \ ' \(_-<
#  /__/\___\__|\__|_\___/_||_/__/

# NOT guaranteed unique (not sets)


section_strategy = st.integers(
    min_value=0,
    max_value=NSECTIONS - 1)


#                     _ _ _
#  __ __ _____ _ _ __| | (_)_ _  ___ ___
#  \ V  V / _ \ '_/ _` | | | ' \/ -_|_-<
#   \_/\_/\___/_| \__,_|_|_|_||_\___/__/

# NOT guaranteed unique (not sets)

wordline_strategy = st.builds(
    lambda x: Subset(max=NPLATS - 1, user_input=x),
    marker_biggish_array_strategy)

wordline_booleans_strategy = st.lists(
    st.booleans(), max_size = NPLATS - 1)


#  __   _____
#  \ \ / / _ \ ___
#   \ V /|   /(_-<
#    \_/ |_|_\/__/

# NOT guaranteed unique (not sets)


sb_strategy = st.integers(
    min_value=0,
    max_value=NSB - 1)

vr_strategy = st.integers(
    min_value=0,
    max_value=NVR - 2)  # Exclude RL! (19 Nov 2020)

vrs_strategy = arrays(
    dtype=numpy.int16,
    elements=st.integers(min_value=0, max_value=NSB-1),
    shape=st.builds(lambda x: (x,),
                    st.integers(min_value=0, max_value=NSB)),
    unique=False)

#  _      ____
# | | /| / / /  ___
# | |/ |/ / /__(_-<
# |__/|__/____/___/


VRs_strategy = st.lists(vr_strategy,
                        min_size=1,
                        max_size=NVR-1,
                        unique=True)


sections_strategy = st.lists(section_strategy,
                             min_size=1,
                             max_size=NSECTIONS,
                             unique=True)


def WLs_strategy(nullable=True):
    if nullable:
        return st.one_of(st.none(), WLs_strategy(False))
    return st.builds(WLs, VRs_strategy, sections_strategy)


#   ___ _    ___ ___ ___
#  | _ ) |  | __|_ _| _ \  __ _ _ _ __ _ _ __  _ __  __ _ _ _
#  | _ \ |__| _| | ||   / / _` | '_/ _` | '  \| '  \/ _` | '_|
#  |___/____|___|___|_|_\ \__, |_| \__,_|_|_|_|_|_|_\__,_|_|
#                         |___/

def Optional_strategy(strategy):
    return st.one_of(strategy, st.none())

# Symbols
RL_EXPR_strategy = st.sampled_from(BLEIR.RL_EXPR)
RSP16_EXPR_strategy = st.sampled_from(BLEIR.RSP16_EXPR)
RSP256_EXPR_strategy = st.sampled_from(BLEIR.RSP256_EXPR)
RSP2K_EXPR_strategy = st.sampled_from(BLEIR.RSP2K_EXPR)
RSP32K_EXPR_strategy = st.sampled_from(BLEIR.RSP32K_EXPR)
BIT_EXPR_strategy = st.sampled_from(BLEIR.BIT_EXPR)

SRC_EXPR_strategy = st.sampled_from([
    BLEIR.SRC_EXPR.RL,
    BLEIR.SRC_EXPR.NRL,
    BLEIR.SRC_EXPR.ERL,
    BLEIR.SRC_EXPR.WRL,
    BLEIR.SRC_EXPR.SRL,
    BLEIR.SRC_EXPR.GL,
    BLEIR.SRC_EXPR.GGL,

    # TODO: Determine the proper way to use RSP16 and then add it back to the pool
    # BLEIR.SRC_EXPR.RSP16,

    # Let the optimizer handle the INV_ cases (rewritten from ~SRC)
    # BLEIR.SRC_EXPR.INV_RL,
    # BLEIR.SRC_EXPR.INV_NRL,
    # BLEIR.SRC_EXPR.INV_ERL,
    # BLEIR.SRC_EXPR.INV_WRL,
    # BLEIR.SRC_EXPR.INV_SRL,
    # BLEIR.SRC_EXPR.INV_GL,
    # BLEIR.SRC_EXPR.INV_GGL,
    # BLEIR.SRC_EXPR.INV_RSP16,
])

# BROADCAST_EXPR_strategy = st.sampled_from(BLEIR.BROADCAST_EXPR)
BROADCAST_EXPR_strategy = st.sampled_from([
    BLEIR.BROADCAST_EXPR.GL,
    BLEIR.BROADCAST_EXPR.GGL,

    # TODO: Determine the proper way to use RSP16 and then add it back to the pool
    # BLEIR.BROADCAST_EXPR.RSP16,
])

# TODO: Determine how to use RSP_END correctly
# SPECIAL_strategy = st.sampled_from(BLEIR.SPECIAL)
SPECIAL_strategy = st.sampled_from([
    BLEIR.SPECIAL.NOOP,
    # BLEIR.SPECIAL.RSP_END,
])

# Operators
ASSIGN_OP_strategy = st.sampled_from([
    BLEIR.ASSIGN_OP.EQ,
    BLEIR.ASSIGN_OP.AND_EQ,
    BLEIR.ASSIGN_OP.OR_EQ,
    BLEIR.ASSIGN_OP.XOR_EQ,

    # COND_EQ (?=) only applies to WRITEs
    # BLEIR.ASSIGN_OP.COND_EQ,
])
UNARY_OP_strategy = st.sampled_from(BLEIR.UNARY_OP)


# Reserve RN_REG_0 for full-VR I/O
rn_reg_id_strategy = st.integers(min_value=1, max_value=15)
sm_reg_id_strategy = st.integers(min_value=0, max_value=15)


@st.composite
def Register_strategy(draw: Callable,
                      formal_parameter: BLEIR.FormalParameter,
                      sm_reg_arena: SmRegArena,
                      rn_reg_arena: RnRegArena) -> str:

    if isinstance(formal_parameter, BLEIR.RN_REG):
        register = rn_reg_arena.allocate()
        assert register is not None
        return register

    if isinstance(formal_parameter, BLEIR.SM_REG):
        register = sm_reg_arena.allocate()
        assert register is not None
        return register

    raise ValueError(f"Unsupported formal_paramter type ({type(formal_parameter).__name__}): {formal_parameter}")


@st.composite
def AllocatedRegister_strategy(draw: Callable,
                               formal_parameter: BLEIR.FormalParameter,
                               sm_reg_arena: SmRegArena,
                               rn_reg_arena: RnRegArena) -> BLEIR.AllocatedRegister:
    return BLEIR.AllocatedRegister(
        parameter=formal_parameter,
        register=draw(Register_strategy(formal_parameter,
                                        sm_reg_arena,
                                        rn_reg_arena)))


@st.composite
def UNARY_SRC_strategy(draw: Callable,
                       assign_op: Optional[BLEIR.ASSIGN_OP] = None,
                       binary_op: Optional[BLEIR.BINOP] = None,
                       lvalue: Optional[BLEIR.UNARY_SB] = None) -> BLEIR.UNARY_SRC:

    operator = None

    if assign_op is None or \
       assign_op is BLEIR.ASSIGN_OP.AND_EQ and binary_op is None or \
       assign_op is BLEIR.ASSIGN_OP.EQ and binary_op is BLEIR.BINOP.AND and lvalue.operator is None:
        operator = draw(Optional_strategy(UNARY_OP_strategy))

    return BLEIR.UNARY_SRC(
        expression=draw(SRC_EXPR_strategy),
        operator=operator)


@st.composite
def SB_EXPR_strategy(draw: Callable,
                     rn_params: Sequence[BLEIR.RN_REG]) -> BLEIR.SB_EXPR:

    available_rn_params = list(rn_params) # copy for manipulation
    rn_regs = []

    x = draw(st.sampled_from(available_rn_params))
    available_rn_params.remove(x)
    rn_regs.append(x)

    if draw(st.booleans()) and len(available_rn_params) > 0:  # draw y
        y = draw(st.sampled_from(available_rn_params))
        available_rn_params.remove(y)
        rn_regs.append(y)

        if draw(st.booleans()) and len(available_rn_params) > 0:  # draw z
            z = draw(st.sampled_from(available_rn_params))
            rn_regs.append(z)

    return BLEIR.SB_EXPR(rn_regs=rn_regs)


@st.composite
def UNARY_SB_strategy(draw: Callable,
                      rn_params: Sequence[BLEIR.RN_REG],
                      assign_op: Optional[BLEIR.ASSIGN_OP] = None,
                      binary_op: Optional[BLEIR.BINOP] = None) -> BLEIR.UNARY_SB:

    operator = None

    if assign_op is None or \
       assign_op is BLEIR.ASSIGN_OP.AND_EQ and binary_op is None or \
       assign_op is BLEIR.ASSIGN_OP.EQ and binary_op is BLEIR.BINOP.AND:
        operator = draw(Optional_strategy(UNARY_OP_strategy))

    return BLEIR.UNARY_SB(
        expression=draw(SB_EXPR_strategy(rn_params)),
        operator=operator)


@st.composite
def UNARY_EXPR_strategy(draw: Callable,
                        rn_params: Sequence[BLEIR.RN_REG],
                        assign_op: BLEIR.ASSIGN_OP) -> BLEIR.UNARY_EXPR:

    candidates = [UNARY_SB_strategy(rn_params, assign_op),
                  UNARY_SRC_strategy(assign_op)]

    if assign_op is BLEIR.ASSIGN_OP.EQ:
        candidates.append(BIT_EXPR_strategy)

    return BLEIR.UNARY_EXPR(
        expression=draw(st.one_of(*candidates)))


@st.composite
def BINARY_EXPR_strategy(draw: Callable,
                         rn_params: Sequence[BLEIR.RN_REG],
                         assign_op: BLEIR.ASSIGN_OP) -> BLEIR.BINARY_EXPR:

    if assign_op is BLEIR.ASSIGN_OP.EQ:
        operator = draw(st.sampled_from([
            BLEIR.BINOP.AND,
            BLEIR.BINOP.OR,
            BLEIR.BINOP.XOR,
        ]))
    else:
        operator = BLEIR.BINOP.AND  # no choice

    left_operand = draw(UNARY_SB_strategy(rn_params, assign_op, operator))
    right_operand = draw(UNARY_SRC_strategy(assign_op, operator, left_operand))
    return BLEIR.BINARY_EXPR(
        operator=operator,
        left_operand=left_operand,
        right_operand=right_operand)


@st.composite
def READ_strategy(draw: Callable,
                  rn_params: Sequence[BLEIR.RN_REG]) -> BLEIR.READ:
    operator = draw(ASSIGN_OP_strategy)
    return BLEIR.READ(
        operator=operator,
        rvalue=draw(st.one_of(UNARY_EXPR_strategy(rn_params, operator),
                              BINARY_EXPR_strategy(rn_params, operator))))


WRITE_operator_strategy = st.sampled_from([
    BLEIR.ASSIGN_OP.EQ,
    BLEIR.ASSIGN_OP.COND_EQ,
])


@st.composite
def WRITE_strategy(draw: Callable,
                   rn_params: Sequence[BLEIR.RN_REG]) -> BLEIR.WRITE:
    return BLEIR.WRITE(
        operator=draw(WRITE_operator_strategy),
        lvalue=draw(SB_EXPR_strategy(rn_params)),
        rvalue=draw(UNARY_SRC_strategy()))


BROADCAST_strategy = st.builds(BLEIR.BROADCAST,
                               lvalue=BROADCAST_EXPR_strategy)


# TODO: Consider adding these back into the assignment pool later
# RSP256_ASSIGNMENT_strategy = st.builds(BLEIR.RSP256_ASSIGNMENT)
# RSP2K_ASSIGNMENT_strategy = st.builds(BLEIR.RSP2K_ASSIGNMENT)
# RSP32K_ASSIGNMENT_strategy = st.builds(BLEIR.RSP32K_ASSIGNMENT)


@st.composite
def AssignmentType_strategy(draw: Callable,
                            rn_params: Sequence[BLEIR.RN_REG],
                            mask: BLEIR.MASK,
                            kind: Optional[Any] = None) -> Optional[BLEIR.AssignmentType]:

    if kind is BLEIR.READ:
        return draw(READ_strategy(rn_params))

    if kind is BLEIR.WRITE:
        return draw(WRITE_strategy(rn_params))

    if kind is BLEIR.BROADCAST:
        return draw(BROADCAST_strategy)

    assignment_strategies = [
        READ_strategy(rn_params),
        WRITE_strategy(rn_params),
        BROADCAST_strategy,
    ]

    return draw(st.one_of(assignment_strategies))


@st.composite
def ASSIGNMENT_strategy(draw: Callable,
                        rn_params: Sequence[BLEIR.RN_REG],
                        mask: BLEIR.MASK,
                        kind: Optional[Any] = None) -> Optional[BLEIR.ASSIGNMENT]:
    operation = draw(AssignmentType_strategy(rn_params, mask,
                                             kind=kind))
    if operation is None:
        return None
    return BLEIR.ASSIGNMENT(operation=operation)


@st.composite
def SHIFTED_SM_REG_strategy(draw: Callable,
                            sm_param: BLEIR.SM_REG) -> BLEIR.SHIFTED_SM_REG:
    return BLEIR.SHIFTED_SM_REG(
        register=sm_param,
        num_bits=draw(st.integers(min_value=0, max_value=15)))


@st.composite
def MASK_strategy(draw: Callable,
                  sm_params: Sequence[BLEIR.SM_REG]) -> BLEIR.MASK:
    sm_param = draw(st.sampled_from(sm_params))
    expression = sm_param
    if draw(st.booleans()):  # shift the sm_param
        expression = draw(SHIFTED_SM_REG_strategy(sm_param))
    return BLEIR.MASK(
        expression=expression,
        operator=draw(Optional_strategy(UNARY_OP_strategy)))


@st.composite
def MASKED_strategy(draw: Callable,
                    rn_params: Sequence[BLEIR.RN_REG],
                    sm_params: Sequence[BLEIR.SM_REG],
                    kind: Optional[Any] = None) -> Optional[BLEIR.MASKED]:
    mask = draw(MASK_strategy(sm_params))
    assignment = draw(ASSIGNMENT_strategy(rn_params, mask,
                                          kind=kind))
    if assignment is None:
        return None
    return BLEIR.MASKED(mask=mask, assignment=assignment)


def read_statement_factory(draw: Callable,
                           rn_params: Sequence[BLEIR.RN_REG],
                           sm_params: Sequence[BLEIR.SM_REG]) -> Optional[BLEIR.MASKED]:
    return draw(MASKED_strategy(rn_params, sm_params,
                                kind=BLEIR.READ))


def write_statement_factory(draw: Callable,
                            rn_params: Sequence[BLEIR.RN_REG],
                            sm_params: Sequence[BLEIR.SM_REG]) -> Optional[BLEIR.MASKED]:
    return draw(MASKED_strategy(rn_params, sm_params,
                                kind=BLEIR.WRITE))


def broadcast_statement_factory(draw: Callable,
                                rn_params: Sequence[BLEIR.RN_REG],
                                sm_params: Sequence[BLEIR.SM_REG]) -> Optional[BLEIR.MASKED]:
    return draw(MASKED_strategy(rn_params, sm_params,
                                kind=BLEIR.BROADCAST))


def noop_statement_factory(draw: Callable,
                           rn_params: Sequence[BLEIR.RN_REG],
                           sm_params: Sequence[BLEIR.SM_REG]) -> Optional[BLEIR.SPECIAL]:
    return BLEIR.SPECIAL.NOOP


def build_statement_distribution() -> Sequence[Tuple[Callable, float]]:
    return [
        (read_statement_factory, 0.34),
        (write_statement_factory, 0.45),
        (broadcast_statement_factory, 0.20),
        (noop_statement_factory, 0.01),
    ]


@dataclass
class StatementDistribution:
    distribution: Sequence[Tuple[Callable, float]] = field(default_factory=build_statement_distribution)
    random: Optional[numpy.random.RandomState] = None

    _alias_method: Optional[AliasMethod] = None

    @property
    def probabilities(self: "StatementDistribution") -> numpy.ndarray:
        probabilities = [pair[1] for pair in self.distribution]
        probabilities = numpy.array(probabilities, dtype=numpy.float64)
        probabilities = probabilities / probabilities.sum()
        return probabilities

    @property
    def factories(self: "StatementDistribution") -> Sequence[Callable]:
        return [pair[0] for pair in self.distribution]

    @property
    def alias_method(self: "StatementDistribution") -> AliasMethod:
        if self._alias_method is None:
            self._alias_method = AliasMethod(self.probabilities, random=self.random)
        return self._alias_method

    def sample(self: "StatementDistribution") -> BLEIR.STATEMENT:
        index = self.alias_method.sample()
        factory_fn = self.factories[index]
        return factory_fn


@st.composite
def STATEMENT_strategy(draw: Callable,
                       rn_params: Sequence[BLEIR.RN_REG],
                       sm_params: Sequence[BLEIR.SM_REG],
                       statement_distribution: Optional[StatementDistribution] = None) -> BLEIR.STATEMENT:

    # NOTE: We aren't going to generate RSP*_ASSIGNMENTs for now
    if len(rn_params) > 0 and len(sm_params) > 0:
        if statement_distribution is None:
            statement_seed = draw(st.integers(min_value=0, max_value=2**32-1))
            statement_random = numpy.random.RandomState(seed=statement_seed)
            statement_distribution = StatementDistribution(random=statement_random)
        factory_fn = statement_distribution.sample()
        operation = factory_fn(draw, rn_params, sm_params)

        # If no valid operation could be generated, default to a NOOP
        if operation is None:
            operation = BLEIR.SPECIAL.NOOP
    else:
        operation = draw(SPECIAL_strategy)

    return BLEIR.STATEMENT(operation=operation)


@st.composite
def MultiStatement_strategy(draw: Callable,
                            rn_params: Sequence[BLEIR.RN_REG],
                            sm_params: Sequence[BLEIR.SM_REG],
                            statement_distribution: Optional[StatementDistribution] = None) -> BLEIR.MultiStatement:
    statements = []

    num_statements = draw(st.integers(min_value=1, max_value=4))
    for _ in range(num_statements):
        statement = draw(STATEMENT_strategy(rn_params, sm_params,
                                            statement_distribution=statement_distribution))
        statements.append(statement)

    return BLEIR.MultiStatement(statements=statements)


@st.composite
def Operation_strategy(draw: Callable,
                       rn_params: Sequence[BLEIR.RN_REG],
                       sm_params: Sequence[BLEIR.SM_REG],
                       multi_statements_only: bool = False,
                       statement_distribution: Optional[StatementDistribution] = None) -> BLEIR.Operation:
    if multi_statements_only:
        return draw(MultiStatement_strategy(rn_params, sm_params))
    return draw(st.one_of(MultiStatement_strategy(rn_params, sm_params,
                                                  statement_distribution=statement_distribution),
                          STATEMENT_strategy(rn_params, sm_params,
                                             statement_distribution=statement_distribution)))


@st.composite
def ActualParameter_RN_REG_strategy(draw: Callable,
                                    formal_parameter: BLEIR.FormalParameter,
                                    colorings: Dict[BLEIR.RN_REG, Set[BLEIR.RN_REG]],
                                    unavailable_sbs: Set[int]) -> Optional[BLEIR.ActualParameter]:
    # Determine the available SB range for the given color (write-compatibility)
    color = colorings[formal_parameter]
    min_value = 8 * color
    max_value = min_value + 7

    # Clamp the min and max values to the available range, 1 -- 16
    if min_value == 0:
        min_value = 1
    if max_value > 16:
        max_value = 16

    # Do not select an SB that has already been selected
    neighborhood_sbs = set(range(min_value, max_value + 1))
    available_sbs = list(neighborhood_sbs - unavailable_sbs)
    if len(available_sbs) == 0:
        return None

    # Randomly draw from the available SBs for the given color
    return draw(st.sampled_from(available_sbs))


@st.composite
def ActualParameter_SM_REG_strategy(draw: Callable,
                                    formal_parameter: BLEIR.SM_REG,
                                    sm_reg_map: Dict[BLEIR.SM_REG, int]) -> Optional[BLEIR.ActualParameter]:
    r"""
    Generates a mask that does not overlap previously drawn masks.

    If we have two SM_REGs:
        1. a_rp = 0x0000
        2. b_rp = 0x0001

    ... then the following multi-statement will consist of overlapping writes,
    which is the kind of scenario we want to avoid:
    {
        ~(a_rp<<0): SB[s1_rp] = RL;
        b_rp: SB[s1_rp] = RSP16;
    }
    """
    return sm_reg_map[formal_parameter]


@st.composite
def ActualParameter_strategy(draw: Callable,
                             formal_parameter: BLEIR.FormalParameter,
                             colorings: Dict[BLEIR.RN_REG, Set[BLEIR.RN_REG]],
                             unavailable_sbs: Set[int],
                             sm_reg_map: Dict[BLEIR.SM_REG, int]) -> Optional[BLEIR.ActualParameter]:

    if isinstance(formal_parameter, BLEIR.RN_REG):
        actual_parameter = draw(ActualParameter_RN_REG_strategy(formal_parameter, colorings, unavailable_sbs))
    elif isinstance(formal_parameter, BLEIR.SM_REG):
        actual_parameter = draw(ActualParameter_SM_REG_strategy(formal_parameter, sm_reg_map))
    else:
        raise ValueError(f"Unsupported formal_parameter type ({type(formal_parameter).__name__}): {formal_parameter}")

    if actual_parameter is None:
        # The actual_parameter could not be generated
        return None

    return actual_parameter


@st.composite
def Fragment_identifier_strategy(draw: Callable) -> str:
    while True:
        identifier = draw(st.from_regex(FRAGMENT_ID_PATTERN, fullmatch=True))
        try:
            identifier.encode()
            return identifier
        except UnicodeEncodeError as error:
            # make sure Python can encode it (e.g. it cannot handle '\udcbe')
            continue


@st.composite
def RegisterParameter_identifier_strategy(draw: Callable) -> str:
    while True:
        identifier = draw(st.from_regex(REGISTER_PARAM_ID_PATTERN, fullmatch=True))
        try:
            identifier.encode()
            return identifier
        except UnicodeEncodeError as error:
            # make sure Python can encode it (e.g. it cannot handle '\udcbe')
            continue


@st.composite
def Fragment_strategy(draw: Callable,
                      multi_statements_only: bool = False,
                      min_num_operations: int = 0,
                      max_num_operations: int = MAX_FRAGMENT_INSTRUCTIONS,
                      fragment_ids: Optional[Set[str]] = None,
                      statement_distribution: Optional[StatementDistribution] = None,
                      num_rn_regs_distribution: Optional[AliasMethod] = None,
                      num_sm_regs_distribution: Optional[AliasMethod] = None) -> BLEIR.Fragment:

    identifier = draw(Fragment_identifier_strategy())

    if fragment_ids is not None:
        # Make sure the fragment identifier is unique (APL constraint)
        while identifier in fragment_ids:
            identifier = draw(Fragment_identifier_strategy())

    param_ids = set()

    # Determine how many RN_REGs to give the fragment
    if num_rn_regs_distribution is None:
        num_rn_params = draw(st.integers(min_value=0, max_value=15))
    else:
        num_rn_params = num_rn_regs_distribution.sample()

    rn_params = []
    for _ in range(num_rn_params):
        param_id = draw(RegisterParameter_identifier_strategy())
        while param_id in param_ids:
            param_id = draw(RegisterParameter_identifier_strategy())
        param_ids.add(param_id)
        rn_params.append(BLEIR.RN_REG(identifier=param_id))

    # Determine how many SM_REGS to give the fragment.
    if num_sm_regs_distribution is None:
        num_sm_params = draw(st.integers(min_value=0, max_value=15))
    else:
        num_sm_params = num_sm_regs_distribution.sample()

    # Be sure to generate at least 1 SM_REG if there is at least one RN_REG
    if num_rn_params > 0 and num_sm_params == 0:
        num_sm_params = 1

    sm_params = []
    for _ in range(num_sm_params):
        param_id = draw(RegisterParameter_identifier_strategy())
        while param_id in param_ids:
            param_id = draw(RegisterParameter_identifier_strategy())
        param_ids.add(param_id)
        sm_params.append(BLEIR.SM_REG(identifier=param_id))

    parameters = rn_params + sm_params
    operations = draw(st.lists(Operation_strategy(rn_params, sm_params,
                                                  multi_statements_only=multi_statements_only,
                                                  statement_distribution=statement_distribution),
                               min_size=min_num_operations,
                               max_size=max_num_operations))
    return BLEIR.Fragment(
        identifier=identifier,
        parameters=parameters,
        operations=operations)


@st.composite
def FragmentCaller_strategy(draw: Callable,
                            fragment: BLEIR.Fragment) -> BLEIR.FragmentCaller:

    sm_reg_arena = SmRegArena()
    rn_reg_arena = RnRegArena()

    registers = []
    for formal_parameter in fragment.parameters:
        register = draw(AllocatedRegister_strategy(formal_parameter,
                                                   sm_reg_arena,
                                                   rn_reg_arena))
        registers.append(register)

    sm_reg_arena.free_all()
    rn_reg_arena.free_all()

    return BLEIR.FragmentCaller(
        fragment=fragment,
        registers=registers)


def colorize_rn_regs(draw: Callable, fragment: BLEIR.Fragment) -> Dict[BLEIR.RN_REG, int]:
    r"""Groups RN_REGs in the fragment into three groups: 0 (red), 1 (green), and 2 (blue).
        1. The red group consists of SBs in the range [0, 8)
        2. The green group consists of SBs in the range [8, 16)
        3. The blue group consists of SBs in the range [16, 24)

    This avoids WRITE conflicts in which cross-group SBs are written to within the same
    multi-statement. For example, take the following fragment:

    APL_FRAG foo(RN_REG s1_rp,
                 RN_REG s2_rp,
                 RN_REG s3_rp,
                 RN_REG s4_rp,
                 SM_REG fs_rp)
    {   {
        fs_rp: SB[s1_rp] = RL;
        fs_rp: SB[s2_rp] = GGL;
        }
        {
        fs_rp: SB[s2_rp, s3_rp] = RSP16;
        }
        {
        fs_rp: SB[s4_rp] = INV_GL;
        }   };

    The SBs (s1_rp, s2_rp, and s3_rp) must belong to the same group (color), but s4_rp may belong
    to any group. The transitive closure of s1_rp contains s2_rp and s3_rp (they are collocated
    within multi-statements), and as such, the group assigned to s1_rp must also be assigned to
    s2_rp and s3_rp. The transitive closure of s4_rp contains no other SB, so it may belong to any
    group (red, green, or blue).
    """

    # Traverse the Fragment graph and extract the un-resolved groups for each RN_REG
    walker = BLEIRWalker()
    analyzer = SBGroupAnalyzer()
    walker.walk(analyzer, fragment)

    # Collect the transitive closure for each RN_REG (its neighborhood) corresponding to its
    # multi-statements
    neighborhoods = {}
    for grouping in analyzer.groupings:
        for rn_reg in grouping:
            if rn_reg not in neighborhoods:
                neighborhoods[rn_reg] = set(grouping)
            else:
                neighborhoods[rn_reg] |= set(grouping)

    # Grow the neighborhoods around each RN_REG to its collocated neighborhoods
    for neighborhood in neighborhoods.values():
        pending = deque(neighborhood)
        while len(pending) > 0:
            rn_reg = pending.popleft()
            for neighbor in neighborhoods[rn_reg]:
                if neighbor not in neighborhood:
                    neighborhood.add(neighbor)
                    pending.append(neighbor)

    # Assign colors to each neighborhood such that the RN_REG in each neighborhood belongs to the
    # same group as its neighbors.
    colorings = {}
    for rn_reg, neighborhood in neighborhoods.items():
        if rn_reg not in colorings:
            color = draw(st.integers(min_value=0, max_value=2))
            for neighbor in neighborhood:
                colorings[neighbor] = color

    return colorings


LValueType = Union[BLEIR.SRC_EXPR, BLEIR.BROADCAST_EXPR, BLEIR.RN_REG]


def find_sm_reg_conflicts(lvalues_to_multi_statements_to_masks: Dict[LValueType, Dict[BLEIR.MultiStatement, List[BLEIR.MASK]]],
                          sm_reg_map: Dict[BLEIR.SM_REG, int]) -> Set[BLEIR.SM_REG]:

    conflicting_regs = set()

    for lvalue, multi_statements_to_masks in lvalues_to_multi_statements_to_masks.items():
        for multi_statement, masks in multi_statements_to_masks.items():
            existing_bits = 0x0000
            for mask in masks:
                mask_value = sm_reg_map[mask.sm_reg]
                # Check whether the resolved mask value conflicts with any of the previous resolved
                # mask values for the lvalue in the current multi-statement
                resolved_value = mask.resolve(mask_value)
                if existing_bits & resolved_value != 0x0000:
                    # If there is a conflict, regenerate all the SM_REGs for the lvalue in the
                    # current multi-statement (one of them could be 0xFFFF)
                    masked_regs = set(mask.sm_reg for mask in masks)
                    conflicting_regs |= masked_regs
                    continue
                else:
                    # If there are no conflicts, record the resolved bits for further comparison
                    existing_bits |= resolved_value

    return conflicting_regs


def generate_section_masks(draw: Callable,
                           fragment: BLEIR.Fragment,
                           max_attempts: int = 100) -> Optional[Dict[BLEIR.SM_REG, int]]:
    # When generating section mask values, the kinds of conflicts we want to avoid are overlapping
    # mask bits for assignments to the same lvalue within a multi-statement. As such, we may
    # localize our search for conflicts within the context of individial lvalues within each
    # multi-statement, which is done with the following map:
    lvalues_to_multi_statements_to_masks = {}

    for operation in fragment.operations:
        # We only care about multi-statements
        if not isinstance(operation, BLEIR.MultiStatement):
            continue

        for statement in operation:
            # We only care about assignments with masked lvalues
            if not isinstance(statement.operation, BLEIR.MASKED):
                continue

            mask = statement.operation.mask
            assignment = statement.operation.assignment

            # If the assignment type is WRITE, the lvalue consists of 1, 2, or 3 RN_REGs:
            if isinstance(assignment.operation, BLEIR.WRITE):
                # list of RN_REGs from SB_EXPR
                lvalues = assignment.operation.lvalue.parameters

            # Otherwise, the lvalue is either a single SRC_EXPR or BROADCAST_EXPR
            else:
                lvalues = [assignment.operation.lvalue]

            # Construct the mapping: LValue -> MultiStatement -> Mask
            for lvalue in lvalues:
                if lvalue not in lvalues_to_multi_statements_to_masks:
                    multi_statements_to_masks = {}
                    lvalues_to_multi_statements_to_masks[lvalue] = multi_statements_to_masks
                else:
                    multi_statements_to_masks = lvalues_to_multi_statements_to_masks[lvalue]

                if operation not in multi_statements_to_masks:
                    masks = []
                    multi_statements_to_masks[operation] = masks
                else:
                    masks = multi_statements_to_masks[operation]

                masks.append(mask)

    # Begin by randomly assigning values to the section masks
    sm_reg_map = {
        sm_reg: draw(st.integers(min_value=0x0000, max_value=0xFFFF))
        for sm_reg in fragment.sm_regs
    }

    # Find all the section mask values that conflict with each other within multi-statements
    conflicting_regs = find_sm_reg_conflicts(lvalues_to_multi_statements_to_masks, sm_reg_map)

    # Attempt to regenerate the conflicting values until there are no more conflicts
    num_attempts = 0
    while len(conflicting_regs) > 0:
        for sm_reg in conflicting_regs:
            sm_reg_map[sm_reg] = draw(st.integers(min_value=0x0000, max_value=0xFFFF))
        conflicting_regs = find_sm_reg_conflicts(lvalues_to_multi_statements_to_masks, sm_reg_map)

        num_attempts += 1
        if num_attempts >= max_attempts:
            break

    # Let the caller know if the section masks could not be generated within a reasonable number of
    # attempts
    if len(conflicting_regs) > 0:
        return None

    return sm_reg_map


@st.composite
def FragmentCallerCall_strategy(draw: Callable,
                                fragment_caller: BLEIR.FragmentCaller) -> Optional[BLEIR.FragmentCallerCall]:

    fragment = fragment_caller.fragment

    # Groups the fragment's RN_REGs into 3 categories: 0 (red), 1 (green), or 2 (blue).
    #     1. The red group consists of SBs in the range [0, 8)
    #     2. The green group consists of SBs in the range [8, 16)
    #     3. The blue group consists of SBs in the range [16, 24)
    colorings = colorize_rn_regs(draw, fragment)

    # Provides a set of SB values that may not be selected.
    # Each SB value drawn is added to this set so it is not drawn again.
    unavailable_sbs = set()

    sm_reg_map = generate_section_masks(draw, fragment)
    if sm_reg_map is None:
        # Failed to generate a valid combination of section masks
        return None

    actual_parameters = []

    # Generate the actual_parameters one-at-a-time to avoid conflicts
    for formal_parameter in fragment_caller.parameters:
        actual_parameter = draw(ActualParameter_strategy(formal_parameter,
                                                         colorings,
                                                         unavailable_sbs,
                                                         sm_reg_map))

        # Fail if the actual_parameter could not be generated
        if actual_parameter is None:
            return None

        # Remove the RN_REG value from the availability pool to avoid conflicts
        if isinstance(formal_parameter, BLEIR.RN_REG):
            unavailable_sbs.add(actual_parameter)

        actual_parameters.append(actual_parameter)

    return BLEIR.FragmentCallerCall(
        caller=fragment_caller,
        parameters=actual_parameters)


def samples_to_distribution(samples: numpy.ndarray, num_bins: int) -> numpy.ndarray:
    # In case there are any negative samples, shift the samples upward by their min, plus an
    # epsilon to avoid zero
    shifted_samples = samples + abs(samples.min()) + 1e-5

    # Make the max value in the samples 1.0 to help with binning
    scaled_samples = shifted_samples / shifted_samples.max()

    # Determine the bin indices for each sample
    bin_indices = numpy.floor(scaled_samples * (num_bins - 1)).astype(numpy.int64)

    # Generate the distribution by normalizing the histogram of binned samples
    distribution = numpy.zeros(num_bins, dtype=numpy.float64)
    for bin_index in bin_indices:
        distribution[bin_index] += 1.0

    if distribution.sum() > 0.0:
        distribution = distribution / distribution.sum()

    return distribution


@st.composite
def Snippet_strategy(draw: Callable) -> BLEIR.Snippet:
    fragment_callers = []
    fragment_caller_calls = []
    fragment_ids = set()

    seed = draw(st.integers(min_value=0, max_value=2**32-1))
    random = numpy.random.RandomState(seed=seed)

    # Sample statement types from a non-uniform distribution
    statement_distribution = StatementDistribution(random=random)

    # Number of samples used to generate the RN_REG and SM_REG distributions
    num_samples = 1_000

    # Sample the number of RN_REGs for a fragment from a normal distribution
    max_num_rn_regs = 16
    num_rn_regs_distribution = random.standard_normal(num_samples)
    num_rn_regs_distribution = samples_to_distribution(num_rn_regs_distribution, max_num_rn_regs)
    num_rn_regs_distribution = AliasMethod(num_rn_regs_distribution, random=random)

    # Sample the number of SM_REGS for a fragment from a beta distribution
    max_num_sm_regs = 16
    num_sm_regs_distribution = random.beta(2, 5, num_samples)
    num_sm_regs_distribution = samples_to_distribution(num_sm_regs_distribution, max_num_sm_regs)
    num_sm_regs_distribution = AliasMethod(num_sm_regs_distribution, random=random)

    # To avoid generating fragments and then fragment-caller-calls atop a subset of them, just to
    # discard the rest, determine up-front the number of fragment-caller-calls to generate and then
    # work backwards to generate corresponding fragments.
    num_calls = draw(st.integers(min_value=0, max_value=50))

    for _ in range(num_calls):

        # Generate a new fragment if a random variable evaluates to true or no fragments have yet
        # been generated
        if draw(st.booleans()) or len(fragment_callers) == 0:
            fragment = draw(Fragment_strategy(fragment_ids=fragment_ids,
                                              statement_distribution=statement_distribution,
                                              num_rn_regs_distribution=num_rn_regs_distribution,
                                              num_sm_regs_distribution=num_sm_regs_distribution))
            fragment_ids.add(fragment.identifier)
            fragment_caller = draw(FragmentCaller_strategy(fragment))
            fragment_callers.append(fragment_caller)
        else:
            # If at least one fragment has been generated and the random variable evaluates to
            # false, generate a new fragment-caller-call against an existing fragment
            fragment_caller = draw(st.sampled_from(fragment_callers))

        fragment_caller_call = draw(FragmentCallerCall_strategy(fragment_caller))

        # If the fragment_caller_call is None, one or more of its necessary parameters could not be
        # generated. Remove it from the availability pool and generate a new fragment-caller-call.
        if fragment_caller_call is None:
            fragment_callers.remove(fragment_caller)
            continue

        fragment_caller_calls.append(fragment_caller_call)

    # Make sure the half-bank is properly initialized before running the application
    fragment_caller_calls = tuple(boilerplate.INITIALIZERS + fragment_caller_calls)
    checksum = md5(pickle.dumps(fragment_caller_calls)).hexdigest()
    name = f"snip_{checksum}"

    return BLEIR.Snippet(
        name=name,
        examples=[],
        calls=fragment_caller_calls)


max_rn_regs_strategy = st.integers(1, NUM_RN_REGS)


@st.composite
def reserved_rn_regs_strategy(draw: Callable,
                              min_rn_regs: int = 1,
                              max_rn_regs: int = NUM_RN_REGS) -> List[int]:
    return draw(st.lists(st.integers(0, NUM_RN_REGS - 1),
                         min_size=min_rn_regs,
                         max_size=max_rn_regs,
                         unique=True))
