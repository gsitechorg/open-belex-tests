r"""
By Dylon Edwards
"""

from typing import List

import pytest

from hypothesis import given

from open_belex.common.stack_manager import StackManager
from open_belex.utils.config_utils import CONFIG, belex_config
from open_belex_tests.strategies import (max_rn_regs_strategy,
                                         reserved_rn_regs_strategy)


@given(max_rn_regs=max_rn_regs_strategy,
       reserved_rn_regs=reserved_rn_regs_strategy())
def test_belex_config(max_rn_regs: int, reserved_rn_regs: List[int]):

    @belex_config(max_rn_regs=max_rn_regs,
                  reservations={
                      "rn_regs": reserved_rn_regs,
                  })
    def check_belex_config():
        StackManager.assert_has_elem(CONFIG)
        config = StackManager.peek(CONFIG)

        assert "max_rn_regs" in config
        assert max_rn_regs == config["max_rn_regs"]

        assert "reservations" in config
        reservations = config["reservations"]

        assert "rn_regs" in reservations
        assert reserved_rn_regs == reservations["rn_regs"]

    check_belex_config()


def test_invalid_belex_config():
    with pytest.raises(RuntimeError):
        belex_config(invalid_attr="value")
