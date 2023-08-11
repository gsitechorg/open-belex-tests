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
