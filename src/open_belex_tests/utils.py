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

from copy import deepcopy
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional
from warnings import warn

import numpy as np

import hypothesis
import hypothesis.strategies as st
from hypothesis import given
# from hypothesis.extra.numpy import arrays

from open_belex.bleir.builders import SnippetBuilder
from open_belex.bleir.interpreters import BLEIRInterpreter
from open_belex.bleir.types import (CFunction, CFunctionMetadata, CParameter,
                                    Example, Snippet, SnippetMetadata,
                                    ValueParameter, build_examples)
from open_belex.bleir.virtual_machines import (VIRTUAL_MACHINE,
                                               BLEIRVirtualMachine, Feature)
from open_belex.common.constants import (NSB, NUM_L1_ROWS, NUM_L2_ROWS,
                                         NUM_PLATS_PER_APUC,
                                         NUM_PLATS_PER_HALF_BANK)
from open_belex.common.seu_layer import SEULayer
from open_belex.common.stack_manager import StackManager
from open_belex.diri.half_bank import DIRI, NUM_LGL_PLATS
from open_belex.literal import SNIPPET_BUILDER, VR
from open_belex.utils.example_utils import convert_to_bool, convert_to_u16
from open_belex.utils.path_utils import user_tmp


# NOTE: The hypothesis strategy is WAY TOO SLOW !!!
# NOTE: Using the hypothesis numpy arrays strategy without `unique=True`
# generates mostly empty VRs. To avoid this, I added `unique=True` to generate
# completely random VRs. This destroyed testing performance making it
# impractical to complete even a single test. I'm not sure what the reason is
# for the poor performance because we need only 2048 integers sampled from the
# range [0x0000, 0xFFFF]. I replaced this strategy with one that sampled
# without replacement from a preallocated list of plat values and that was much
# faster. The results did not look random, though (e.g. sometimes all
# sequential VR plats differed exactly by 2), so I tried sampling with
# replacement from the same list. The results looked sufficiently random and
# the performance was good so we continue to use sampling with replacement from
# the preallocated plat values.

# vr_strategy = arrays(
#     dtype=np.uint16,
#     shape=NUM_PLATS_PER_APUC,
#     elements=st.integers(min_value=0x0000, max_value=0xFFFF),
#     unique=True)

AVAILABLE_PLAT_VALUES = np.array(list(range(1 + 0xFFFF)), dtype=np.uint16)
AVAILABLE_GL_VALUES = np.array([0, 1], dtype=np.uint16)
AVAILABLE_GGL_VALUES = np.array(list(range(1 + 0xF)), dtype=np.uint16)

OUTPUT_DIRS: str = "output_dirs"


seed_strategy = st.integers(min_value=0, max_value=2**32-1)


@st.composite
def l1_strategy(draw, seed: Optional[int] = None) -> np.ndarray:
    if seed is None:
        seed = draw(seed_strategy)
    random = np.random.RandomState(seed)

    min_index = 0
    max_index = len(AVAILABLE_GGL_VALUES) - 1
    indices = random.randint(low=min_index,
                             high=max_index,
                             size=(NUM_L1_ROWS * NUM_PLATS_PER_APUC)).astype(np.uint16)
    samples = AVAILABLE_GGL_VALUES[indices]
    return samples


@st.composite
def l2_strategy(draw, seed: Optional[int] = None) -> np.ndarray:
    if seed is None:
        seed = draw(seed_strategy)
    random = np.random.RandomState(seed)

    min_index = 0
    max_index = len(AVAILABLE_GL_VALUES) - 1
    indices = random.randint(low=min_index,
                             high=max_index,
                             size=(NUM_L2_ROWS * NUM_PLATS_PER_HALF_BANK)).astype(np.uint16)
    samples = AVAILABLE_GL_VALUES[indices]
    return samples


@st.composite
def vr_strategy(draw: Callable,
                num_plats: int = NUM_PLATS_PER_APUC,
                seed: Optional[int] = None) -> np.ndarray:
    if seed is None:
        seed = draw(seed_strategy)
    random = np.random.RandomState(seed)

    min_index = 0
    max_index = len(AVAILABLE_PLAT_VALUES) - 1
    indices = random.randint(low=min_index,
                             high=max_index,
                             size=num_plats).astype(np.uint16)
    samples = AVAILABLE_PLAT_VALUES[indices]
    return samples


@st.composite
def gl_strategy(draw, seed: Optional[int] = None) -> np.ndarray:
    if seed is None:
        seed = draw(seed_strategy)
    random = np.random.RandomState(seed)

    min_index = 0
    max_index = len(AVAILABLE_GL_VALUES) - 1
    indices = random.randint(low=min_index,
                             high=max_index,
                             size=NUM_PLATS_PER_APUC).astype(np.uint16)
    samples = AVAILABLE_GL_VALUES[indices]
    return samples


@st.composite
def ggl_strategy(draw, seed: Optional[int] = None) -> np.ndarray:
    if seed is None:
        seed = draw(seed_strategy)
    random = np.random.RandomState(seed)

    min_index = 0
    max_index = len(AVAILABLE_GGL_VALUES) - 1
    indices = random.randint(low=min_index,
                             high=max_index,
                             size=NUM_PLATS_PER_APUC).astype(np.uint16)
    samples = AVAILABLE_GGL_VALUES[indices]
    return samples


@st.composite
def lgl_strategy(draw, seed: Optional[int] = None) -> np.ndarray:
    if seed is None:
        seed = draw(seed_strategy)
    random = np.random.RandomState(seed)

    min_index = 0
    max_index = len(AVAILABLE_GL_VALUES) - 1
    indices = random.randint(low=min_index,
                             high=max_index,
                             size=NUM_LGL_PLATS).astype(np.uint16)
    samples = AVAILABLE_GL_VALUES[indices]
    return samples


@st.composite
def rsp16_strategy(draw, seed: Optional[int] = None) -> np.ndarray:
    if seed is None:
        seed = draw(seed_strategy)
    random = np.random.RandomState(seed)

    min_index = 0
    max_index = len(AVAILABLE_PLAT_VALUES) - 1
    indices = random.randint(low=min_index,
                             high=max_index,
                             size=NUM_PLATS_PER_APUC // 16).astype(np.uint16)
    samples = AVAILABLE_PLAT_VALUES[indices]
    return samples


@st.composite
def rsp256_strategy(draw, seed: Optional[int] = None) -> np.ndarray:
    if seed is None:
        seed = draw(seed_strategy)
    random = np.random.RandomState(seed)

    min_index = 0
    max_index = len(AVAILABLE_PLAT_VALUES) - 1
    indices = random.randint(low=min_index,
                             high=max_index,
                             size=NUM_PLATS_PER_APUC // 256).astype(np.uint16)
    samples = AVAILABLE_PLAT_VALUES[indices]
    return samples


@st.composite
def rsp2k_strategy(draw, seed: Optional[int] = None) -> np.ndarray:
    if seed is None:
        seed = draw(seed_strategy)
    random = np.random.RandomState(seed)

    min_index = 0
    max_index = len(AVAILABLE_PLAT_VALUES) - 1
    indices = random.randint(low=min_index,
                             high=max_index,
                             size=NUM_PLATS_PER_APUC // 2048).astype(np.uint16)
    samples = AVAILABLE_PLAT_VALUES[indices]
    return samples


@st.composite
def rsp32k_strategy(draw, seed: Optional[int] = None) -> np.ndarray:
    if seed is None:
        seed = draw(seed_strategy)
    random = np.random.RandomState(seed)

    min_index = 0
    max_index = len(AVAILABLE_PLAT_VALUES) - 1
    indices = random.randint(low=min_index,
                             high=max_index,
                             size=NUM_PLATS_PER_APUC // 32768).astype(np.uint16)
    samples = AVAILABLE_PLAT_VALUES[indices]
    return samples


@st.composite
def u16_strategy(draw, min_value=0x0000, max_value=0xFFFF, seed: Optional[int] = None) -> int:
    if seed is None:
        seed = draw(seed_strategy)
    random = np.random.RandomState(seed)
    u16 = np.uint16(random.randint(min_value, max_value + 1))
    return u16


Mask_strategy = u16_strategy


def belex_property_test(belex_fn: Callable,
                        interpret: Optional[bool] = None,
                        generate_code: Optional[bool] = None,
                        max_examples: int = 10,
                        out_nym: str = "out") -> Callable:

    def decorator(test_fn: Callable) -> Callable:
        nonlocal belex_fn

        belex_fn_sig = signature(belex_fn)

        arg_nyms = list(belex_fn_sig.parameters.keys())[1:]

        out_nym = arg_nyms[0]
        arg_nyms = arg_nyms[1:]

        test_fn_sig = signature(test_fn)
        test_nyms = list(test_fn_sig.parameters.keys())
        if test_nyms != arg_nyms:
            raise AssertionError(f"Expected {test_nyms} to be {arg_nyms}")

        param_row_numbers = list(range(1, 1 + len(arg_nyms)))

        out_row_number = 0
        assert out_nym not in arg_nyms

        def wrapper() -> None:
            nonlocal belex_fn, test_fn, arg_nyms, out_nym

            arg_specs = list(belex_fn_sig.parameters.items())[1:]

            snippet_name = test_fn.__name__[len("test_"):]
            examples = []

            @hypothesis.settings(deadline=None,
                                 max_examples=max_examples,  # 3
                                 print_blob=True,
                                 # Options: quiet, normal, verbose, debug
                                 verbosity=hypothesis.Verbosity.normal,
                                 report_multiple_bugs=False,
                                 database=None,  # no storage
                                 phases=(
                                     # hypothesis.Phase.explicit,
                                     # hypothesis.Phase.reuse,
                                     hypothesis.Phase.generate,
                                     # hypothesis.Phase.target,
                                     # hypothesis.Phase.shrink
                                 ),
                                 suppress_health_check=[
                                     hypothesis.HealthCheck.data_too_large,
                                     hypothesis.HealthCheck.large_base_example,
                                     hypothesis.HealthCheck.too_slow,
                                 ])
            @given(data=st.data())
            def run_all_tests(data) -> None:
                nonlocal belex_fn, test_fn, out_nym, out_row_number, \
                    examples, snippet_name, reservations, arg_specs

                numpy_seed = data.draw(st.integers(min_value=0, max_value=2**32-1))
                random = np.random.RandomState(numpy_seed)

                kwarg_vals = {}
                for _, arg_spec in arg_specs:
                    if arg_spec.name != out_nym:
                        if arg_spec.annotation is VR:
                            seed = random.randint(0, 2**32)
                            kwarg_vals[arg_spec.name] = data.draw(vr_strategy(seed=seed))
                        else:
                            raise RuntimeError(
                                f"Unsupported parameter type for "
                                f"'{arg_spec.name}' in '{test_fn.__name__}': "
                                f"{arg_spec.annotation}")

                parameters = []
                for index, (arg_nym, arg_val) in enumerate(kwarg_vals.items()):
                    if isinstance(arg_val, int):
                        continue

                    row_number = 1 + index
                    parameter = ValueParameter(identifier=arg_nym,
                                               row_number=row_number,
                                               value=deepcopy(arg_val))
                    parameters.append(parameter)

                out_val = test_fn(**kwarg_vals).astype(np.uint16)
                expected_value = ValueParameter(identifier=out_nym,
                                                row_number=out_row_number,
                                                value=out_val)

                example = Example(expected_value=expected_value,
                                  parameters=parameters)

                reservations = None
                features = None
                target = "baryon"
                if StackManager.has_elem(VIRTUAL_MACHINE):
                    vm = StackManager.peek(VIRTUAL_MACHINE)
                    _interpret = vm.interpret
                    reservations = vm.reservations
                    features = vm.features
                    target = vm.target
                elif interpret is None:
                    _interpret = True
                else:
                    _interpret = interpret

                if _interpret:
                    vm = BLEIRVirtualMachine(
                        interpret=True,
                        generate_code=False,
                        reservations=reservations,
                        features=features,
                        target=target)

                    diri = vm.interpreter.diri
                    StackManager.push(DIRI.__STACK_NYM__, diri)
                    for parameter in parameters:
                        diri.hb[parameter.row_number] = \
                            convert_to_bool(parameter.value)

                    snippet = Snippet(
                        name=snippet_name,
                        examples=examples,
                        calls=[belex_fn(out_row_number, *param_row_numbers)],
                        metadata={
                            SnippetMetadata.TARGET: target,
                        })

                    assert diri is StackManager.pop(DIRI.__STACK_NYM__)

                    vm.compile(snippet)
                    vm.assert_no_interpreter_failures()

                examples.append(example)

            run_all_tests()

            if StackManager.has_elem(VIRTUAL_MACHINE):
                vm = StackManager.peek(VIRTUAL_MACHINE)
                output_dir = vm.output_dir
                print_params = vm.print_params
                reservations = vm.reservations
                features = vm.features
                target = vm.target
            else:
                output_dir = user_tmp() / "blecci" / snippet_name
                print_params = False
                reservations = None
                features = None
                target = "baryon"

            if target == "baryon":
                source_file = f"{snippet_name}-funcs.c"
                header_file = f"{snippet_name}-funcs.h"

            snippet = Snippet(
                name=snippet_name,
                examples=examples,
                calls=[belex_fn(out_row_number, *param_row_numbers)],
                metadata={
                    SnippetMetadata.SOURCE_FILE: source_file,
                    SnippetMetadata.HEADER_FILE: header_file,
                    SnippetMetadata.TARGET: target,
                })

            if interpret is None:
                _interpret = False
            else:
                _interpret = interpret

            if generate_code is None:
                _generate_code = True
            else:
                _generate_code = generate_code

            vm = BLEIRVirtualMachine(
                interpret=_interpret,
                generate_code=_generate_code,
                output_dir=output_dir,
                print_params=print_params,
                reservations=reservations,
                features=features,
                target=target)

            vm.compile(snippet)

            if StackManager.has_elem(OUTPUT_DIRS):
                output_dirs = StackManager.peek(OUTPUT_DIRS)
                output_dirs.append(output_dir)

        wrapper.__name__ = test_fn.__name__
        wrapper.__low_level_test__ = True
        return wrapper

    return decorator


InitVR = Callable[[int, Optional[np.ndarray]], None]


def parameterized_belex_test(
        test_fn: Optional[Callable] = None,
        interpret: bool = True,
        generate_code: bool = True,
        repeatably_randomize_half_bank: bool = False,
        features: Optional[Dict[str, bool]] = None) -> Callable:

    if features is not None:
        features = {Feature.find_by_value(feature): is_enabled
                    for feature, is_enabled in features.items()}


    def decorator(test_fn):
        nonlocal interpret

        snippet_name = test_fn.__name__[len("test_"):]
        serial_id = 0

        def wrapper(*args, **kwargs):
            nonlocal test_fn, snippet_name, generate_code, \
                serial_id, repeatably_randomize_half_bank

            diri = DIRI.push_context()
            seu = SEULayer.push_context()
            interpreter = BLEIRInterpreter.push_context(diri=diri)

            if repeatably_randomize_half_bank:
                diri.repeatably_randomize_half_bank()

            if StackManager.has_elem(VIRTUAL_MACHINE):
                vm = StackManager.peek(VIRTUAL_MACHINE)
                print_params = vm.print_params
                reservations = vm.reservations
                local_features = deepcopy(vm.features)
                target = vm.target
            else:
                print_params = False
                reservations = None
                local_features = None
                target = "baryon"

            vm = BLEIRVirtualMachine(
                interpret=False,
                generate_code=False,
                diri=diri,
                print_params=print_params,
                reservations=reservations,
                features=local_features,
                extra_features=features,
                target=target)

            snippet_builder = StackManager.push(
                SNIPPET_BUILDER,
                SnippetBuilder(vm))

            ### ======================
            ### BEGIN: Monkey Patching
            ### ======================

            initial_state = None

            # Previously, the target was compiled to run it through the
            # interpreter but the instructions for the fragment caller call have
            # already been run through the interpreter as part of making it
            # debuggable. All we care about, now, is grabbing the initial state.
            def compile(target):
                # nonlocal initial_state
                # if isinstance(target, FragmentCallerCall) and initial_state is None:
                #     # No need to deepcopy since this will be a copy with the new API
                #     # initial_state = deepcopy(diri.hb[:NSB])
                #     initial_state = diri.hb[:NSB]
                return target
            _compile = vm.compile
            vm.compile = compile

            _visit_operation = interpreter.visit_operation
            def visit_operation(operation):
                nonlocal initial_state
                if initial_state is None:
                    initial_state = diri.hb[:NSB]
                return _visit_operation(operation)
            interpreter.visit_operation = visit_operation

            ### ====================
            ### END: Monkey Patching
            ### ====================

            used_sbs = set()
            StackManager.push("used_sbs", used_sbs)

            try:
                out_sb = test_fn(diri, *args, **kwargs)
            finally:
                assert snippet_builder is StackManager.pop(SNIPPET_BUILDER)
                assert interpreter is BLEIRInterpreter.pop_context()
                assert diri is DIRI.pop_context()
                assert used_sbs is StackManager.pop("used_sbs")

            # Restore vm.compile for codegen and expected value assertion
            vm.compile = _compile
            interpreter.visit_operation = _visit_operation

            serial_id += 1

            if initial_state is None:
                # Probably a negative test case
                return

            final_state = deepcopy(diri.hb[:NSB])

            expected_value = None
            parameters = []

            if out_sb is not None:
                used_sbs.add(out_sb)
                final_vr = final_state[out_sb]
                vr_out = convert_to_u16(final_vr)
                out_nym = f"out"
                expected_value = (out_nym, out_sb, vr_out)

            # NOTE: If no output SB is returned the SB corresponding to the
            # left-most changed VR is assumed to be the output SB, where
            # iterating from left-to-right implies iterating from 0 to NSB.
            for sb in range(NSB):
                initial_vr = initial_state[sb]
                final_vr = final_state[sb]

                if sb in used_sbs:
                    vr_param = convert_to_u16(initial_vr)
                    param_nym = f"param_{sb}"
                    parameters.append((param_nym, sb, vr_param))
                    if expected_value is None:
                        vr_out = convert_to_u16(final_vr)
                        out_nym = f"out_{sb}"
                        expected_value = (out_nym, sb, vr_out)

            if expected_value is None:
                warn("No expected value given.")
                return

            if not generate_code:
                return

            examples = build_examples([[expected_value, parameters]])

            if serial_id == 1:
                name = snippet_name
            else:
                name = f"{snippet_name}_{serial_id}"

            if StackManager.has_elem(VIRTUAL_MACHINE):
                vm = StackManager.peek(VIRTUAL_MACHINE)
                output_dir = vm.output_dir

                if serial_id > 1:
                    output_nym = f"{output_dir.name}_{serial_id}"
                    output_dir = output_dir.parent / output_nym
            else:
                output_dir = user_tmp() / "blecci" / name

            snippet_builder.build(
                name=name,
                examples=examples,
                interpret=interpret,
                generate_code=generate_code,
                output_dir=output_dir)

            if StackManager.has_elem(OUTPUT_DIRS):
                output_dirs = StackManager.peek(OUTPUT_DIRS)
                output_dirs.append(output_dir)

            assert seu is SEULayer.pop_context()

        wrapper.__name__ = test_fn.__name__
        wrapper.__low_level_test__ = True
        return wrapper

    if test_fn is not None:
        return decorator(test_fn)

    return decorator


def seu_context(fn: Callable) -> Callable:

    def wrapper(*args, **kwargs) -> Any:
        seu = SEULayer.push_context()
        try:
            retval = fn(seu, *args, **kwargs)
            return retval
        finally:
            assert seu is SEULayer.pop_context()

    wrapper.__name__ = fn.__name__
    return wrapper


def c_caller(fn: Callable) -> Callable:
    identifier = fn.__name__

    spec = getfullargspec(fn)
    formal_parameters = []
    for index, (name, kind) in enumerate(spec.annotations.items()):
        if index > 1 and name != "return":
            c_parameter = CParameter(name=name,
                                     kind=kind)
            formal_parameters.append(c_parameter)
    formal_parameters = tuple(formal_parameters)

    def wrapper(*args, **kwargs):
        diri = DIRI.context()
        retval = fn(diri, *args, **kwargs)
        return retval

    metadata = {
        CFunctionMetadata.PYTHON_FUNCTION: wrapper,
    }

    c_function = CFunction(identifier=identifier,
                           formal_parameters=formal_parameters,
                           metadata=metadata)

    def decorator(*args, **kwargs):
        c_function_call = c_function(*args, **kwargs)
        if StackManager.has_elem("SnippetBuilder"):
            snippet_builder = StackManager.peek("SnippetBuilder")
            snippet_builder.append(c_function_call)
        return c_function_call

    decorator.__name__ = identifier
    return decorator
