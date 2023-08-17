r"""
By Dylon Edwards
"""

import logging
import logging.handlers
import sys
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import click

from open_belex.apl_optimizations import APL_OPTIMIZATIONS
from open_belex.bleir.types import FragmentCaller, Snippet, SnippetMetadata
from open_belex.bleir.virtual_machines import (VIRTUAL_MACHINE,
                                               BLEIRVirtualMachine, Feature)
from open_belex.common.stack_manager import StackManager
from open_belex.decorators import build_examples, collect_actual_values
from open_belex.utils.log_utils import LogLevel, init_logger
from open_belex.utils.script_utils import (collect_config, collect_log_level,
                                           collect_optimizations,
                                           enable_optimizations_by_level,
                                           for_each_belex_caller,
                                           generate_belex_examples_meson_build)

SCRIPT_NAME = "generate-belex-examples"

OUTPUT_DIRS: str = "output_dirs"

LOGGER = logging.getLogger()


def generate_belex_example(
        output_dir: Path,
        belex_caller_or_test_fn: Union[FragmentCaller, Callable],
        source_file: Optional[Path],
        header_file: Optional[Path],
        generate_apl_sources: bool,
        generate_entry_point: bool,
        print_params: bool,
        config: Dict[str, Any],
        optimizations: Sequence[Callable],
        features: Dict[str, bool],
        target: str) -> Optional[Sequence[Path]]:

    outer_fn = None
    if hasattr(belex_caller_or_test_fn, "is_hypothesis_test"):
        outer_fn = belex_caller_or_test_fn
        belex_caller_or_test_fn = belex_caller_or_test_fn.hypothesis.inner_test

    if hasattr(belex_caller_or_test_fn, "__high_level_block__"):
        belex_optimizer = belex_caller_or_test_fn
        fragment_caller = belex_optimizer(optimizations=optimizations)
        examples = build_examples(fragment_caller)
        actual_values = collect_actual_values(examples)
        fragment_caller_call = fragment_caller(*actual_values)
        fragment = fragment_caller_call.fragment
        snippet_name = fragment.identifier

        if generate_entry_point:
            # example_dir = output_dir / f"test_belex_{snippet_name}"
            example_dir = output_dir / f"test_hlb_{snippet_name}"
        else:
            example_dir = output_dir

        if source_file is None:
            source_file = f"{snippet_name}-funcs.c"

        elif generate_entry_point and not source_file.exists():
            source_file = source_file.name

        if header_file is None:
            header_file = f"{snippet_name}-funcs.h"

        elif generate_entry_point and not header_file.exists():
            header_file = header_file.name

        snippet = Snippet(name=snippet_name,
                          examples=examples,
                          calls=[fragment_caller_call],
                          # calls=INITIALIZERS,
                          # library_callers=[belex_caller],
                          metadata={
                              SnippetMetadata.HEADER_FILE: header_file,
                              SnippetMetadata.SOURCE_FILE: source_file,
                              SnippetMetadata.TARGET: target,
                          })

        virtual_machine = BLEIRVirtualMachine(
            output_dir=example_dir,
            interpret=False,
            generate_code=True,
            generate_apl_sources=generate_apl_sources,
            generate_entry_point=generate_entry_point,
            print_params=print_params,
            reservations=config["reservations"],
            features=features,
            target=target)

        virtual_machine.compile(snippet)
        virtual_machine.assert_no_interpreter_failures()
        return [example_dir]

    elif hasattr(belex_caller_or_test_fn, "__low_level_test__"):
        if outer_fn is not None:
            belex_test_fn = outer_fn
        else:
            belex_test_fn = belex_caller_or_test_fn

        example_dir = \
            output_dir / f"test_llb_{belex_test_fn.__name__[len('test_'):]}"

        # Prototype VM for testing use
        virtual_machine = BLEIRVirtualMachine(
            interpret=False,
            generate_code=True,
            output_dir=example_dir,
            print_params=print_params,
            reservations=config["reservations"],
            features=features,
            target=target)

        output_dirs = []
        StackManager.push(OUTPUT_DIRS, output_dirs)
        StackManager.push(VIRTUAL_MACHINE, virtual_machine)
        belex_test_fn()
        assert virtual_machine is StackManager.pop(VIRTUAL_MACHINE)
        assert output_dirs is StackManager.pop(OUTPUT_DIRS)

        return output_dirs

    return None


@click.command()
@click.option("-O", "--optimization-level", "optimization_level",
              help="Specifies the optimization level.",
              type=click.IntRange(0, 4), default=2, show_default=True)
@click.option("-f", "--enable-optimization", "optimizations",
              help="Specifies an optimization over the generated code.",
              type=click.Choice(list(chain(APL_OPTIMIZATIONS.keys(),
                                           Feature.values(),
                                           [f"no-{feature}"
                                            for feature in Feature]))),
              multiple=True,
              callback=collect_optimizations)
@click.option("-b", "--belex-dir", "belex_dir",
              help="Path to the folder containing the BELEX files.",
              type=click.Path(exists=True, file_okay=False),
              required=False)
@click.option("-o", "--output-dir", "output_dir",
              help="Where to place the generated examples.",
              type=click.Path(exists=False, file_okay=False))
@click.option("-s", "--source-file", "source_file",
              help="Path to the APL file to generate (should end in .apl).",
              type=click.Path(exists=False, file_okay=True, dir_okay=False),
              required=False)
@click.option("-h", "--header-file", "header_file",
              help="Path to the APL file to generate (should end in .apl.h). "
                   "[Default: {source_file}.apl.h]",
              type=click.Path(exists=False, file_okay=True, dir_okay=False),
              required=False)
@click.option("--apl-sources/--no-apl-sources", "generate_apl_sources",
              help="Whether to generate the APL sources and headers.",
              default=True)
@click.option("--entry-point/--no-entry-point", "generate_entry_point",
              help="Whether to generate a main function and corresponding "
                   "meson.build for the applications.",
              default=True)
@click.option("--include-high-level/--no-include-high-level", "include_high_level",
              help="Whether to include high-level BELEX frags and tests.",
              default=True)
@click.option("--include-low-level/--no-include-low-level", "include_low_level",
              help="Whether to include low-level BELEX frags and tests.",
              default=True)
@click.option("--include-tests/--no-include-tests", "include_tests",
              help="Whether to include BELEX tests.",
              default=True)
@click.option("--target", "target",
              help="Type of sources to generate.",
              type=click.Choice(["baryon"]),
              default="baryon",
              required=False)
@click.option("--config", "config",
              help="Specifies path to the BELEX config YAML file.",
              callback=collect_config,
              required=False)
@click.option("--log-level", "log_level",
              help="Specifies the verbosity of output from the compiler.",
              type=click.Choice(LogLevel.names()),
              default=LogLevel.DEFAULT.name,
              callback=collect_log_level,
              required=False)
@click.option("--print-params/--no-print-params", "print_params",
              help="Whether to print the input parameters alongside the actual "
                   "and expected output values.",
              default=False)
@click.argument("belex_files", nargs=-1)
@enable_optimizations_by_level
def main(**kwargs):
    """Generates unit tests for each BELEX function found in the specified
    directory. The unit tests are designed to be run by the C-sim."""

    global LOGGER, SCRIPT_NAME
    init_logger(LOGGER, SCRIPT_NAME, log_level=kwargs["log_level"])

    for arg, val in kwargs.items():
        LOGGER.info("%s = %s", arg, val)

    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    belex_files = kwargs["belex_files"]
    source_file = kwargs["source_file"]
    header_file = kwargs["header_file"]

    if len(belex_files) > 1 \
    and (source_file is not None \
            or header_file is not None):
        raise RuntimeError(
            f"You may specify only one belex_file if you specify the source "
            f"and/or header: {belex_files}")

    if source_file is not None:
        source_file = Path(source_file)

    if header_file is not None:
        header_file = Path(header_file)

    # Avoid duplicate examples (by name)
    example_dirs = set()
    manifest = {}

    def callback(belex_file, belex_caller_or_test_fn):
        nonlocal example_dirs, kwargs, header_file, manifest, source_file, \
            output_dir

        output_dirs = generate_belex_example(output_dir,
                                             belex_caller_or_test_fn,
                                             source_file, header_file,
                                             kwargs["generate_apl_sources"],
                                             kwargs["generate_entry_point"],
                                             kwargs["print_params"],
                                             kwargs["config"],
                                             kwargs["optimizations"]["high-level"],
                                             kwargs["optimizations"]["low-level"],
                                             kwargs["target"])

        if kwargs["generate_entry_point"] and output_dirs is not None:
            for example_dir in output_dirs:
                example_dirs.add(example_dir)
                manifest[example_dir.name] = belex_file

    for_each_belex_caller(kwargs["belex_dir"],
                          belex_files,
                          callback,
                          kwargs["optimizations"]["high-level"],
                          include_high_level=kwargs["include_high_level"],
                          include_low_level=kwargs["include_low_level"],
                          include_tests=kwargs["include_tests"])

    if kwargs["generate_entry_point"]:
        LOGGER.info(f"Generating examples in {output_dir} ...")
        example_dirs = list(example_dirs)
        generate_belex_examples_meson_build(output_dir, example_dirs, manifest)

    LOGGER.info(f"Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Failed to generate BELEX examples")
        sys.exit(1)
