"""Run open-ended evaluation with discriminative-question variant by default."""

from __future__ import annotations

import sys


def has_variant_override(args: list[str]) -> bool:
    for arg in args:
        if arg == "--variant" or arg.startswith("--variant="):
            return True
    return False


def apply_default_variant(argv: list[str]) -> list[str]:
    if not has_variant_override(argv[1:]):
        argv.extend(["--variant", "discriminative_question"])
    return argv


def run(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = list(sys.argv)
    else:
        argv = list(argv)

    apply_default_variant(argv)

    from gpt_free_text_eval_refined import main

    prev_argv = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = prev_argv


if __name__ == "__main__":
    run()
