"""
Run open-ended evaluation with the domain-routed prompt specialization variant.

This wrapper delegates to gpt_free_text_eval_refined.py while forcing
`--variant domain_routed` unless explicitly overridden.
"""

import sys


def has_variant_override(args: list[str]) -> bool:
    """
    Return True when variant is already provided by the user.

    Supports both:
    - --variant baseline
    - --variant=baseline
    """
    for arg in args:
        if arg == "--variant" or arg.startswith("--variant="):
            return True
    return False


def apply_default_variant(argv: list[str]) -> list[str]:
    """
    Append `--variant domain_routed` when no explicit override is present.

    Args:
        argv: Full argv list including program name

    Returns:
        The same argv list, potentially extended in-place
    """
    if not has_variant_override(argv[1:]):
        argv.extend(["--variant", "domain_routed"])
    return argv


def run(argv: list[str] | None = None) -> None:
    """Entry point for script and tests."""
    if argv is None:
        argv = sys.argv
    apply_default_variant(argv)

    # Local import keeps this wrapper testable without heavy runtime deps.
    from gpt_free_text_eval_refined import main

    main()


if __name__ == "__main__":
    run()
