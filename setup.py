from __future__ import annotations

import argparse
from pathlib import Path
import sys

from setuptools import setup as setuptools_setup


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


PACKAGING_COMMANDS = {
    "bdist_wheel",
    "build",
    "build_py",
    "develop",
    "dist_info",
    "egg_info",
    "install",
    "sdist",
}


def is_packaging_invocation(argv: list[str]) -> bool:
    for arg in argv[1:]:
        if arg.startswith("-"):
            continue
        return arg in PACKAGING_COMMANDS
    return False

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap a local Arignan user installation.")
    parser.add_argument("--dev", action="store_true", help="Install the repository with dev dependencies.")
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use the default light local model for both normal and light answer modes during setup.",
    )
    parser.add_argument("--app-home", type=Path, default=None, help="Override the Arignan application home directory.")
    parser.add_argument(
        "--llm-backend",
        default=None,
        help="Override local_llm_backend in settings.json before model downloads begin.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override local_llm_model in settings.json before model downloads begin.",
    )
    return parser


def main() -> int:
    if is_packaging_invocation(sys.argv):
        setuptools_setup(
            options={
                "egg_info": {"egg_base": ".setuptools"},
                "build": {"build_base": ".setuptools/build"},
            }
        )
        return 0
    from arignan.setup_flow import render_summary, run_setup

    args = build_parser().parse_args()
    print("Starting Arignan setup...")
    try:
        result = run_setup(
            dev=args.dev,
            lightweight=args.lightweight,
            app_home=args.app_home,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            progress=print,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(render_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
