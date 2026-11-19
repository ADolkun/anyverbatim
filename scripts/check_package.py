"""Smoke-test the built wheel from an environment that holds only its runtime dependencies.

Run by `make package-check` through `uv run --isolated --no-project --with dist/<wheel>`, so an import of a dev-only
package, a file missing from the wheel or a broken console script fails here instead of for the library's users.
"""

import importlib
import importlib.resources
import subprocess
import sys
from pathlib import Path

import anyverbatim

PUBLIC_MODULES = ("anyverbatim.config", "anyverbatim.contract", "anyverbatim.errors")


def main() -> None:
    if "site-packages" not in Path(anyverbatim.__file__).parts:
        sys.exit(f"imported {anyverbatim.__file__}, not the installed wheel")
    if not importlib.resources.files(anyverbatim).joinpath("py.typed").is_file():
        sys.exit("the wheel is missing py.typed")
    for name in anyverbatim.__all__:
        getattr(anyverbatim, name)
    for module in PUBLIC_MODULES:
        importlib.import_module(module)
    subprocess.run([Path(sys.executable).parent / "anyverbatim", "--help"], check=True, stdout=subprocess.DEVNULL)
    print(f"anyverbatim {anyverbatim.__version__}: wheel imports, public API resolves, CLI starts")


if __name__ == "__main__":
    main()
