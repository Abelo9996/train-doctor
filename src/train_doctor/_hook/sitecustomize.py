"""train-doctor injection point.

train-doctor prepends this directory to PYTHONPATH for the command it runs,
so Python imports this file at startup. It installs the recorder from
``td_hook.py`` when ``TRAIN_DOCTOR_RUN_DIR`` is set, then hands over to any
other ``sitecustomize`` module that this one is shadowing.
"""

import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _install():
    if not os.environ.get("TRAIN_DOCTOR_RUN_DIR"):
        return
    try:
        import td_hook

        td_hook.install()
    except Exception as e:  # never break the user's program
        sys.stderr.write(f"[train-doctor] hook not installed: {e!r}\n")


def _chain():
    """Import the next sitecustomize on sys.path, if there is one."""
    ours = sys.modules.get("sitecustomize")
    saved = list(sys.path)
    try:
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]
        sys.modules.pop("sitecustomize", None)
        try:
            importlib.import_module("sitecustomize")
        except ImportError:
            pass
        except Exception as e:
            sys.stderr.write(f"[train-doctor] chained sitecustomize failed: {e!r}\n")
    finally:
        sys.path[:] = saved
        if ours is not None:
            sys.modules["sitecustomize"] = ours


_install()
_chain()
