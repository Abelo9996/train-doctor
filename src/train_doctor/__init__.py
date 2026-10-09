"""train-doctor: find out why a training run is slow, fix it, and prove the speedup.

The only thing a training script ever needs from this package is the optional
step marker::

    import train_doctor
    ...
    optimizer.step()
    train_doctor.step(samples=len(batch), loss=loss)

It is a no-op unless the script runs under ``train-doctor profile`` or
``train-doctor compare``. Without markers, train-doctor detects steps from
``optimizer.step()`` automatically.
"""

from __future__ import annotations

import sys

__version__ = "0.1.2"

__all__ = ["__version__", "step"]


def step(samples: int | None = None, loss=None, device: str | None = None) -> None:
    """Mark the end of one training step.

    Args:
        samples: number of samples processed in this step (for samples/s).
        loss: the step's loss (a tensor or a float). Tensors are converted at
            the end of the run, so passing one does not add a device sync.
        device: "cuda", "mps" or "cpu" if it can't be inferred from ``loss``.
    """
    hook = sys.modules.get("td_hook")
    if hook is not None:
        hook.step(samples=samples, loss=loss, device=device)
