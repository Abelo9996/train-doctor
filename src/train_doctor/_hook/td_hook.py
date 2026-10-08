"""In-process recorder that train-doctor injects into the training command.

This file must stay importable with the standard library only. It is loaded
by the bundled ``sitecustomize.py`` in whatever Python runs the user's
training script, which is often a different environment from the one
train-doctor itself is installed in.

What it records, per optimizer step (or per explicit ``step()`` marker):

* wall time between step boundaries
* time spent waiting in ``DataLoader.__next__``
* samples seen (first dimension of the batch)
* calls and time spent in host/device sync points (``.item()``, ``.cpu()``,
  ``.tolist()``, ``float(t)``, ``int(t)``, ``bool(t)`` on accelerator tensors,
  explicit ``synchronize()``)
* time in ``torch.save`` and in writes to stdout/stderr
* backward calls (to detect gradient accumulation)
* the loss value passed to ``backward()`` (kept as a tensor and converted at
  the end, so recording it does not add a sync to every step)

It also records configuration facts: DataLoader settings, autocast and
GradScaler use, torch.compile use, DDP use, device, dtype and versions.

Nothing here talks to the network. Output goes to ``$TRAIN_DOCTOR_RUN_DIR``.
"""

from __future__ import annotations

import atexit
import importlib
import importlib.abc
import importlib.util
import json
import os
import platform
import random
import signal
import sys
import threading
import time

ENV_RUN_DIR = "TRAIN_DOCTOR_RUN_DIR"

_installed = False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _redact(s: str) -> str:
    home = os.path.expanduser("~")
    return s.replace(home, "~", 1) if home and home != "/" and isinstance(s, str) and s.startswith(home) else s


def _batch_len(batch, depth: int = 0) -> int | None:
    """Best-effort number of samples in a batch: first dim of the first tensor."""
    if depth > 3 or batch is None:
        return None
    shape = getattr(batch, "shape", None)
    if shape is not None:
        try:
            return int(shape[0]) if len(shape) > 0 else None
        except Exception:
            return None
    if isinstance(batch, dict):
        for v in batch.values():
            n = _batch_len(v, depth + 1)
            if n is not None:
                return n
        return None
    if isinstance(batch, (list, tuple)):
        for v in batch:
            n = _batch_len(v, depth + 1)
            if n is not None:
                return n
    return None


class Recorder:
    def __init__(self) -> None:
        self.run_dir = os.environ.get(ENV_RUN_DIR, "")
        self.warmup = max(1, _env_int("TRAIN_DOCTOR_WARMUP", 5))  # step 0 has no defined start
        self.steps = max(1, _env_int("TRAIN_DOCTOR_STEPS", 50))
        self.seconds = _env_float("TRAIN_DOCTOR_SECONDS", 0.0)
        self.profiler_steps = _env_int("TRAIN_DOCTOR_PROFILER_STEPS", 0)
        self.stop_at_end = os.environ.get("TRAIN_DOCTOR_STOP", "1") == "1"
        self.auto = os.environ.get("TRAIN_DOCTOR_AUTO", "1") == "1"
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.file = None
        self.explicit = False
        self.internal = 0  # >0 while train-doctor itself calls patched functions
        self.n = 0  # completed steps
        self.t_last = None  # perf_counter at last boundary
        self.t_first_activity = None
        self.window_start = None  # (perf, wall)
        self.window_end = None
        self.done = False
        self.device = None
        self.meta: dict = {
            "pid": self.pid,
            "python": platform.python_version(),
            "executable": _redact(sys.executable),
            "argv": [_redact(a) for a in sys.argv],
            "dataloaders": [],
            "autocast": [],
            "grad_scaler": False,
            "compile_calls": [],
            "ddp": False,
            "ddp_no_sync_calls": 0,
            "conv_modules": 0,
            "torch": None,
        }
        self.losses: list = []  # (step, value-or-tensor, scale-or-None)
        self.scaled: dict = {}  # id(scaled tensor) -> scale tensor
        self.profiler = None
        self.profile_until = 0
        self._reset_counters()

    # ------------------------------------------------------------------ io
    def _open(self) -> None:
        if self.file is None and self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            path = os.path.join(self.run_dir, f"events.{self.pid}.jsonl")
            self.file = open(path, "a", buffering=1 << 16)  # noqa: SIM115

    def emit(self, ev: dict) -> None:
        self._open()
        if self.file is not None:
            self.internal += 1
            try:
                self.file.write(json.dumps(ev, default=str) + "\n")
            finally:
                self.internal -= 1

    def flush(self) -> None:
        if self.file is not None:
            self.internal += 1
            try:
                self.file.flush()
            finally:
                self.internal -= 1

    # ------------------------------------------------------------ counters
    def _reset_counters(self) -> None:
        self.c_data_wait = 0.0
        self.c_samples = 0
        self.c_samples_known = False
        self.c_sync_calls = 0
        self.c_sync_time = 0.0
        self.c_cpu_scalar_calls = 0
        self.c_save_time = 0.0
        self.c_save_calls = 0
        self.c_io_time = 0.0
        self.c_io_lines = 0
        self.c_backward = 0
        self.c_getitem_time = 0.0
        self.c_getitem_calls = 0
        self.c_micro_losses: list = []

    def activity(self) -> None:
        if self.t_first_activity is None:
            self.t_first_activity = time.perf_counter()

    # ------------------------------------------------------------- device
    def sync_device(self) -> None:
        torch = sys.modules.get("torch")
        if torch is None:
            return
        self.internal += 1
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif self.device == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
        except Exception:
            pass
        finally:
            self.internal -= 1

    def memory_snapshot(self) -> dict:
        torch = sys.modules.get("torch")
        out: dict = {}
        if torch is None:
            return out
        self.internal += 1
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                out["allocated"] = int(torch.cuda.memory_allocated())
                out["peak_allocated"] = int(torch.cuda.max_memory_allocated())
                out["reserved"] = int(torch.cuda.memory_reserved())
                out["total"] = int(torch.cuda.get_device_properties(0).total_memory)
            elif self.device == "mps":
                out["allocated"] = int(torch.mps.current_allocated_memory())
                out["driver_allocated"] = int(torch.mps.driver_allocated_memory())
                rec = getattr(torch.mps, "recommended_max_memory", None)
                if rec is not None:
                    out["total"] = int(rec())
        except Exception as e:  # pragma: no cover - depends on hardware
            out["error"] = repr(e)
        finally:
            self.internal -= 1
        return out

    # ------------------------------------------------------------ boundary
    def boundary(self, samples=None, loss=None, source: str = "optimizer", device=None) -> None:
        if self.done:
            return
        with self.lock:
            now = time.perf_counter()
            if device and not self.device:
                self.device = device
            if loss is not None:
                self._capture_loss(loss)
            if self.n == 0:
                self._first_boundary()
            start = self.t_last if self.t_last is not None else (self.t_first_activity or now)
            dt = now - start
            if samples is not None:
                self.c_samples = int(samples)
                self.c_samples_known = True
            profiled = self.profiler is not None
            ev = {
                "type": "step",
                "i": self.n,
                "source": source,
                "t": now,
                "dt": dt,
                "data_wait": self.c_data_wait,
                "samples": self.c_samples if self.c_samples_known else None,
                "sync_calls": self.c_sync_calls,
                "sync_time": self.c_sync_time,
                "cpu_scalar_calls": self.c_cpu_scalar_calls,
                "save_calls": self.c_save_calls,
                "save_time": self.c_save_time,
                "io_lines": self.c_io_lines,
                "io_time": self.c_io_time,
                "backward_calls": self.c_backward,
                "getitem_calls": self.c_getitem_calls,
                "getitem_time": self.c_getitem_time,
                "warmup": self.n < self.warmup,
                "profiled": profiled,
            }
            if self.c_micro_losses:
                self.losses.append((self.n, list(self.c_micro_losses)))
            if self.n % 10 == 0:
                ev["mem"] = self.memory_snapshot()
            self.emit(ev)
            self.n += 1
            self._reset_counters()
            if self.profiler is not None:
                self.internal += 1
                try:
                    self.profiler.step()
                finally:
                    self.internal -= 1
            started = self._window_logic()
            if not started:
                # train-doctor's own bookkeeping is excluded from the next step's dt
                self.t_last = time.perf_counter()

    def _first_boundary(self) -> None:
        self.meta["first_step_wall"] = time.time()
        if self.t_first_activity is not None:
            self.meta["startup_to_first_step_s"] = time.perf_counter() - self.t_first_activity
        self.meta["device"] = self.device
        self._collect_torch_meta()
        self.emit({"type": "meta", **self.meta})
        self._install_sigterm()

    def _window_logic(self) -> bool:
        """Advance the measurement window. Returns True when the window starts at this boundary."""
        if self.window_start is None and self.n >= self.warmup:
            self.sync_device()
            self.window_start = (time.perf_counter(), time.time())
            self.t_last = self.window_start[0]
            self.emit({"type": "window_start", "t": self.window_start[0], "wall": self.window_start[1], "after_step": self.n - 1})
            self.flush()
            return True
        if self.window_start is not None and self.window_end is None:
            measured = self.n - self.warmup
            elapsed = time.perf_counter() - self.window_start[0]
            reached = measured >= self.steps and elapsed >= self.seconds  # at least N steps and at least T seconds
            if reached:
                self.sync_device()
                self.window_end = (time.perf_counter(), time.time())
                self.emit(
                    {
                        "type": "window_end",
                        "t": self.window_end[0],
                        "wall": self.window_end[1],
                        "after_step": self.n - 1,
                        "mem": self.memory_snapshot(),
                    }
                )
                self.flush()
                if self.profiler_steps > 0 and self._start_profiler():
                    self.profile_until = self.n + self.profiler_steps
                    return False
                self.finish(stop=True)
                return False
        if self.profiler is not None and self.n >= self.profile_until:
            self._stop_profiler()
            self.finish(stop=True)
        return False

    # ------------------------------------------------------------ profiler
    def _start_profiler(self) -> bool:
        torch = sys.modules.get("torch")
        if torch is None:
            return False
        self.internal += 1
        try:
            from torch.profiler import ProfilerActivity, profile

            acts = [ProfilerActivity.CPU]
            if self.device == "cuda" and torch.cuda.is_available():
                acts.append(ProfilerActivity.CUDA)
            self.profiler = profile(activities=acts, record_shapes=False, with_stack=False)
            self.profiler.start()
            self.emit({"type": "profiler_start", "after_step": self.n - 1})
            return True
        except Exception as e:
            self.emit({"type": "profiler_error", "error": repr(e)})
            self.profiler = None
            return False
        finally:
            self.internal -= 1

    def _stop_profiler(self) -> None:
        prof = self.profiler
        self.profiler = None
        if prof is None:
            return
        self.internal += 1
        try:
            prof.stop()
            steps = max(1, self.profiler_steps)
            rows = []
            for evt in prof.key_averages():
                dev_self = getattr(evt, "self_device_time_total", None)
                if dev_self is None:
                    dev_self = getattr(evt, "self_cuda_time_total", 0)
                rows.append(
                    {
                        "name": evt.key,
                        "count": int(evt.count),
                        "self_cpu_us": float(evt.self_cpu_time_total),
                        "cpu_total_us": float(evt.cpu_time_total),
                        "self_device_us": float(dev_self or 0.0),
                    }
                )
            rows.sort(key=lambda r: r["self_cpu_us"] + r["self_device_us"], reverse=True)
            with open(os.path.join(self.run_dir, "torch_ops.json"), "w") as f:
                json.dump({"profiled_steps": steps, "device": self.device, "ops": rows[:200]}, f, indent=1)
            try:
                trace = os.path.join(self.run_dir, "torch_trace.json")
                prof.export_chrome_trace(trace)
                import gzip
                import shutil

                with open(trace, "rb") as src, gzip.open(trace + ".gz", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.remove(trace)
            except Exception as e:
                self.emit({"type": "profiler_error", "error": "trace export: " + repr(e)})
            self.emit({"type": "profiler_end", "after_step": self.n - 1})
        except Exception as e:
            self.emit({"type": "profiler_error", "error": repr(e)})
        finally:
            self.internal -= 1

    # ---------------------------------------------------------------- loss
    def _capture_loss(self, loss) -> None:
        try:
            if hasattr(loss, "detach"):
                if loss.numel() != 1:
                    return
                scale = self.scaled.pop(id(loss), None)
                self.c_micro_losses.append((loss.detach(), scale))
            else:
                self.c_micro_losses.append((float(loss), None))
        except Exception:
            pass

    def _loss_values(self) -> list:
        out = []
        self.internal += 1
        try:
            for step, micro in self.losses:
                vals = []
                for v, scale in micro:
                    try:
                        x = float(v)
                        if scale is not None:
                            s = float(scale)
                            if s:
                                x = x / s
                        vals.append(x)
                    except Exception:
                        continue
                if vals:
                    out.append([step, sum(vals) / len(vals), len(vals)])
        finally:
            self.internal -= 1
        return out

    # -------------------------------------------------------------- finish
    def finish(self, stop: bool = False, reason: str = "window_complete") -> None:
        if self.done:
            return
        if self.profiler is not None:
            self._stop_profiler()
        self.done = True
        if self.n == 0 and self.file is None:
            return
        self.meta["device"] = self.meta.get("device") or self.device
        self.emit({"type": "meta", **self.meta})  # final copy: config seen after step 0 is included
        self.emit({"type": "losses", "values": self._loss_values()})
        self.emit({"type": "end", "reason": reason, "steps": self.n, "wall": time.time()})
        self.flush()
        if stop and self.stop_at_end:
            sys.stderr.write(f"[train-doctor] measurement window complete after {self.n} steps, stopping the run\n")
            raise SystemExit(0)

    def _install_sigterm(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, None):

                def _on_term(signum, frame):
                    try:
                        self.finish(stop=False, reason="sigterm")
                    finally:
                        os._exit(143)

                signal.signal(signal.SIGTERM, _on_term)
        except Exception:
            pass

    # ---------------------------------------------------------------- meta
    def _collect_torch_meta(self) -> None:
        torch = sys.modules.get("torch")
        if torch is None:
            return
        self.internal += 1
        try:
            m: dict = {"version": getattr(torch, "__version__", "?")}
            m["num_threads"] = torch.get_num_threads()
            m["cuda_available"] = bool(torch.cuda.is_available())
            mps = getattr(torch.backends, "mps", None)
            m["mps_available"] = bool(mps and mps.is_available())
            if m["cuda_available"]:
                idx = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(idx)
                m["cuda_device"] = props.name
                m["cuda_capability"] = [props.major, props.minor]
                m["cuda_total_memory"] = int(props.total_memory)
                m["cuda_version"] = torch.version.cuda
                m["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
                m["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
            m["compile_available"] = hasattr(torch, "compile")
            m["mps_autocast_supported"] = _mps_autocast_supported(torch)
            self.meta["torch"] = m
            self.meta["device"] = self.device
        except Exception as e:
            self.meta["torch_meta_error"] = repr(e)
        finally:
            self.internal -= 1


def _mps_autocast_supported(torch) -> bool:
    try:
        with torch.autocast(device_type="mps", dtype=torch.float16, enabled=False):
            pass
        return True
    except Exception:
        return False


R: Recorder | None = None


def recorder() -> Recorder | None:
    return R


# ---------------------------------------------------------------- public API
def step(samples=None, loss=None, device=None) -> None:
    """Explicit step marker. Once called, automatic optimizer-step markers are ignored."""
    r = R
    if r is None:
        return
    r.explicit = True
    r.activity()
    if loss is not None:
        r.c_micro_losses = []  # an explicit loss replaces anything captured from backward()
    if device is None and loss is not None and hasattr(loss, "device"):
        device = loss.device.type
    r.boundary(samples=samples, loss=loss, source="marker", device=device)


# -------------------------------------------------------------- torch patches
def _is_compiling() -> bool:
    torch = sys.modules.get("torch")
    try:
        return bool(torch is not None and torch.compiler.is_compiling())
    except Exception:
        return False


def _timed_sync(orig, name):
    def wrapper(self, *a, **k):
        r = R
        if r is None or r.internal or _is_compiling():
            return orig(self, *a, **k)
        try:
            dev = self.device.type
        except Exception:
            return orig(self, *a, **k)
        if dev == "cpu" or dev == "meta":
            if name in ("item", "__float__", "__int__", "__bool__"):
                r.c_cpu_scalar_calls += 1
            return orig(self, *a, **k)
        t0 = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            r.c_sync_calls += 1
            r.c_sync_time += time.perf_counter() - t0

    wrapper.__name__ = name
    wrapper.__wrapped__ = orig
    return wrapper


def _patch_torch(torch) -> None:
    r = R
    if r is None:
        return
    seed = os.environ.get("TRAIN_DOCTOR_SEED")
    if seed not in (None, ""):
        try:
            s = int(seed)
            torch.manual_seed(s)
            random.seed(s)
            np = sys.modules.get("numpy")
            if np is not None:
                np.random.seed(s)
        except Exception:
            pass

    # optimizer step boundary
    try:
        from torch.optim.optimizer import register_optimizer_step_post_hook

        def _post(optimizer, args, kwargs):
            rr = R
            if rr is None or rr.explicit or not rr.auto:
                return
            dev = None
            try:
                p0 = optimizer.param_groups[0]["params"][0]
                dev = p0.device.type
                if "params" not in rr.meta:
                    total = 0
                    dtypes = set()
                    for g in optimizer.param_groups:
                        for p in g["params"]:
                            total += p.numel()
                            dtypes.add(str(p.dtype).replace("torch.", ""))
                    rr.meta["params"] = total
                    rr.meta["param_dtypes"] = sorted(dtypes)
                    rr.meta["optimizer"] = type(optimizer).__name__
            except Exception:
                pass
            rr.boundary(source="optimizer", device=dev)

        register_optimizer_step_post_hook(_post)
    except Exception as e:  # pragma: no cover - very old torch
        r.meta["optimizer_hook_error"] = repr(e)

    # dataloader config + wait time
    try:
        from torch.utils.data import dataloader as dl

        orig_iter = dl.DataLoader.__iter__

        def __iter__(self):
            rr = R
            if rr is not None and not getattr(self, "_td_seen", False):
                try:
                    self._td_seen = True
                    ds = self.dataset
                    info = {
                        "dataset": type(ds).__name__,
                        "num_workers": self.num_workers,
                        "pin_memory": bool(self.pin_memory),
                        "batch_size": self.batch_size,
                        "persistent_workers": bool(getattr(self, "persistent_workers", False)),
                        "prefetch_factor": getattr(self, "prefetch_factor", None),
                        "drop_last": bool(getattr(self, "drop_last", False)),
                    }
                    try:
                        info["len"] = len(ds)
                    except Exception:
                        info["len"] = None
                    rr.meta["dataloaders"].append(info)
                    if self.num_workers == 0:
                        _wrap_getitem(type(ds))
                except Exception:
                    pass
            if rr is not None:
                rr.activity()
            return orig_iter(self)

        dl.DataLoader.__iter__ = __iter__

        orig_next = dl._BaseDataLoaderIter.__next__

        def __next__(self):
            rr = R
            if rr is None:
                return orig_next(self)
            rr.activity()
            t0 = time.perf_counter()
            try:
                out = orig_next(self)
            finally:
                rr.c_data_wait += time.perf_counter() - t0
            n = _batch_len(out)
            if n is not None:
                rr.c_samples += n
                rr.c_samples_known = True
            return out

        dl._BaseDataLoaderIter.__next__ = __next__
    except Exception as e:  # pragma: no cover
        r.meta["dataloader_patch_error"] = repr(e)

    # sync points
    T = torch.Tensor
    for name in ("item", "tolist", "cpu", "__float__", "__int__", "__bool__"):
        orig = getattr(T, name, None)
        if orig is not None and not hasattr(orig, "__wrapped__"):
            try:
                setattr(T, name, _timed_sync(orig, name))
            except Exception:
                pass
    for modname in ("cuda", "mps"):
        mod = getattr(torch, modname, None)
        if mod is not None and hasattr(mod, "synchronize"):
            _wrap_module_sync(mod)

    # backward: loss capture + accumulation count
    orig_backward = T.backward

    def backward(self, *a, **k):
        rr = R
        if rr is not None and not rr.internal:
            rr.c_backward += 1
            rr.activity()
            if not rr.explicit:
                rr._capture_loss(self)
        return orig_backward(self, *a, **k)

    T.backward = backward

    # torch.save timing
    orig_save = torch.save

    def save(*a, **k):
        rr = R
        t0 = time.perf_counter()
        try:
            return orig_save(*a, **k)
        finally:
            if rr is not None and not rr.internal:
                rr.c_save_calls += 1
                rr.c_save_time += time.perf_counter() - t0

    torch.save = save

    # autocast use
    try:
        from torch.amp import autocast_mode

        ac = autocast_mode.autocast
        orig_enter = ac.__enter__

        def __enter__(self):
            rr = R
            if rr is not None:
                try:
                    key = [
                        str(getattr(self, "device", "?")),
                        str(getattr(self, "fast_dtype", "?")).replace("torch.", ""),
                        bool(getattr(self, "_enabled", True)),
                    ]
                    if key not in rr.meta["autocast"]:
                        rr.meta["autocast"].append(key)
                except Exception:
                    pass
            return orig_enter(self)

        ac.__enter__ = __enter__
    except Exception:
        pass

    # GradScaler: remember scale so the recorded loss can be unscaled
    for path in ("amp", "cuda.amp"):
        try:
            mod = torch
            for part in path.split("."):
                mod = getattr(mod, part)
            cls = getattr(mod, "GradScaler", None)
            if cls is None or getattr(cls.scale, "_td", False):
                continue
            orig_scale = cls.scale

            def scale(self, outputs, _orig=orig_scale):
                out = _orig(self, outputs)
                rr = R
                if rr is not None:
                    rr.meta["grad_scaler"] = True
                    s = getattr(self, "_scale", None)
                    if s is not None and hasattr(out, "detach"):
                        rr.scaled[id(out)] = s.detach().clone() if hasattr(s, "clone") else s
                return out

            scale._td = True
            cls.scale = scale
        except Exception:
            pass

    # torch.compile use
    orig_compile = getattr(torch, "compile", None)
    if orig_compile is not None:

        def compile(*a, **k):
            rr = R
            if rr is not None:
                rr.meta["compile_calls"].append({"mode": k.get("mode"), "backend": str(k.get("backend", "inductor"))})
            return orig_compile(*a, **k)

        torch.compile = compile

    # DDP and conv modules
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP

        orig_ddp_init = DDP.__init__

        def ddp_init(self, *a, **k):
            if R is not None:
                R.meta["ddp"] = True
            return orig_ddp_init(self, *a, **k)

        DDP.__init__ = ddp_init
        orig_no_sync = DDP.no_sync

        def no_sync(self):
            if R is not None:
                R.meta["ddp_no_sync_calls"] += 1
            return orig_no_sync(self)

        DDP.no_sync = no_sync
    except Exception:
        pass
    try:
        from torch.nn.modules import conv

        orig_conv_init = conv._ConvNd.__init__

        def conv_init(self, *a, **k):
            if R is not None:
                R.meta["conv_modules"] += 1
            return orig_conv_init(self, *a, **k)

        conv._ConvNd.__init__ = conv_init
    except Exception:
        pass


def _wrap_module_sync(mod) -> None:
    orig = mod.synchronize
    if hasattr(orig, "__wrapped__"):
        return

    def synchronize(*a, **k):
        r = R
        if r is None or r.internal:
            return orig(*a, **k)
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            r.c_sync_calls += 1
            r.c_sync_time += time.perf_counter() - t0

    synchronize.__wrapped__ = orig
    mod.synchronize = synchronize


def _wrap_getitem(cls) -> None:
    for attr in ("__getitem__", "__getitems__"):
        orig = cls.__dict__.get(attr)
        if orig is None or getattr(orig, "_td", False):
            continue

        def make(orig):
            def wrapped(self, *a, **k):
                r = R
                t0 = time.perf_counter()
                try:
                    return orig(self, *a, **k)
                finally:
                    if r is not None and r.pid == os.getpid():
                        r.c_getitem_calls += 1
                        r.c_getitem_time += time.perf_counter() - t0

            wrapped._td = True
            wrapped.__wrapped__ = orig
            return wrapped

        try:
            setattr(cls, attr, make(orig))
        except Exception:
            pass


class _TimedStream:
    """Wraps stdout/stderr to count lines and time spent writing them."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def write(self, s):
        r = R
        if r is None or r.internal:
            return self._inner.write(s)
        t0 = time.perf_counter()
        try:
            return self._inner.write(s)
        finally:
            r.c_io_time += time.perf_counter() - t0
            try:
                r.c_io_lines += s.count("\n")
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ------------------------------------------------------ post-import machinery
class _TorchFinder(importlib.abc.MetaPathFinder):
    """Runs _patch_torch right after ``import torch`` finishes."""

    def __init__(self) -> None:
        self._busy = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "torch" or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec("torch")
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                _patch_torch(module)
            except Exception as e:
                sys.stderr.write(f"[train-doctor] could not instrument torch: {e!r}\n")

        loader.exec_module = exec_module
        return spec


def install() -> None:
    """Install the recorder. Called from sitecustomize when TRAIN_DOCTOR_RUN_DIR is set."""
    global R, _installed
    if _installed or not os.environ.get(ENV_RUN_DIR):
        return
    _installed = True
    R = Recorder()
    if "torch" in sys.modules:
        _patch_torch(sys.modules["torch"])
    else:
        sys.meta_path.insert(0, _TorchFinder())
    if os.environ.get("TRAIN_DOCTOR_IO", "1") == "1":
        sys.stdout = _TimedStream(sys.stdout)
        sys.stderr = _TimedStream(sys.stderr)

    def _at_exit():
        r = R
        if r is not None and not r.done and r.pid == os.getpid():
            try:
                r.finish(stop=False, reason="process_exit")
            except SystemExit:
                pass

    atexit.register(_at_exit)
