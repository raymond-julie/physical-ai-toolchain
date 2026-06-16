#!/usr/bin/env python3
"""Model runner for the UR edge runtime.

Loads a LeRobot SmolVLA checkpoint from
``ai_models/<model_name>/<version>/<model_name>/pretrained_model`` and runs a
periodic inference loop. Each step builds an observation from the latest camera
frames and destination joint state, feeds it to the policy, and publishes the
resulting joint-target + gripper-position via callbacks the GUI node provides
(so the existing destination_writer state machine moves the robot).

LeRobot + torch are imported lazily so the GUI process still starts on machines
that don't have them installed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from gr00t_runner import FLAVOR_TCP_EE, Gr00tBackend, is_gr00t_checkpoint

_LOGGER = logging.getLogger(__name__)

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.CRITICAL,
}


# ── Defaults ─────────────────────────────────────────────────────────────────

# Frequency at which select_action() is called. SmolVLA returns chunks of
# actions, so this only sets how often we dequeue / publish a new target.
DEFAULT_HZ = 15.0
DEFAULT_TASK = "pick up the object"

# Policy backend types.
POLICY_SMOLVLA = "smolvla"
POLICY_GR00T = "gr00t"
POLICY_ACT = "act"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _peek_policy_type(ckpt_dir: Path) -> str | None:
    """Read ``config.json`` inside a SmolVLA/ACT-style checkpoint and return its
    ``type`` field (e.g. ``"smolvla"``, ``"act"``). Returns ``None`` if the file
    is missing or unreadable.
    """
    cfg = ckpt_dir / "config.json"
    if not cfg.exists():
        return None
    try:
        with open(cfg) as f:
            data = json.load(f)
    except Exception:
        return None
    t = data.get("type")
    if isinstance(t, str):
        return t.lower()
    return None


def _resolve_checkpoint_dir(version_dir: Path) -> Path | None:
    """Find the ``pretrained_model`` directory inside a version folder.

    Layout is ``<base>/<model_name>/<version>/<model_name>/pretrained_model`` or,
    more loosely, the first ``pretrained_model`` directory found below
    ``version_dir``.
    """
    direct = list(version_dir.glob("*/pretrained_model"))
    if direct:
        return direct[0]
    deep = list(version_dir.rglob("pretrained_model"))
    return deep[0] if deep else None


def list_versions(base_dir: Path) -> list[dict]:
    """Enumerate every runnable checkpoint under ``base_dir``.

    Expected layout::

        <base_dir>/<family>/<version_or_checkpoint>/...

    For each ``<family>/<version>`` we report:
      * ``version`` — ``"<family>/<version>"`` (used as the start() id)
      * ``family`` / ``subversion``
      * ``policy_type`` — ``"smolvla"``, ``"act"`` or ``"gr00t"``
      * ``has_checkpoint`` / ``checkpoint_path``

    SmolVLA and ACT versions are detected by the presence of a nested
    ``pretrained_model/`` directory; the actual backend is then chosen by reading
    ``config.json["type"]``. GR00T checkpoints are detected by a top-level
    ``config.json`` whose ``architectures`` list mentions a ``GR00T_*`` class.
    Anything else is skipped.
    """
    out: list[dict] = []
    if not base_dir.exists():
        return out
    for family in sorted(base_dir.iterdir(), key=lambda p: p.name.lower()):
        if not family.is_dir():
            continue
        for child in sorted(family.iterdir(),
                            key=lambda p: _version_sort_key(p.name)):
            if not child.is_dir():
                continue
            policy_type = None
            ckpt = None
            if is_gr00t_checkpoint(child):
                policy_type = POLICY_GR00T
                ckpt = child
            else:
                smol_ckpt = _resolve_checkpoint_dir(child)
                if smol_ckpt is not None:
                    ckpt = smol_ckpt
                    cfg_type = _peek_policy_type(smol_ckpt)
                    if cfg_type == POLICY_ACT:
                        policy_type = POLICY_ACT
                    else:
                        # Default to SmolVLA for backward compatibility when
                        # config.json is missing or has another type.
                        policy_type = POLICY_SMOLVLA
            if policy_type is None:
                # Not a recognised checkpoint — skip silently.
                continue
            out.append({
                "version": f"{family.name}/{child.name}",
                "family": family.name,
                "subversion": child.name,
                "path": str(child),
                "policy_type": policy_type,
                "has_checkpoint": ckpt is not None,
                "checkpoint_path": str(ckpt) if ckpt else None,
            })
    return out


def _version_sort_key(name: str) -> tuple[int, int | str]:
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


# ── Model runner ────────────────────────────────────────────────────────────

class ModelRunner:
    """Owns the worker thread + policy. Thread-safe public API.

    Lifecycle (a separate axis from execution)::

        STOPPED -> LOADING -> LOADED <-> RUNNING

    ``LOADED`` means the policy is resident in GPU memory but the inference loop
    is not driving the robot. ``RUNNING`` adds the loop. ``stop()`` returns to
    ``LOADED`` (model stays warm); ``unload()`` returns to ``STOPPED``.
    ``load()`` while already loaded swaps the checkpoint atomically.
    """

    STATE_STOPPED = "stopped"
    STATE_LOADING = "loading"
    STATE_LOADED = "loaded"
    STATE_RUNNING = "running"
    STATE_ERROR = "error"

    def __init__(self,
                 base_dir: Path,
                 publish_action: Callable[[list[float], float], None],
                 logger: Any = None,
                 hz: float = DEFAULT_HZ,
                 publish_tcp: Callable[[list[float], float], None] | None = None) -> None:
        """:param base_dir: ``ai_models/<model_name>`` directory whose children
            are versions (``6``, ``7``, ...).
        :param publish_action: callback ``(joint_targets[6], gripper_pos[0..1])``
            invoked from the worker thread on every successful inference step.
        :param logger: ROS logger or ``None``.
        :param hz: inference frequency.
        :param publish_tcp: optional callback ``(tcp_pose[6], gripper_pos[0..1])``
            used when the loaded policy emits Cartesian targets (e.g. the GR00T
            TCP-EE checkpoint). When None, TCP-flavour models cannot drive the
            robot and ``_step_gr00t`` logs a warning instead.
        """
        self.base_dir = Path(base_dir)
        self.publish_action = publish_action
        self.publish_tcp = publish_tcp
        self.logger = logger
        self.hz = hz

        self._lock = threading.Lock()
        self._state = self.STATE_STOPPED
        self._message = ""
        self._version: str | None = None
        self._task = DEFAULT_TASK
        # GR00T chunk-consumption cap (see Gr00tBackend.max_steps_per_chunk).
        # Honored on next load and live-applied to the backend if loaded.
        self._max_steps_per_chunk: int | None = None
        # GR00T temporal ensemble window K (see Gr00tBackend.ensemble_window).
        # 1 = no ensembling (default).
        self._ensemble_window: int = 1

        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

        self._policy = None
        self._device = "cpu"
        # Which backend the currently-loaded checkpoint uses. Set inside
        # _load_policy. One of POLICY_SMOLVLA / POLICY_GR00T.
        self._policy_type: str | None = None
        self._gr00t: Gr00tBackend | None = None

        # Observation source — set by the GUI node before start().
        self._obs_provider: Callable[[], dict | None] | None = None

        # Stats
        self._last_action: list[float] | None = None
        self._last_step_time: float = 0.0
        self._step_count: int = 0
        self._fps: float = 0.0
        # Live-tunable requested loop frequency. Initialised from the
        # constructor's ``hz`` argument; reassigned at the top of _run and
        # updated by set_inference_hz().
        self._requested_hz: float = float(self.hz)

        # ── RTC / async-inference state ──
        # Set inside _load_policy when RTC is available.
        self._rtc_enabled: bool = False
        self._rtc_cfg = None
        self._action_queue = None
        # Inference worker — only one inference may run at a time.
        self._infer_busy = threading.Event()
        self._infer_lock = threading.Lock()
        # Smoothed estimate of how many ticks pass during one inference (used as
        # ``inference_delay`` argument to RTC). Updated each cycle from measured
        # wall-clock latency * loop hz.
        self._inference_delay_ticks: int = 5
        self._infer_count: int = 0
        self._infer_total_s: float = 0.0
        # EMA of inference wall-clock latency (seconds). Used to clamp playback
        # hz so chunk_size / hz >> inference_time, preventing the action queue
        # from emptying between chunks.
        self._infer_latency_ema: float = 0.0
        # EMA of wall-clock interval between successive inferences. This is the
        # *actual* inference Hz the worker is achieving (1/period), which differs
        # from the requested hz whenever inference cannot keep up or RTC throttles
        # via _effective_hz.
        self._infer_period_ema: float = 0.0
        self._last_infer_t: float = 0.0
        # Last announced effective hz (only re-logged when it changes).
        self._announced_hz: float = 0.0
        # GR00T-only: live playback-rate scale driven by /gui/speed_scale. Joint
        # and SmolVLA flavours ignore this; they rely on the downstream
        # destination_writer velocity clamp instead.
        self._speed_scale: float = 1.0

    # ── Public API ───────────────────────────────────────────────────

    def set_observation_provider(self, fn: Callable[[], dict | None]) -> None:
        """Register a callable returning a dict with keys ``joints`` (list[6]),
        ``camera1`` (BGR uint8 ndarray HxWx3 or None), ``camera2`` (same) and
        optional ``camera3``. Returning ``None`` skips the inference step (no
        action is published).
        """
        self._obs_provider = fn

    def set_speed_scale(self, scale: float) -> None:
        """Live update of the playback-rate scale (0.01..1.0).

        GR00T TCP-EE rollouts are absolute Cartesian setpoints on a fixed time
        grid. Slowing them requires stretching the time between setpoints, not
        clamping per-tick velocity — which is what the downstream servoL clamp
        already does on its own and is the source of the "spotty / jerks faster
        than the setting" behaviour. Scaling the loop hz here gives true
        proportional slow-down. SmolVLA / ACT paths ignore this and continue to
        rely on destination_writer's velocity clamp.
        """
        try:
            s = float(scale)
        except (TypeError, ValueError):
            return
        s = max(0.01, min(1.0, s))
        with self._lock:
            changed = abs(s - self._speed_scale) > 1e-4
            self._speed_scale = s
        if changed:
            self._log("info", f"GR00T playback speed_scale -> {s * 100:.0f}%")

    def list_versions(self) -> list[dict]:
        return list_versions(self.base_dir)

    def status(self) -> dict:
        with self._lock:
            horizon = self._gr00t.chunk_horizon if self._gr00t is not None else None
            return {
                "state": self._state,
                "message": self._message,
                "version": self._version,
                "task": self._task,
                "fps": round(self._fps, 1),
                "steps": self._step_count,
                "last_action": self._last_action,
                "available": [v["version"] for v in list_versions(self.base_dir)],
                "max_steps_per_chunk": self._max_steps_per_chunk,
                "chunk_horizon": horizon,
                "ensemble_window": self._ensemble_window,
                "inference_hz": float(self._requested_hz),
                "inference_latency_ms": round(self._infer_latency_ema * 1000.0, 1),
                "inference_actual_hz": (
                    round(1.0 / self._infer_period_ema, 2)
                    if self._infer_period_ema > 0.0 else 0.0),
                "effective_hz": round(self._announced_hz, 2),
            }

    def start(self, version: str, task: str | None = None) -> dict:
        """Begin inference using the already-loaded model.

        ``version`` is accepted for backward compatibility; if it differs from
        the currently-loaded version (or no model is loaded), this triggers a
        load+start in sequence. Otherwise it just resumes execution against the
        warm policy.
        """
        with self._lock:
            current = self._version
            state = self._state
            if task:
                self._task = task
                if self._gr00t is not None:
                    self._gr00t.task = task
        if state == self.STATE_RUNNING:
            return {"ok": False, "msg": "Model already running"}
        if state == self.STATE_LOADING:
            return {"ok": False, "msg": "Model still loading"}
        # If a different version is requested (or nothing is loaded), fall back
        # to the legacy load+start path so external callers that only know about
        # /api/model/start keep working.
        if state != self.STATE_LOADED or (version and version != current):
            return self.load(version, task=task, autostart=True)
        return self._begin_inference_loop()

    def load(self, version: str, task: str | None = None,
             autostart: bool = False) -> dict:
        """Load (or swap to) a checkpoint, keeping it warm in GPU memory.

        Returns immediately; the heavy work runs in a background thread and the
        runner transitions ``LOADING`` -> ``LOADED`` (or ``ERROR``). If
        ``autostart`` is True, the inference loop is engaged as soon as the load
        completes.
        """
        with self._lock:
            if self._state == self.STATE_LOADING:
                return {"ok": False, "msg": "Already loading"}
            if self._state == self.STATE_RUNNING:
                return {"ok": False,
                        "msg": "Stop the model before loading a different one"}
            self._version = str(version)
            if task:
                self._task = task
            self._state = self.STATE_LOADING
            self._message = f"Loading version {version}..."

        # Run the load in a background thread so the HTTP request returns
        # immediately and the UI can render "Loading..." while Cosmos comes up
        # (~60 s cold).
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._load_and_optionally_run, args=(autostart,),
            name=f"ModelRunner-load-{version}", daemon=True)
        self._thread.start()
        return {"ok": True, "msg": f"Loading model version {version}"}

    def stop(self) -> dict:
        """Halt the inference loop. Model stays loaded and warm."""
        with self._lock:
            if self._state not in (self.STATE_RUNNING,):
                # Nothing to stop. Idempotent.
                return {"ok": True, "msg": f"Not running (state={self._state})"}
        self._stop_evt.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=5.0)
        # Reset chunk state so the next Start gets a fresh plan from current obs
        # (rather than continuing a stale chunk).
        if self._gr00t is not None:
            try:
                self._gr00t.reset()
            except Exception as exc:
                self._log("warn", f"Gr00tBackend.reset() failed: {exc}")
        with self._lock:
            self._state = self.STATE_LOADED
            self._message = "Loaded (idle)"
        return {"ok": True, "msg": "Model stopped (still loaded)"}

    def unload(self) -> dict:
        """Free the policy from GPU. Used on shutdown or when the user wants a
        clean tear-down. Loading a different model auto-unloads the current one
        as part of load().
        """
        # Stop first if running.
        with self._lock:
            running = self._state == self.STATE_RUNNING
        if running:
            self.stop()
        with self._lock:
            self._policy = None
            self._action_queue = None
            self._rtc_enabled = False
            self._gr00t = None
            self._policy_type = None
            self._state = self.STATE_STOPPED
            self._message = "Stopped (unloaded)"
        return {"ok": True, "msg": "Model unloaded"}

    def set_task(self, task: str) -> dict:
        """Live-update the language task description for GR00T inference.

        Takes effect on the next inference call; no reload required.
        """
        task = (task or "").strip() or DEFAULT_TASK
        with self._lock:
            self._task = task
            if self._gr00t is not None:
                self._gr00t.task = task
        return {"ok": True, "task": task}

    def set_max_steps_per_chunk(self, n: int | None) -> dict:
        """Cap chunk consumption to the first ``n`` actions before re-infer.

        ``None`` or non-positive disables the cap (plays full action_horizon).
        Live-applies to the GR00T backend if loaded; otherwise stored for next
        load.
        """
        if n is None or (isinstance(n, int) and n <= 0):
            value: int | None = None
        else:
            value = max(1, int(n))
        with self._lock:
            self._max_steps_per_chunk = value
            if self._gr00t is not None:
                self._gr00t.max_steps_per_chunk = value
        return {"ok": True, "max_steps_per_chunk": value}

    def set_ensemble_window(self, k: int | None) -> dict:
        """Set ACT-style temporal ensemble window (number of recent chunks to
        weighted-average per tick). 1 disables ensembling.
        """
        if k is None:
            value = 1
        else:
            try:
                value = max(1, int(k))
            except (TypeError, ValueError):
                value = 1
        with self._lock:
            self._ensemble_window = value
            if self._gr00t is not None:
                self._gr00t.ensemble_window = value
        return {"ok": True, "ensemble_window": value}

    def set_inference_hz(self, hz: float) -> dict:
        """Live update of the requested inference loop frequency. The worker's
        actual rate may be clamped lower by _effective_hz when RTC is enabled
        and inference latency would otherwise drain the action queue.
        """
        try:
            v = float(hz)
        except (TypeError, ValueError):
            return {"ok": False, "msg": "hz must be a number"}
        v = max(0.5, min(60.0, v))
        with self._lock:
            self.hz = v
            self._requested_hz = v
        self._log("info", f"inference hz -> {v:.2f}")
        return {"ok": True, "hz": v}

    # ── Worker thread ────────────────────────────────────────────────

    def _log(self, level: str, msg: str) -> None:
        if self.logger is None:
            _LOGGER.log(_LOG_LEVELS.get(level, logging.INFO), "%s", msg)
            return
        # rclpy's RcutilsLogger ties severity to the call site (file/line).
        # Calling .info() then .error() from the same helper raises "Logger
        # severity cannot be changed between calls". Use the explicit
        # .log(msg, severity) form which accepts severity as a runtime argument.
        try:
            from rclpy.logging import LoggingSeverity
            sev_map = {
                "debug": LoggingSeverity.DEBUG,
                "info": LoggingSeverity.INFO,
                "warn": LoggingSeverity.WARN,
                "warning": LoggingSeverity.WARN,
                "error": LoggingSeverity.ERROR,
                "fatal": LoggingSeverity.FATAL,
            }
            sev = sev_map.get(level, LoggingSeverity.INFO)
            self.logger.log(msg, sev)
        except Exception:
            # Best-effort fallback: prefix the level into the message.
            self.logger.info(f"[{level.upper()}] {msg}")

    def _load_and_optionally_run(self, autostart: bool) -> None:
        """Background thread for ``load()``.

        Unloads any currently-resident policy, loads the requested version,
        transitions to ``LOADED``, and (optionally) starts the inference loop on
        the same thread.
        """
        # Drop any previously-loaded policy first. We do this lazily inside the
        # worker thread so the HTTP request returns quickly and the UI sees the
        # LOADING state immediately.
        with self._lock:
            if self._gr00t is not None or self._policy is not None:
                self._policy = None
                self._action_queue = None
                self._rtc_enabled = False
                self._gr00t = None
                self._policy_type = None
        try:
            match = self._resolve()
            self._load_policy(match)
        except Exception as exc:
            tb = traceback.format_exc()
            self._log("error", f"Model load failed: {exc}\n{tb}")
            with self._lock:
                self._state = self.STATE_ERROR
                self._message = f"Load failed: {exc}"
            return

        with self._lock:
            self._state = self.STATE_LOADED
            self._message = f"Loaded version {self._version} (idle)"

        if autostart:
            # Same thread continues into the inference loop so the transition
            # LOADED -> RUNNING is seamless.
            self._inference_loop()

    def _begin_inference_loop(self) -> dict:
        """Spawn the inference thread against the already-loaded policy."""
        with self._lock:
            if self._state != self.STATE_LOADED:
                return {"ok": False,
                        "msg": f"Cannot start (state={self._state})"}
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._inference_loop,
            name=f"ModelRunner-run-{self._version}", daemon=True)
        self._thread.start()
        return {"ok": True, "msg": f"Started version {self._version}"}

    def _inference_loop(self) -> None:
        with self._lock:
            self._state = self.STATE_RUNNING
            self._message = f"Running version {self._version}"
            self._step_count = 0
        # Clear any leftover chunk state from a prior run so this Start plans
        # fresh from current obs.
        if self._gr00t is not None:
            try:
                self._gr00t.reset()
            except Exception as exc:
                self._log("warn", f"Gr00tBackend.reset() failed: {exc}")
        # ``hz`` is the *requested* playback rate. The actual rate is
        # auto-clamped each tick to chunk_size / (2 * inference_time) so the
        # action queue never starves between chunks. With T=4s inference and
        # chunk=20, this caps playback at ~2.5 Hz, which eliminates the "1s of
        # motion, 3s of pause" burst pattern.
        self._requested_hz = float(self.hz)
        self._log("info",
                  f"Inference loop starting at {self.hz:.1f} Hz "
                  f"(period={1.0 / self.hz * 1000:.1f} ms); "
                  "will auto-clamp to chunk_size/(2*inference_time) once measured.")

        next_t = time.monotonic()
        last_log = time.monotonic()
        steps_since_log = 0

        while not self._stop_evt.is_set():
            try:
                self._step()
                steps_since_log += 1
                with self._lock:
                    self._step_count += 1
            except Exception as exc:
                tb = traceback.format_exc()
                self._log("error", f"Inference step failed: {exc}\n{tb}")
                with self._lock:
                    self._state = self.STATE_ERROR
                    self._message = f"Step failed: {exc}"
                return

            # Recompute period every tick from current effective hz so the loop
            # slows down once we have measured inference latency.
            period = 1.0 / max(self._effective_hz(), 1e-3)

            now = time.monotonic()
            if now - last_log >= 1.0:
                with self._lock:
                    self._fps = steps_since_log / (now - last_log)
                    fps = self._fps
                    steps = self._step_count
                self._log("info",
                          f"heartbeat steps={steps} fps={fps:.1f} state={self._state}")
                steps_since_log = 0
                last_log = now

            next_t += period
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                if self._stop_evt.wait(sleep_for):
                    break
            else:
                # Fell behind — reset cadence.
                next_t = time.monotonic()

    def _resolve(self) -> dict:
        versions = list_versions(self.base_dir)
        match = next((v for v in versions if v["version"] == self._version), None)
        if match is None:
            raise FileNotFoundError(
                f"Version {self._version!r} not found under {self.base_dir}")
        if not match["has_checkpoint"]:
            raise FileNotFoundError(
                f"No checkpoint resolved inside version {self._version}")
        return match

    def _load_policy(self, match: dict) -> None:
        """Dispatch to the correct backend loader based on ``match["policy_type"]``."""
        ckpt_dir = Path(match["checkpoint_path"])
        ptype = match.get("policy_type", POLICY_SMOLVLA)
        self._policy_type = ptype
        if ptype == POLICY_GR00T:
            self._gr00t = Gr00tBackend(
                ckpt_dir=ckpt_dir,
                logger=self.logger,
                task=self._task,
            )
            self._gr00t.max_steps_per_chunk = self._max_steps_per_chunk
            self._gr00t.ensemble_window = self._ensemble_window
            self._gr00t.load()
            # Mark the policy slot as occupied so _step does not skip.
            self._policy = self._gr00t
            return
        if ptype == POLICY_ACT:
            self._load_policy_act(ckpt_dir)
            return
        # Default: SmolVLA path.
        self._load_policy_smolvla(ckpt_dir)

    def _load_policy_act(self, ckpt_dir: Path) -> None:
        """Load a LeRobot ACT (Action Chunking Transformer) checkpoint.

        ACT does not use RTC — the policy maintains its own internal action
        queue and ``select_action`` returns one action per call. Inference is run
        synchronously in the step loop.
        """
        import torch

        ACTPolicy = None
        import_errors = []
        for path in (
            "lerobot.policies.act.modeling_act",
            "lerobot.common.policies.act.modeling_act",
        ):
            try:
                module = __import__(path, fromlist=["ACTPolicy"])
                ACTPolicy = module.ACTPolicy
                self._log("info", f"Imported ACTPolicy from {path}")
                break
            except Exception as exc:
                import_errors.append((path, exc))
                self._log("warn", f"Import {path} failed: {exc!r}")
        if ACTPolicy is None:
            errs = "; ".join(f"{p!r}: {e!r}" for p, e in import_errors)
            raise ImportError(
                "Could not import ACTPolicy from lerobot. "
                f"Install/upgrade lerobot. All attempts failed: {errs}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._log("info",
                  f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
                  f"device={device}")
        self._log("info", f"Loading ACT checkpoint dir: {ckpt_dir}")
        t0 = time.monotonic()
        policy = ACTPolicy.from_pretrained(str(ckpt_dir))
        self._log("info",
                  f"from_pretrained returned in {time.monotonic() - t0:.2f}s; "
                  f"moving to {device}")
        policy.to(device)
        policy.eval()
        if hasattr(policy, "reset"):
            policy.reset()

        # Introspect expected input / output features so users see if the
        # observation we build doesn't match what the policy wants.
        cfg = getattr(policy, "config", None)
        try:
            inp = getattr(cfg, "input_features", None) if cfg else None
            if inp:
                self._log("info", f"Policy input_features keys: {list(inp.keys())}")
            out = getattr(cfg, "output_features", None) if cfg else None
            if out:
                self._log("info", f"Policy output_features keys: {list(out.keys())}")
        except Exception as exc:
            self._log("warn", f"Could not introspect policy config: {exc}")

        # Cache ACT-specific shapes so _build_batch_act can match them.
        self._act_state_dim = 6
        self._act_action_dim = 6
        self._act_image_keys: list[str] = []
        try:
            if cfg and getattr(cfg, "input_features", None):
                for key, feat in cfg.input_features.items():
                    if key == "observation.state":
                        shape = getattr(feat, "shape", None) or feat.get("shape")  # type: ignore[union-attr]
                        if shape:
                            self._act_state_dim = int(shape[0])
                    elif key.startswith("observation.images."):
                        self._act_image_keys.append(key)
            if cfg and getattr(cfg, "output_features", None):
                act_feat = cfg.output_features.get("action")
                shape = getattr(act_feat, "shape", None) or (act_feat.get("shape") if act_feat else None)  # type: ignore[union-attr]
                if shape:
                    self._act_action_dim = int(shape[0])
        except Exception as exc:
            self._log("warn", f"Could not parse ACT feature shapes: {exc}")
        self._log("info",
                  f"ACT shapes: state_dim={self._act_state_dim} "
                  f"action_dim={self._act_action_dim} "
                  f"image_keys={self._act_image_keys}")

        self._torch = torch
        self._policy = policy
        self._device = device

        # Pre/post processors (same format as SmolVLA).
        self._preprocessor = None
        self._postprocessor = None
        try:
            from lerobot.processor.pipeline import DataProcessorPipeline
            pre_cfg = ckpt_dir / "policy_preprocessor.json"
            post_cfg = ckpt_dir / "policy_postprocessor.json"
            if pre_cfg.exists():
                overrides = {"device_processor": {"device": device}}
                self._preprocessor = DataProcessorPipeline.from_pretrained(
                    str(ckpt_dir),
                    config_filename="policy_preprocessor.json",
                    overrides=overrides,
                )
                self._log("info", "Loaded preprocessor pipeline: "
                          f"{[s.__class__.__name__ for s in self._preprocessor.steps]}")
            if post_cfg.exists():
                self._postprocessor = DataProcessorPipeline.from_pretrained(
                    str(ckpt_dir),
                    config_filename="policy_postprocessor.json",
                )
                self._log("info", "Loaded postprocessor pipeline: "
                          f"{[s.__class__.__name__ for s in self._postprocessor.steps]}")
        except Exception as exc:
            self._log("warn", f"Failed to load pre/post-processors: {exc!r}")

        # ── Action mode (absolute joint targets vs per-tick deltas) ───
        # LeRobot ACT configs do not record whether the dataset's ``action``
        # column held absolute joint targets or per-step deltas — the policy just
        # learns to reproduce whatever was labelled "action". If we mis-apply a
        # delta-trained model as absolute, the arm flies to ≈ zero joints
        # (because deltas are millirad-scale near zero), which is exactly the
        # failure mode 'houston_lerobot_fixed' exhibits. Resolution order:
        #   1. Sidecar ``action_mode.json`` next to config.json:
        #        {"mode": "delta"}   or   {"mode": "absolute"}
        #   2. ACT_ACTION_MODE env override (same values).
        #   3. Heuristic on the un-normalizer stats: if max(|mean|) is tiny AND
        #      the action range is tiny relative to typical joint travel, treat
        #      as delta.
        self._act_action_is_delta = self._detect_act_action_mode(ckpt_dir)
        self._act_delta_clip = 0.20  # rad/tick safety cap when delta
        self._log("info",
                  "ACT action mode: "
                  f'{"DELTA (added to current state)" if self._act_action_is_delta else "ABSOLUTE"} '
                  f"(delta clip = ±{self._act_delta_clip} rad/tick)")

        # ACT does not use RTC; inference is synchronous.
        self._rtc_enabled = False
        self._rtc_cfg = None
        self._action_queue = None
        self._log("info", "ACT policy loaded and ready.")

    def _detect_act_action_mode(self, ckpt_dir: Path) -> bool:
        """Return True if the ACT checkpoint's ``action`` column is per-tick deltas.

        Resolution order: sidecar ``action_mode.json`` -> ``ACT_ACTION_MODE`` env
        var -> heuristic on the un-normalizer mean/min/max stats.
        """
        # 1. Sidecar file
        sidecar = ckpt_dir / "action_mode.json"
        if sidecar.exists():
            try:
                with open(sidecar) as f:
                    mode = str(json.load(f).get("mode", "")).lower().strip()
                if mode in ("delta", "absolute"):
                    self._log("info", f"action_mode.json -> {mode}")
                    return mode == "delta"
                self._log("warn", f"action_mode.json: unknown mode {mode!r}; falling back")
            except Exception as exc:
                self._log("warn", f"action_mode.json read failed: {exc!r}")

        # 2. Env override
        env_mode = os.environ.get("ACT_ACTION_MODE", "").lower().strip()
        if env_mode in ("delta", "absolute"):
            self._log("info", f"ACT_ACTION_MODE={env_mode}")
            return env_mode == "delta"

        # 3. Heuristic on un-normalizer stats
        stats_file = ckpt_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        if not stats_file.exists():
            self._log("warn", "No unnormalizer stats found; defaulting to ABSOLUTE")
            return False
        try:
            from safetensors import safe_open
            stats = {}
            with safe_open(str(stats_file), framework="np") as g:
                for k in g.keys():
                    if k.startswith("action."):
                        stats[k.split(".", 1)[1]] = g.get_tensor(k)
            mean = np.asarray(stats.get("mean", []), dtype=np.float64)
            mn = np.asarray(stats.get("min", []), dtype=np.float64)
            mx = np.asarray(stats.get("max", []), dtype=np.float64)
            if mean.size == 0 or mn.size == 0 or mx.size == 0:
                self._log("warn", "Unnormalizer stats incomplete; defaulting to ABSOLUTE")
                return False
            max_abs_mean = float(np.max(np.abs(mean)))
            max_range = float(np.max(mx - mn))
            # Absolute UR joint trajectories typically have |mean| ~ 1-2 rad and
            # per-joint range > 1 rad. Delta-action datasets sit near zero mean
            # with sub-radian total range.
            is_delta = (max_abs_mean < 0.10) and (max_range < 0.5)
            self._log("info",
                      f"action stats: max(|mean|)={max_abs_mean:.4f} "
                      f"max(range)={max_range:.4f} -> "
                      f'{"DELTA" if is_delta else "ABSOLUTE"} (heuristic)')
            return is_delta
        except Exception as exc:
            self._log("warn", f"Action-mode heuristic failed: {exc!r}; defaulting to ABSOLUTE")
            return False

    def _load_policy_smolvla(self, ckpt_dir: Path) -> None:
        """Lazy-import torch + lerobot, then load the policy."""
        import torch

        # LeRobot has reorganised modules across versions. Try newest first.
        SmolVLAPolicy = None
        import_errors = []
        for path in (
            "lerobot.policies.smolvla.modeling_smolvla",
            "lerobot.common.policies.smolvla.modeling_smolvla",
        ):
            try:
                module = __import__(path, fromlist=["SmolVLAPolicy"])
                SmolVLAPolicy = module.SmolVLAPolicy
                self._log("info", f"Imported SmolVLAPolicy from {path}")
                break
            except Exception as exc:
                import_errors.append((path, exc))
                self._log("warn", f"Import {path} failed: {exc!r}")
        if SmolVLAPolicy is None:
            errs = "; ".join(f"{p!r}: {e!r}" for p, e in import_errors)
            raise ImportError(
                "Could not import SmolVLAPolicy from lerobot. "
                f"Install/upgrade lerobot[smolvla]. All attempts failed: {errs}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._log("info",
                  f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
                  f"device={device}")
        if device == "cuda":
            try:
                self._log("info", f"cuda_device={torch.cuda.get_device_name(0)}")
            except Exception:
                pass
        self._log("info", f"Loading SmolVLA checkpoint dir: {ckpt_dir}")
        t0 = time.monotonic()
        policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir))
        self._log("info",
                  f"from_pretrained returned in {time.monotonic() - t0:.2f}s; "
                  f"moving to {device}")
        policy.to(device)
        policy.eval()
        # ── Enable Real-Time Chunking (RTC) ──
        # Instead of running select_action() (which blocks for ~700ms on Orin
        # every chunk_size ticks), use predict_action_chunk() with RTC guidance.
        # RTC overlaps each new chunk with the un-executed tail of the previous
        # one, eliminating the discontinuity that makes plain chunked rollout
        # look like ``20 fast steps then a 1s pause``. Inference runs on a
        # background thread so the main loop pops actions from a shared queue at
        # full rate while the GPU computes the next chunk in parallel.
        self._rtc_enabled = False
        try:
            from lerobot.policies.rtc.action_queue import ActionQueue
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            # execution_horizon: how far into the chunk RTC blends with the
            # previous chunk's tail. A typical value is half of the chunk size;
            # the SmolVLA default is 10 for chunk_size=20.
            chunk_size = int(getattr(policy.config, "chunk_size", 20))
            rtc_cfg = RTCConfig(
                enabled=True,
                execution_horizon=min(10, chunk_size),
                max_guidance_weight=10.0,
            )
            policy.config.rtc_config = rtc_cfg
            policy.init_rtc_processor()
            self._rtc_cfg = rtc_cfg
            self._action_queue = ActionQueue(rtc_cfg)
            self._rtc_enabled = True
            self._log("info",
                      f"RTC enabled: chunk_size={chunk_size}, "
                      f"execution_horizon={rtc_cfg.execution_horizon}")
        except Exception as exc:
            self._log("warn",
                      f"Could not enable RTC ({exc!r}); falling back to "
                      "select_action()")
            self._rtc_cfg = None
            self._action_queue = None
        if hasattr(policy, "reset"):
            policy.reset()
        # Best-effort: print expected input keys so the user sees if the
        # observation we build doesn't match what the policy wants.
        try:
            cfg = getattr(policy, "config", None)
            inp = getattr(cfg, "input_features", None) if cfg else None
            if inp:
                self._log("info", f"Policy input_features keys: {list(inp.keys())}")
            out = getattr(cfg, "output_features", None) if cfg else None
            if out:
                self._log("info", f"Policy output_features keys: {list(out.keys())}")
        except Exception as exc:
            self._log("warn", f"Could not introspect policy config: {exc}")

        self._torch = torch
        self._policy = policy
        self._device = device

        # ── Load preprocessor / postprocessor pipelines ──
        # Modern lerobot SmolVLA checkpoints ship with a preprocessor that
        # tokenises the task string into ``observation.language.tokens`` and
        # normalises state/images, plus a postprocessor that un-normalises the
        # action. Without these, ``select_action`` raises KeyError on
        # ``observation.language.tokens``.
        self._preprocessor = None
        self._postprocessor = None
        try:
            from lerobot.processor.pipeline import DataProcessorPipeline
            pre_cfg = ckpt_dir / "policy_preprocessor.json"
            post_cfg = ckpt_dir / "policy_postprocessor.json"
            if pre_cfg.exists():
                # Force the device_processor step to our actual device so the
                # pipeline does not try to move tensors to a missing GPU.
                overrides = {"device_processor": {"device": device}}
                self._preprocessor = DataProcessorPipeline.from_pretrained(
                    str(ckpt_dir),
                    config_filename="policy_preprocessor.json",
                    overrides=overrides,
                )
                self._log("info", "Loaded preprocessor pipeline: "
                          f"{[s.__class__.__name__ for s in self._preprocessor.steps]}")
            else:
                self._log("warn", "No policy_preprocessor.json found; "
                          "feeding raw batch to policy (may fail).")
            if post_cfg.exists():
                self._postprocessor = DataProcessorPipeline.from_pretrained(
                    str(ckpt_dir),
                    config_filename="policy_postprocessor.json",
                )
                self._log("info", "Loaded postprocessor pipeline: "
                          f"{[s.__class__.__name__ for s in self._postprocessor.steps]}")
        except Exception as exc:
            self._log("warn", f"Failed to load pre/post-processors: {exc!r}")

        self._log("info", "Policy loaded and ready.")

    def _effective_hz(self) -> float:
        """Clamp the requested playback rate so the action queue does not drain
        faster than the GPU can produce new chunks.

        Steady-state requirement (see comments at top of _run): for the queue to
        stay non-empty across inferences, we need
        ``chunk_size >= 2 * hz * inference_time``. Solving for hz gives
        ``hz <= chunk_size / (2 * inference_time)``. We also leave a 20% safety
        margin and never go below 1 Hz.
        """
        req = max(0.1, float(self._requested_hz))
        # NOTE: speed_scale is no longer applied here. The UR controller now
        # enforces the speed cap directly via setSpeedSlider (driven from
        # destination_writer), so scaling the playback hz on top would
        # double-scale GR00T rollouts. We keep _speed_scale on the runner only
        # for status reporting / future use.
        if not self._rtc_enabled or self._infer_latency_ema <= 0.0:
            return req
        try:
            chunk_size = int(getattr(self._policy.config, "chunk_size", 20))
        except Exception:
            chunk_size = 20
        cap = chunk_size / (2.0 * self._infer_latency_ema * 1.2)
        cap = max(1.0, cap)
        eff = min(req, cap)
        # Re-log only on meaningful change.
        if abs(eff - self._announced_hz) >= 0.25:
            self._announced_hz = eff
            self._log("info",
                      f"Effective hz adapted: requested={req:.1f} cap={cap:.2f} "
                      f"using={eff:.2f} (T_inf={self._infer_latency_ema * 1000:.0f}ms, "
                      f"chunk={chunk_size})")
        return eff

    # ── Single step ──────────────────────────────────────────────────

    def _step(self) -> None:
        if self._obs_provider is None:
            self._skip_log("no observation_provider registered")
            return
        if self._policy is None:
            self._skip_log("policy not loaded yet")
            return
        obs = self._obs_provider()
        if obs is None:
            self._skip_log("observation_provider returned None (cameras/joints not ready?)")
            return
        # Per-step debug (rate-limited): show what we're feeding in.
        self._debug_obs(obs)

        # GR00T uses a self-contained chunked rollout — dispatch and return.
        if self._policy_type == POLICY_GR00T:
            self._step_gr00t(obs)
            return

        # ACT uses a synchronous select_action() call with its own internal
        # action queue.
        if self._policy_type == POLICY_ACT:
            self._step_act(obs)
            return

        # ── Decide whether to launch a fresh inference ──
        # Trigger background inference whenever the queue is empty (first tick)
        # or its remaining size has dropped below the RTC execution horizon. This
        # keeps the queue topped up and lets RTC blend the new chunk with the
        # un-executed tail of the previous one.
        if self._rtc_enabled and not self._infer_busy.is_set():
            qsize = self._action_queue.qsize()
            threshold = max(1, self._rtc_cfg.execution_horizon // 2)
            if qsize <= threshold:
                self._launch_inference(obs)

        # ── Pop the next action from the queue ──
        action_t = None
        if self._rtc_enabled:
            action_t = self._action_queue.get()
        if action_t is None:
            # Queue empty (first inference still running or RTC disabled). Skip
            # this tick — destination_writer holds last commanded pose.
            if self._rtc_enabled:
                self._skip_log("action queue empty (waiting for first chunk)")
            else:
                self._skip_log("RTC disabled and no fallback wired")
            return

        action = action_t.detach().cpu().numpy().reshape(-1)
        self._publish(action)

    def _step_gr00t(self, obs: dict) -> None:
        """One inference tick for the GR00T backend."""
        assert self._gr00t is not None
        try:
            nxt = self._gr00t.next_action(obs)
        except Exception as exc:
            tb = traceback.format_exc()
            self._log("error", f"gr00t inference failed: {exc}\n{tb}")
            return
        if nxt is None:
            self._skip_log("gr00t produced no action (obs incomplete or first chunk pending)")
            return
        vec6, gripper = nxt
        # TCP-EE flavour outputs a Cartesian pose target; publish it on
        # /mirror/tcp_pose_cmd via the dedicated callback so destination_writer
        # drives the arm with servoL (UR controller does the IK).
        if self._gr00t.flavor == FLAVOR_TCP_EE:
            if self.publish_tcp is None:
                self._skip_log("GR00T TCP-EE policy loaded but no publish_tcp callback wired")
                return
            g = max(0.0, min(1.0, float(gripper)))
            with self._lock:
                self._last_action = list(vec6) + [g]
                self._last_step_time = time.time()
            now = time.monotonic()
            if now - getattr(self, "_last_action_log", 0.0) >= 1.0:
                self._last_action_log = now
                p_str = ", ".join(f"{x:+.3f}" for x in vec6)
                self._log("info",
                          f"step#{self._step_count} tcp_pose=[{p_str}] "
                          f"gripper={g:.3f}")
            try:
                self.publish_tcp(list(vec6), g)
            except Exception as exc:
                self._log("warn", f"publish_tcp raised: {exc}")
            return
        # Joint flavour: legacy path.
        action_vec = list(vec6) + [gripper]
        self._publish(np.asarray(action_vec, dtype=np.float64))

    def _step_act(self, obs: dict) -> None:
        """One inference tick for an ACT policy.

        ACT manages its own internal action chunk queue, so we simply call
        ``select_action`` synchronously on every tick. The policy produces a
        fresh chunk every ``chunk_size`` calls.
        """
        torch = self._torch
        batch = self._build_batch_act(obs)
        if batch is None:
            return
        try:
            if self._preprocessor is not None:
                batch = self._preprocessor(batch)
            with torch.no_grad():
                action = self._policy.select_action(batch)
            if self._postprocessor is not None:
                out = self._postprocessor({"action": action})
                if isinstance(out, dict) and "action" in out:
                    action = out["action"]
                else:
                    action = out
        except Exception as exc:
            tb = traceback.format_exc()
            self._log("error", f"ACT inference failed: {exc}\n{tb}")
            return
        # action: (B=1, A) — squeeze batch dim.
        if action.ndim == 2:
            action = action[0]
        action_np = action.detach().cpu().numpy().reshape(-1)
        # Delta-action models predict per-tick joint deltas; convert to absolute
        # targets by adding the current observed joints. Clip each delta to
        # ±self._act_delta_clip rad so a garbage chunk cannot fling the arm.
        if getattr(self, "_act_action_is_delta", False):
            cur = np.asarray(obs.get("joints") or [0.0] * 6, dtype=np.float64)[:6]
            n = min(6, action_np.shape[0])
            delta = np.clip(action_np[:n].astype(np.float64),
                            -self._act_delta_clip, self._act_delta_clip)
            abs_joints = cur[:n] + delta
            action_np = np.concatenate([abs_joints, action_np[n:].astype(np.float64)])
        # If the ACT model has no gripper dim (action_dim=6), pass through the
        # currently observed gripper so we don't force it open every tick.
        if action_np.shape[0] < 7:
            cur_gripper = float(obs.get("gripper", 0.0))
            action_np = np.concatenate([action_np[:6], [cur_gripper]])
        self._publish(action_np)

    def _publish(self, action: Any) -> None:
        """Convert raw action vector to (joints, gripper) and publish."""
        action_list = [float(x) for x in action]
        joints = action_list[:6]
        gripper = action_list[6] if len(action_list) > 6 else 0.0
        # SmolVLA gripper is sometimes scaled to [0, 255]. Normalise to 0..1.
        if gripper > 1.5:
            gripper = max(0.0, min(1.0, gripper / 255.0))
        else:
            gripper = max(0.0, min(1.0, gripper))

        with self._lock:
            self._last_action = action_list
            self._last_step_time = time.time()

        # Periodic action log: every ~1s so we can see something is moving.
        now = time.monotonic()
        if now - getattr(self, "_last_action_log", 0.0) >= 1.0:
            self._last_action_log = now
            j_str = ", ".join(f"{x:+.3f}" for x in joints)
            qs = self._action_queue.qsize() if self._action_queue is not None else -1
            self._log("info",
                      f"step#{self._step_count} action joints=[{j_str}] "
                      f"gripper={gripper:.3f} qsize={qs} "
                      f"inf_delay={self._inference_delay_ticks}")

        try:
            self.publish_action(joints, gripper)
        except Exception as exc:
            self._log("warn", f"publish_action raised: {exc}")

    # ── Background inference worker ──────────────────────────────────

    def _launch_inference(self, obs: dict) -> None:
        """Build a batch from the current obs and dispatch a background
        inference. Returns immediately; the worker thread will populate the
        action queue when it finishes.
        """
        # Build the batch synchronously here so the obs (joints + cameras)
        # reflects the state at the moment inference was queued.
        batch = self._build_batch(obs)
        if batch is None:
            return
        # Snapshot of the current queue state for RTC guidance.
        prev_left_over = self._action_queue.get_left_over()
        idx_before = self._action_queue.get_action_index()
        # If the previous chunk's leftover lives on a different device than the
        # policy, move it. ActionQueue stores tensors on the device they were
        # merged with (we always merge on CPU below).
        if prev_left_over is not None:
            prev_left_over = prev_left_over.to(self._device)

        self._infer_busy.set()
        threading.Thread(
            target=self._run_inference,
            args=(batch, prev_left_over, idx_before),
            name="SmolVLA-inference",
            daemon=True,
        ).start()

    def _run_inference(self, batch: Any, prev_left_over: Any, idx_before: int) -> None:
        torch = self._torch
        t_start = time.monotonic()
        try:
            # Apply preprocessor (rename, batchify, tokenize, device, normalize)
            # inside the worker so the main loop is not blocked by the tokenizer.
            if self._preprocessor is not None:
                try:
                    batch = self._preprocessor(batch)
                except Exception as exc:
                    self._log("warn", f"preprocessor raised: {exc!r}")
                    return

            # Build kwargs for RTC-aware predict_action_chunk.
            kwargs = {}
            if self._rtc_enabled and prev_left_over is not None:
                # ``inference_delay`` tells RTC how many of the new chunk's
                # leading actions are already locked in (because they will be
                # executed before this inference completes). Use the smoothed
                # estimate from prior cycles.
                kwargs["inference_delay"] = int(self._inference_delay_ticks)
                kwargs["prev_chunk_left_over"] = prev_left_over

            # Inference itself. Serialised by the lock so two threads cannot
            # share the policy at once.
            #
            # NOTE: use torch.no_grad() rather than torch.inference_mode(). RTC's
            # denoise_step (lerobot/policies/rtc/modeling_rtc.py) opens a `with
            # torch.enable_grad():` block and calls torch.autograd.grad() to
            # compute the guidance correction. `inference_mode` is stronger than
            # `no_grad` and CANNOT be overridden by `enable_grad`, so tensors
            # created inside it have no autograd graph and the grad call raises
            # "element 0 of tensors does not require grad and does not have a
            # grad_fn". Symptom: the first chunk runs fine (no
            # prev_chunk_left_over → RTC path is skipped) but every subsequent
            # inference fails, the queue drains, and the robot stops moving after
            # a few seconds.
            with self._infer_lock, torch.no_grad():
                actions = self._policy.predict_action_chunk(batch, **kwargs)
            # actions: (B=1, T, A) where A may be > action_dim due to
            # zero-padding in SmolVLA; predict_action_chunk already trims to
            # original_action_dim.
            if actions.ndim == 2:
                actions = actions.unsqueeze(0)
            B, T, A = actions.shape

            # Postprocess each timestep individually — the
            # UnnormalizerProcessorStep expects (B, action_dim), not (B, T,
            # action_dim). This matches lerobot.async_inference.
            processed = []
            for i in range(T):
                single = actions[:, i, :]  # (B, A)
                if self._postprocessor is not None:
                    out = self._postprocessor({"action": single})
                    if isinstance(out, dict) and "action" in out:
                        single = out["action"]
                    else:
                        single = out
                processed.append(single)
            processed_actions = torch.stack(processed, dim=1).squeeze(0).detach().cpu()  # (T, A)
            original_actions = actions.squeeze(0).detach().cpu()  # (T, A)

            # Compute real_delay from how many actions the main loop consumed
            # while we were running.
            idx_after = self._action_queue.get_action_index()
            real_delay = max(0, idx_after - idx_before)
            # Cap real_delay so we don't drop the entire chunk if the worker took
            # longer than chunk_size ticks.
            real_delay = min(real_delay, T - 1)

            self._action_queue.merge(
                original_actions=original_actions,
                processed_actions=processed_actions,
                real_delay=real_delay,
                action_index_before_inference=idx_before,
            )

            # Update inference_delay smoothed estimate (in ticks).
            elapsed = time.monotonic() - t_start
            self._infer_count += 1
            self._infer_total_s += elapsed
            # Latency EMA used by _effective_hz to clamp playback rate.
            if self._infer_latency_ema <= 0.0:
                self._infer_latency_ema = elapsed
            else:
                self._infer_latency_ema = (
                    0.7 * self._infer_latency_ema + 0.3 * elapsed)
            # Inter-inference period EMA (actual achieved inference Hz).
            now_done = time.monotonic()
            if self._last_infer_t > 0.0:
                period = now_done - self._last_infer_t
                if self._infer_period_ema <= 0.0:
                    self._infer_period_ema = period
                else:
                    self._infer_period_ema = (
                        0.7 * self._infer_period_ema + 0.3 * period)
            self._last_infer_t = now_done
            ticks_consumed = elapsed * self._effective_hz()
            # EMA with alpha=0.5 — adapts in 2-3 cycles.
            self._inference_delay_ticks = int(round(
                0.5 * self._inference_delay_ticks + 0.5 * ticks_consumed))
            self._inference_delay_ticks = max(1, min(
                self._inference_delay_ticks, T - 1))

            now = time.monotonic()
            if now - getattr(self, "_last_infer_log", 0.0) >= 1.0:
                self._last_infer_log = now
                avg_s = self._infer_total_s / self._infer_count
                self._log("info",
                          f"inference done: chunk_T={T} elapsed={elapsed * 1000:.0f}ms "
                          f"avg={avg_s * 1000:.0f}ms real_delay={real_delay} "
                          f"qsize_after_merge={self._action_queue.qsize()}")
        except Exception as exc:
            tb = traceback.format_exc()
            self._log("error", f"inference worker failed: {exc}\n{tb}")
        finally:
            self._infer_busy.clear()

    def _build_batch(self, obs: dict) -> dict | None:
        """Build the raw observation batch (pre-preprocessor) from a snapshot of
        the GUI's observation dict. Returns ``None`` if the observation is
        malformed.
        """
        torch = self._torch
        device = self._device

        joints = obs.get("joints")
        if not joints or len(joints) < 6:
            self._skip_log(f"joints invalid (len={len(joints) if joints else 0})")
            return None
        # State is 7-dim: 6 joint positions + 1 gripper position (0..1).
        gripper = float(obs.get("gripper", 0.0))
        state_vec = list(joints[:6]) + [gripper]
        state = torch.tensor(state_vec, dtype=torch.float32, device=device)
        state = state.unsqueeze(0)  # (1, 7)

        cam1 = self._prep_image(obs.get("camera1"))
        cam2 = self._prep_image(obs.get("camera2"))

        if cam1 is None or cam2 is None:
            self._skip_log(
                f"image prep failed cam1={cam1 is not None} "
                f"cam2={cam2 is not None}")
            return None

        empty = torch.zeros(1, 3, 480, 640, dtype=torch.float32, device=device)

        return {
            "observation.state": state,
            "observation.images.camera1": cam1,
            "observation.images.camera2": cam2,
            "observation.images.empty_camera_0": empty,
            "task": [self._task],
        }

    def _build_batch_act(self, obs: dict) -> dict | None:
        """Build the raw observation batch for an ACT policy.

        ACT models expect:
          * ``observation.state``: joints-only vector matching
            ``config.input_features["observation.state"].shape`` (typically 6 for
            a single UR10e arm, with no gripper).
          * One or more ``observation.images.<key>`` tensors. We map the
            available cameras to the keys advertised in ``self._act_image_keys``
            in order: first key gets camera1, second gets camera2, etc. An
            ``ACT_CAMERA_MAP`` env var of the form ``key1=camera1,key2=camera2``
            overrides this.
        """
        torch = self._torch
        device = self._device

        joints = obs.get("joints")
        if not joints or len(joints) < 6:
            self._skip_log(f"joints invalid (len={len(joints) if joints else 0})")
            return None

        # State dim: trim or extend joints to match expected dim. If the model
        # wants 7-d state we include the gripper at the end.
        sd = int(getattr(self, "_act_state_dim", 6))
        if sd >= 7:
            state_vec = list(joints[:6]) + [float(obs.get("gripper", 0.0))]
            state_vec = state_vec[:sd]
        else:
            state_vec = list(joints[:sd])
        state = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)

        # Resolve camera mapping.
        image_keys = list(getattr(self, "_act_image_keys", []) or [])
        if not image_keys:
            # Fall back to the conventional single-camera key.
            image_keys = ["observation.images.color"]

        cam_map_env = os.environ.get("ACT_CAMERA_MAP", "").strip()
        cam_map = {}
        if cam_map_env:
            for pair in cam_map_env.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    cam_map[k.strip()] = v.strip()

        available = ["camera1", "camera2", "camera3"]
        batch = {"observation.state": state}
        for i, key in enumerate(image_keys):
            cam_name = cam_map.get(key, available[i] if i < len(available) else "camera1")
            tensor = self._prep_image(obs.get(cam_name))
            if tensor is None:
                self._skip_log(f"image prep failed for {key} <- {cam_name}")
                return None
            batch[key] = tensor

        batch["task"] = [self._task]
        return batch

    def _skip_log(self, reason: str) -> None:
        """Rate-limited warning when a step is skipped."""
        now = time.monotonic()
        last = getattr(self, "_last_skip_log", 0.0)
        if now - last < 2.0:
            return
        self._last_skip_log = now
        self._log("warn", f"Inference step skipped: {reason}")

    def _debug_obs(self, obs: dict) -> None:
        now = time.monotonic()
        last = getattr(self, "_last_obs_log", 0.0)
        if now - last < 2.0:
            return
        self._last_obs_log = now
        joints = obs.get("joints")
        c1 = obs.get("camera1")
        c2 = obs.get("camera2")
        c3 = obs.get("camera3")

        def _shape(x: Any) -> tuple | None:
            try:
                return tuple(x.shape)
            except Exception:
                return None

        self._log("info",
                  f"obs joints_len={len(joints) if joints else 0} "
                  f"cam1={_shape(c1)} cam2={_shape(c2)} cam3={_shape(c3)} "
                  f"task={self._task!r}")

    def _infer(self, obs: dict) -> np.ndarray | None:
        """Deprecated synchronous path. Kept only as a fallback for environments
        where RTC could not be enabled (see ``_load_policy``). Mirrors the old
        behaviour: build batch, preprocess, select_action, postprocess. Returns a
        numpy array or ``None``.
        """
        torch = self._torch
        batch = self._build_batch(obs)
        if batch is None:
            return None

        if self._preprocessor is not None:
            try:
                batch = self._preprocessor(batch)
            except Exception as exc:
                self._skip_log(f"preprocessor raised: {exc!r}")
                return None

        with torch.inference_mode():
            action = self._policy.select_action(batch)

        if self._postprocessor is not None:
            try:
                out = self._postprocessor({"action": action})
                if isinstance(out, dict) and "action" in out:
                    action = out["action"]
                else:
                    action = out
            except Exception as exc:
                self._log("warn", f"postprocessor raised: {exc!r}")

        if hasattr(action, "detach"):
            action = action.detach().to("cpu").numpy()
        return np.asarray(action).reshape(-1)

    def _prep_image(self, frame: np.ndarray | None,
                    size: tuple[int, int] | None = None) -> Any:
        """Convert a BGR uint8 ndarray to a (1, 3, H, W) float tensor in [0,1].

        When ``size`` is None the frame is left at its native resolution and
        SmolVLA's internal aspect-preserving ``resize_with_pad`` will scale it to
        512x512 with letterbox padding. This is what the policy saw during
        training; pre-resizing to a square here introduces the depth-perception
        bug seen on a stretched view.
        """
        if frame is None:
            return None
        import cv2

        torch = self._torch
        if frame.ndim != 3 or frame.shape[2] != 3:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if size is not None:
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        arr = rgb.astype(np.float32) / 255.0
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).unsqueeze(0).to(self._device)
