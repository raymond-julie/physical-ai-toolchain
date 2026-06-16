#!/usr/bin/env python3
"""GR00T backend for the UR edge inference runtime.

Loads an NVIDIA Isaac-GR00T N1.5/N1.7 checkpoint (e.g. the one finetuned in
``ai_models/gr00t-n15-ur10e-train/checkpoint-40000``) and exposes a small
chunked-rollout interface that :class:`model_runner.ModelRunner` uses to drive
the destination robot.

The ``gr00t`` Python package is lazy-imported so the GUI process still starts on
machines that have not yet installed it. If a user picks a GR00T checkpoint in
the combo box and the package is missing, a clear ImportError with installation
hints is surfaced (see ``install_gr00t.sh``).

This module also defines a custom data-config matching the
``experiment_cfg/metadata.json`` that ships with the trained checkpoint::

    video.color  + video.color2          (two RGB cameras)
    state.single_arm (6) + state.gripper (1)
    action.single_arm (6) + action.gripper (1)
    annotation.human.task_description    (language)
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

DEFAULT_TASK = "pick up the object"

# Two backend variants recognised by this module:
#   FLAVOR_JOINT  -> 6 joints + 1 gripper, two cameras keyed video.color/color2
#                    (legacy gr00t-n15-ur10e-train checkpoints).
#   FLAVOR_TCP_EE -> task-space TCP pose with rot6d rotation + 1 gripper,
#                    cameras keyed video.scene / video.wrist
#                    (groot-ur10-tcp-ee-* checkpoints; see the README in
#                    ai_models/groot-ur10-tcp-ee-20k for details).
FLAVOR_JOINT = "joint"
FLAVOR_TCP_EE = "tcp_ee"


def _axis_angle_to_rot6d(rvec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector (axis*angle radians) to a rot6d vector.

    rot6d is the first two columns of the rotation matrix flattened in
    column-major order: [R[:,0], R[:,1]]. Matches the encoding written by
    ``lerobot_recorder_node._axis_angle_to_rot6d`` so inference state layout is
    identical to what the model saw during training.
    """
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-8:
        K = np.array([[0.0, -r[2], r[1]],
                      [r[2], 0.0, -r[0]],
                      [-r[1], r[0], 0.0]])
        R = np.eye(3) + K
    else:
        axis = r / theta
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        R = (np.eye(3)
             + np.sin(theta) * K
             + (1.0 - np.cos(theta)) * (K @ K))
    return R[:, :2].T.reshape(6).astype(np.float64)


def _rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_axis_angle_to_rot6d` — recover URScript rvec.

    Reconstructs the rotation matrix via Gram-Schmidt on the two predicted
    columns (Zhou et al. 2019), then converts to axis-angle.
    """
    v = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1 = v[0:3]
    a2 = v[3:6]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        return np.zeros(3, dtype=np.float64)
    b1 = a1 / n1
    a2_proj = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2_proj)
    if n2 < 1e-8:
        # Degenerate — fall back to identity rotation.
        return np.zeros(3, dtype=np.float64)
    b2 = a2_proj / n2
    b3 = np.cross(b1, b2)
    R = np.column_stack([b1, b2, b3])
    # Rotation matrix -> axis-angle (Rodrigues inverse, numerically safe).
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if abs(np.pi - theta) < 1e-6:
        # Near pi: extract axis from the symmetric part.
        M = 0.5 * (R + np.eye(3))
        diag = np.clip(np.diag(M), 0.0, None)
        axis = np.sqrt(diag)
        # Recover signs from off-diagonal terms.
        if axis[0] > 1e-6:
            axis[1] = np.copysign(axis[1], M[0, 1])
            axis[2] = np.copysign(axis[2], M[0, 2])
        elif axis[1] > 1e-6:
            axis[2] = np.copysign(axis[2], M[1, 2])
        return (axis * theta).astype(np.float64)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz]) / (2.0 * np.sin(theta))
    return (axis * theta).astype(np.float64)


def _continuous_rvec(r_new: np.ndarray, r_prev: np.ndarray | None) -> np.ndarray:
    """Pick the axis-angle representation closest to ``r_prev``.

    Any rotation has two rvec forms with the same matrix: ``r`` (short way,
    |r| < pi) and ``r * (1 - 2pi/|r|)`` (long way). When the underlying rotation
    passes near pi the canonical axis can flip sign, producing a ~2pi jump in
    rvec that the robot interprets as a real wrist spin. This picks the form
    whose euclidean distance to the previous rvec is smaller, keeping the
    commanded trajectory continuous.
    """
    r_new = np.asarray(r_new, dtype=np.float64).reshape(3)
    if r_prev is None:
        return r_new
    theta = float(np.linalg.norm(r_new))
    if theta < 1e-6:
        return r_new
    r_alt = r_new * (1.0 - 2.0 * np.pi / theta)
    if np.linalg.norm(r_alt - r_prev) < np.linalg.norm(r_new - r_prev):
        return r_alt
    return r_new


# Match the resolution recorded in the training metadata (cf. metadata.json
# ``modalities.video.color.resolution``). Inference is forgiving — gr00t
# resizes internally to 224x224 — but matching the source resolution avoids
# surprises on the user side.
DEFAULT_IMG_HW = (480, 640)


def is_gr00t_checkpoint(ckpt_dir: Path) -> bool:
    """Return True if ``ckpt_dir`` looks like an Isaac-GR00T checkpoint.

    Detection is purely a directory-structure check (no torch / gr00t imports)
    so it is safe to call from the GUI thread when enumerating available models.
    Accepts both N1.5 (``GR00T_*``) and N1.7 (``Gr00tN1d7``) architecture names.
    """
    cfg = ckpt_dir / "config.json"
    if not cfg.exists():
        return False
    try:
        with cfg.open("r") as f:
            data = json.load(f)
    except Exception:
        return False
    arch = data.get("architectures") or []
    if isinstance(arch, list):
        for a in arch:
            if isinstance(a, str) and (a.startswith("GR00T_") or a.startswith("Gr00t")):
                return True
    return False


def detect_gr00t_flavor(ckpt_dir: Path) -> str:
    """Read ``experiment_cfg/conf.yaml`` and classify the action space.

    Returns :data:`FLAVOR_TCP_EE` when the model was trained on task-space TCP
    poses (state keys include ``eef`` / video keys include ``scene``) and
    :data:`FLAVOR_JOINT` otherwise (legacy joint-space checkpoints).
    """
    conf = ckpt_dir / "experiment_cfg" / "conf.yaml"
    if not conf.exists():
        return FLAVOR_JOINT
    try:
        import yaml  # PyYAML ships with the Isaac-GR00T stack.
        with conf.open("r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return FLAVOR_JOINT
    try:
        mods = cfg["data"]["modality_configs"]
        embodiment_cfg = next(iter(mods.values()))
        state_keys = list(embodiment_cfg["state"]["modality_keys"])
        video_keys = list(embodiment_cfg["video"]["modality_keys"])
    except Exception:
        return FLAVOR_JOINT
    if "eef" in state_keys or "scene" in video_keys or "wrist" in video_keys:
        return FLAVOR_TCP_EE
    return FLAVOR_JOINT


# Action tuple types returned by Gr00tBackend.next_action():
#   joint flavor  : (joints6: list[float], gripper: float)
#   tcp-ee flavor : (tcp_pose_axis_angle6: list[float], gripper: float)
# The shape is identical; ModelRunner dispatches based on Gr00tBackend.flavor.
ActionTuple = tuple[list[float], float]


class Gr00tBackend:
    """Loads a GR00T policy and produces (vec6, gripper) actions.

    The shape of ``vec6`` depends on ``self.flavor``:
      * :data:`FLAVOR_JOINT`  — six joint targets (rad).
      * :data:`FLAVOR_TCP_EE` — TCP pose ``[x,y,z,rx,ry,rz]`` in URScript
        base-frame convention (metres + axis-angle rad), ready to publish on
        ``/mirror/tcp_pose_cmd``.

    Thread-safety: ``load`` and ``next_action`` are not internally locked —
    ModelRunner already serialises calls into ``_step``.
    """

    def __init__(self, ckpt_dir: Path, logger: Any = None,
                 task: str = DEFAULT_TASK) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.logger = logger
        self.task = task or DEFAULT_TASK
        self.flavor = detect_gr00t_flavor(self.ckpt_dir)

        self.policy = None
        self.device = "cpu"
        # Discovered from policy.modality_configs after load().
        self._language_key = "annotation.human.task_description"
        self._state_T = 1
        self._video_T = 1
        # Pending action chunk + cursor.
        self._chunk: list[ActionTuple] | None = None
        self._chunk_idx = 0
        # If set, only the first ``max_steps_per_chunk`` actions of each
        # inferred chunk are executed before forcing a re-infer. Lets the
        # operator trade compute for reactivity (smaller = more reactive,
        # higher inference rate, more compute). ``None`` plays the full
        # action_horizon (legacy behavior).
        self.max_steps_per_chunk: int | None = None
        # Discovered after the first successful inference: how many actions the
        # model emits per chunk (the action_horizon baked into training).
        # Exposed so the UI can clamp the max_steps_per_chunk control.
        self.chunk_horizon: int | None = None
        # ACT-style temporal ensembling. ``ensemble_window`` is K: how many
        # recent chunks blend at each tick. K=1 disables blending (today's
        # behavior). Older chunks contribute exponentially less via
        # ensemble_alpha (per-chunk-age weight decay).
        self.ensemble_window: int = 1
        self.ensemble_alpha: float = 0.5
        # Monotonic tick counter (one increment per next_action call) and the
        # ensemble ring buffer of (origin_tick, chunk) entries.
        self._tick: int = 0
        self._ensemble: list[tuple[int, list[ActionTuple]]] = []
        # Background inference plumbing.
        self._infer_busy = threading.Event()
        self._next_chunk: list[ActionTuple] | None = None
        self._next_chunk_lock = threading.Lock()
        self._last_infer_s = 0.0
        # Last commanded rvec, used to keep axis-angle representation continuous
        # across the theta=pi boundary (see _continuous_rvec).
        self._prev_rvec: np.ndarray | None = None

    # ── Logging ───────────────────────────────────────────────────

    def _log(self, level: str, msg: str) -> None:
        if self.logger is None:
            _LOGGER.log(_LOG_LEVELS.get(level, logging.INFO), "%s", msg)
            return
        try:
            from rclpy.logging import LoggingSeverity
            sev = {
                "debug": LoggingSeverity.DEBUG,
                "info": LoggingSeverity.INFO,
                "warn": LoggingSeverity.WARN,
                "error": LoggingSeverity.ERROR,
            }.get(level, LoggingSeverity.INFO)
            self.logger.log(msg, sev)
        except Exception:
            self.logger.info(f"[{level.upper()}] {msg}")

    # ── Model load ────────────────────────────────────────────────

    def load(self) -> None:
        """Lazy-import gr00t and instantiate Gr00tPolicy (N1.7 API)."""
        try:
            import torch  # noqa: F401
            from gr00t.data.embodiment_tags import EmbodimentTag
            from gr00t.policy.gr00t_policy import Gr00tPolicy
        except ImportError as exc:
            raise ImportError(
                "Could not import the `gr00t` package required to run this "
                "checkpoint. Install Isaac-GR00T N1.7 first: see "
                "`ur_edge/install_gr00t.sh`.\n"
                f"Underlying error: {exc!r}"
            ) from exc

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            self._log("warn", "CUDA not available — GR00T inference on CPU "
                              "will be unusably slow (>30s/step).")

        # N1.7: Gr00tPolicy reads its modality config from the checkpoint's
        # processor_config.json. All ur10 checkpoints here are registered under
        # the "new_embodiment" tag (id 10 in embodiment_id.json).
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

        self._log("info", f"Loading Gr00tPolicy ({self.flavor}) from {self.ckpt_dir} "
                          f"on {device}")
        t0 = time.monotonic()
        self.policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=str(self.ckpt_dir),
            device=device,
        )
        self.device = device
        # Discover horizons / language key from the loaded processor so
        # _build_obs assembles the exact shape check_observation expects.
        try:
            mcfg = self.policy.modality_configs
            self._language_key = mcfg["language"].modality_keys[0]
            self._state_T = len(mcfg["state"].delta_indices)
            self._video_T = len(mcfg["video"].delta_indices)
        except Exception as exc:
            self._log("warn", f"could not introspect modality_configs: {exc}")
        self._log("info",
                  f"Gr00tPolicy loaded in {time.monotonic() - t0:.1f}s "
                  f"(lang_key={self._language_key}, "
                  f"state_T={self._state_T}, video_T={self._video_T})")

    # ── Inference ─────────────────────────────────────────────────

    def _build_obs(self, obs: dict) -> dict | None:
        """Convert GUI obs into the batched N1.7 observation schema.

        Returns a dict shaped as ``check_observation`` expects:
          video    : {key: uint8 (B=1, T=video_T, H, W, 3)}
          state    : {key: float32 (B=1, T=state_T, D)}
          language : {lang_key: [[task_text]]}  (B=1, T=1)
        """
        import cv2

        cam1 = obs.get("camera1")
        cam2 = obs.get("camera2")
        if cam1 is None or cam2 is None:
            return None

        # BGR uint8 (H, W, 3) -> RGB uint8 (1, T, H, W, 3); we just repeat the
        # latest frame across the video horizon (training used the most-recent
        # stack too).
        def _video(frame: np.ndarray) -> np.ndarray:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
            stacked = np.repeat(rgb[np.newaxis, ...], self._video_T, axis=0)
            return stacked[np.newaxis, ...]  # (1, T, H, W, 3)

        def _state(vec: np.ndarray) -> np.ndarray:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            stacked = np.repeat(arr[np.newaxis, :], self._state_T, axis=0)
            return stacked[np.newaxis, ...]  # (1, T, D)

        if self.flavor == FLAVOR_TCP_EE:
            tcp = obs.get("tcp_pose")
            if tcp is None or len(tcp) < 6:
                return None
            xyz = np.asarray(tcp[:3], dtype=np.float32)
            rot6d = _axis_angle_to_rot6d(tcp[3:6]).astype(np.float32)
            eef = np.concatenate([xyz, rot6d])  # (9,)
            gripper = float(obs.get("gripper", 0.0))
            return {
                "video": {
                    "scene": _video(cam1),
                    "wrist": _video(cam2),
                },
                "state": {
                    "eef": _state(eef),
                    "gripper": _state([gripper]),
                },
                "language": {self._language_key: [[self.task]]},
            }

        # FLAVOR_JOINT (fallback for any joint-space N1.7 checkpoint)
        joints = obs.get("joints")
        if not joints or len(joints) < 6:
            return None
        gripper = float(obs.get("gripper", 0.0))
        return {
            "video": {
                "color": _video(cam1),
                "color2": _video(cam2),
            },
            "state": {
                "single_arm": _state(joints[:6]),
                "gripper": _state([gripper]),
            },
            "language": {self._language_key: [[self.task]]},
        }

    def _infer_chunk(self, obs: dict) -> list[ActionTuple] | None:
        """Run a synchronous forward pass; return a list of ``(vec6, gripper)``
        tuples (one per future tick). ``vec6`` is joint targets in
        :data:`FLAVOR_JOINT` and an axis-angle TCP pose ``[x,y,z,rx,ry,rz]`` in
        :data:`FLAVOR_TCP_EE`.
        """
        if self.policy is None:
            return None
        gobs = self._build_obs(obs)
        if gobs is None:
            return None
        t0 = time.monotonic()
        # N1.7 returns (action_dict, info_dict); keys are bare modality names
        # (no "action." prefix) and arrays are batched (B, T, D).
        action, _info = self.policy.get_action(gobs)
        self._last_infer_s = time.monotonic() - t0

        def _unbatch(key: str) -> np.ndarray:
            arr = np.asarray(action.get(key))
            if arr.ndim == 3:
                return arr[0]   # (T, D)
            return arr

        if self.flavor == FLAVOR_TCP_EE:
            eef = _unbatch("eef")        # (T, 9) absolute
            grip = _unbatch("gripper")   # (T, 1)
            if eef.ndim != 2 or eef.shape[1] < 9:
                self._log("error", f"unexpected action.eef shape {eef.shape}")
                return None
            T = eef.shape[0]
            # Seed the continuity reference from the robot's current pose if we
            # don't have a previous command yet, so the first chunk step doesn't
            # already look like a flip.
            prev_rvec = self._prev_rvec
            if prev_rvec is None:
                tcp = obs.get("tcp_pose")
                if tcp is not None and len(tcp) >= 6:
                    prev_rvec = np.asarray(tcp[3:6], dtype=np.float64)
            out: list[ActionTuple] = []
            for t in range(T):
                xyz = eef[t, 0:3]
                rvec = _rot6d_to_axis_angle(eef[t, 3:9])
                rvec = _continuous_rvec(rvec, prev_rvec)
                prev_rvec = rvec
                pose6 = [float(xyz[0]), float(xyz[1]), float(xyz[2]),
                         float(rvec[0]), float(rvec[1]), float(rvec[2])]
                g = float(grip[t, 0]) if grip.size else 0.0
                g = max(0.0, min(1.0, g))
                out.append((pose6, g))
            # Remember the last commanded rvec so the NEXT chunk also starts
            # continuously.
            self._prev_rvec = prev_rvec
            return out

        # FLAVOR_JOINT
        arm = _unbatch("single_arm")   # (T, 6)
        grip = _unbatch("gripper")     # (T, 1)
        if arm.ndim != 2 or arm.shape[1] < 6:
            self._log("error", f"unexpected action.single_arm shape {arm.shape}")
            return None
        T = arm.shape[0]
        out = []
        for t in range(T):
            joints6 = [float(x) for x in arm[t, :6]]
            g = float(grip[t, 0]) if grip.size else 0.0
            g = max(0.0, min(1.0, g))
            out.append((joints6, g))
        return out

    # ── Background prefetch ───────────────────────────────────────

    def _bg_infer(self, obs: dict) -> None:
        try:
            chunk = self._infer_chunk(obs)
            with self._next_chunk_lock:
                self._next_chunk = chunk
        except Exception as exc:
            self._log("error", f"background inference failed: {exc}")
        finally:
            self._infer_busy.clear()

    def status(self) -> dict:
        with self._next_chunk_lock:
            eff = self._effective_len(self._chunk)
            qsize = (eff - self._chunk_idx) if self._chunk else 0
        return {
            "busy": self._infer_busy.is_set(),
            "queued": qsize,
            "last_infer_ms": int(self._last_infer_s * 1000),
            "max_steps_per_chunk": self.max_steps_per_chunk,
            "chunk_horizon": self.chunk_horizon,
            "ensemble_window": self.ensemble_window,
        }

    def _effective_len(self, chunk: list[ActionTuple] | None) -> int:
        """Number of actions to actually execute from ``chunk``.

        Honors ``self.max_steps_per_chunk`` if set, otherwise returns the full
        chunk length. Returns 0 for None/empty input.
        """
        if not chunk:
            return 0
        n = len(chunk)
        cap = self.max_steps_per_chunk
        if cap and cap > 0:
            return min(n, int(cap))
        return n

    def _closest_chunk_idx(self, chunk: list[ActionTuple],
                           obs: dict) -> tuple[int, float, float]:
        """Pick the chunk index whose planned pose is closest to current obs.

        Mitigates chunk-boundary snap when the prefetch used obs that had gone
        stale (e.g. at low playback hz). Returns
        ``(idx, head_dist, picked_dist)`` for logging.
        """
        if not chunk:
            return 0, 0.0, 0.0
        try:
            if self.flavor == FLAVOR_TCP_EE:
                ref = obs.get("tcp_pose")
                if ref is None or len(ref) < 3:
                    return 0, 0.0, 0.0
                cur = np.asarray(ref[:3], dtype=np.float32)
                dists = [float(np.linalg.norm(
                    np.asarray(a[:3], dtype=np.float32) - cur))
                    for (a, _g) in chunk]
            else:
                ref = obs.get("joints")
                if not ref or len(ref) < 6:
                    return 0, 0.0, 0.0
                cur = np.asarray(ref[:6], dtype=np.float32)
                dists = [float(np.linalg.norm(
                    np.asarray(a[:6], dtype=np.float32) - cur))
                    for (a, _g) in chunk]
        except Exception:
            return 0, 0.0, 0.0
        best = int(np.argmin(dists))
        # Keep at least one action to play.
        best = min(best, len(chunk) - 1)
        return best, dists[0], dists[best]

    def next_action(self, obs: dict) -> ActionTuple | None:
        """Return the next ``(vec6, gripper)`` to publish, or None.

        Maintains a one-deep prefetch of action chunks: as soon as the current
        chunk crosses the half-way mark we kick off a background inference, then
        swap chunks when the worker finishes.
        """
        if self.policy is None:
            return None

        # Swap in a prefetched chunk if available and current is exhausted.
        if (self._chunk is None
                or self._chunk_idx >= self._effective_len(self._chunk)):
            with self._next_chunk_lock:
                nxt = self._next_chunk
                self._next_chunk = None
            if nxt is not None:
                self._chunk = nxt
                start, head_d, pick_d = self._closest_chunk_idx(nxt, obs)
                # Cap start so we still play at least one action under the
                # max_steps_per_chunk budget.
                eff = self._effective_len(nxt)
                start = min(start, max(0, eff - 1))
                if start > 0:
                    self._log("info",
                              f"chunk swap: skip {start}/{eff} stale "
                              f"actions (head_dist={head_d:.4f}, "
                              f"pick_dist={pick_d:.4f})")
                self._chunk_idx = start
                self._install_chunk(nxt, start)

        # First-ever call (or prefetch hadn't completed) → synchronous infer.
        if (self._chunk is None
                or self._chunk_idx >= self._effective_len(self._chunk)):
            chunk = self._infer_chunk(obs)
            if chunk is None:
                return None
            self._chunk = chunk
            start, head_d, pick_d = self._closest_chunk_idx(chunk, obs)
            eff = self._effective_len(chunk)
            start = min(start, max(0, eff - 1))
            if start > 0:
                self._log("info",
                          f"sync infer: skip {start}/{eff} stale "
                          f"actions (head_dist={head_d:.4f}, "
                          f"pick_dist={pick_d:.4f})")
            self._chunk_idx = start
            self._install_chunk(chunk, start)

        # Kick off background prefetch around the half-way mark of the
        # *effective* (post-cap) chunk length, so smaller caps prefetch sooner.
        eff_len = self._effective_len(self._chunk)
        if (not self._infer_busy.is_set()
                and self._chunk_idx >= eff_len // 2):
            self._infer_busy.set()
            threading.Thread(
                target=self._bg_infer, args=(obs,),
                name="Gr00t-prefetch", daemon=True,
            ).start()

        action = self._blended_action()
        self._chunk_idx += 1
        self._tick += 1
        if action is None:
            # Fall back to raw freshest action if blend yielded nothing
            # (shouldn't happen, but keeps us from publishing None when we have
            # a valid chunk loaded).
            return self._chunk[self._chunk_idx - 1]
        return action

    def _install_chunk(self, chunk: list[ActionTuple], start: int) -> None:
        """Push a freshly-installed chunk into the ensemble ring buffer.

        ``start`` is the index inside ``chunk`` that will be executed at the
        *current* tick (after stale-skip). origin_tick is therefore
        ``_tick - start`` so chunk[start] == action at tick ``_tick``.
        """
        if self.chunk_horizon is None:
            self.chunk_horizon = len(chunk)
        origin = self._tick - start
        self._ensemble.append((origin, chunk))
        # Bound to ensemble_window (>=1).
        k = max(1, int(self.ensemble_window))
        if len(self._ensemble) > k:
            self._ensemble = self._ensemble[-k:]

    def _blended_action(self) -> ActionTuple | None:
        """Weighted-average over chunks in the ensemble ring that cover the
        current tick. K=1 reduces to the freshest chunk's action.

        Pose channels (xyz + rvec or joints) blend with exponential weights.
        Gripper is taken from the freshest chunk *only* — averaging across an
        open/close transition would produce a nonsensical mid value that
        triggers neither behavior.
        """
        if not self._ensemble:
            return None
        fresh_origin = self._ensemble[-1][0]
        fresh_chunk = self._ensemble[-1][1]
        weights: list[float] = []
        vecs: list[list[float]] = []
        for origin, chunk in self._ensemble:
            idx = self._tick - origin
            if idx < 0 or idx >= len(chunk):
                continue
            age = fresh_origin - origin  # 0 for the freshest chunk
            w = math.exp(-max(0.0, self.ensemble_alpha) * age)
            vec, _g = chunk[idx]
            weights.append(w)
            vecs.append(list(vec))
        if not weights:
            return None
        total = sum(weights)
        n = len(vecs[0])
        blended_vec = [
            sum(weights[k] * vecs[k][i] for k in range(len(vecs))) / total
            for i in range(n)
        ]
        # Gripper: pass through the freshest chunk's value verbatim.
        fresh_idx = self._tick - fresh_origin
        if 0 <= fresh_idx < len(fresh_chunk):
            fresh_g = float(fresh_chunk[fresh_idx][1])
        else:
            fresh_g = float(self._ensemble[-1][1][-1][1])
        fresh_g = max(0.0, min(1.0, fresh_g))
        return (blended_vec, fresh_g)

    def reset(self) -> None:
        self._chunk = None
        self._chunk_idx = 0
        self._prev_rvec = None
        self._tick = 0
        self._ensemble = []
        with self._next_chunk_lock:
            self._next_chunk = None
