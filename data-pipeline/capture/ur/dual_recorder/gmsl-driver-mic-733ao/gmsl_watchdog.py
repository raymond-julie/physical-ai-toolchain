#!/usr/bin/env python3
"""Per-camera GMSL watchdog for the MIC-733-AO carrier.

Goal: when a single GMSL camera drops (the classic
``Device response with bad magic, magic=0x0`` link fault), reconnect *just that
camera* so the other three keep streaming. Only fall back to the full driver
reload (``reconnect.sh``, which resets every camera) as a last resort.

How it works
------------
1. Detection (no device contention): poll the UrCameraStreamer HTTP snapshots
   per serial and hash them. A live camera's frame changes; a dropped camera
   serves a frozen last frame (HTTP 200 but identical bytes). A camera whose
   snapshot is unchanged for ``--stale`` seconds is considered down.

2. Recovery, tiered (least disruptive first):
     Tier 1  surgical: unbind/bind only that camera's g300 I2C addresses via
             /sys/bus/i2c/drivers/g300/{unbind,bind}. Re-trains the one GMSL
             link without touching the other cameras.
     Tier 2  last resort: run reconnect.sh (rmmod/insmod g300 + max9296 +
             max96712) which resets ALL cameras on the bus. Used only after
             ``--max-surgical-fails`` consecutive surgical attempts fail.

Bridge from serial to kernel I2C devices
----------------------------------------
The Orbbec SDK reports each camera's GMSL port as its ``uid`` (``gmsl2-N``).
On this MIC-733-AO the cameras enumerate behind a pca9546 mux (i2c-9 / i2c-10),
each branch holding a max9296 deserializer (``*-0048``) and two cameras, each
camera exposing four g300 virtual addresses (``g2m0..g2m3``). The CAMERAS map
below pins each serial to its GMSL port and its four g300 I2C ids. Confirm/auto-
fill it with ``--derive`` (see below) before trusting Tier 1.

Run as root on the HOST (needs sysfs writes + rmmod/insmod):

    sudo ./gmsl_watchdog.py                 # monitor + auto-recover
    sudo ./gmsl_watchdog.py --recover cam_low   # one-shot surgical recover
    sudo ./gmsl_watchdog.py --derive            # confirm serial->i2c mapping
    sudo ./gmsl_watchdog.py --list              # show config and exit
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
RECONNECT_SH = SCRIPT_DIR / "reconnect.sh"

G300_DRIVER = Path("/sys/bus/i2c/drivers/g300")


@dataclass
class Camera:
    """A single GMSL camera and the kernel handles needed to reset it.

    Attributes
    ----------
    name:      friendly name (matches config_v3 device_id, e.g. ``cam_low``).
    serial:    Orbbec serial; also the streamer's snapshot id.
    gmsl_port: SDK ``uid`` (``gmsl2-N``); identifies the physical GMSL link.
    i2c_ids:   the four g300 I2C device ids (e.g. ``10-006a``..``10-006d``)
               that the surgical unbind/bind targets. Empty until derived.
    """

    name: str
    serial: str
    gmsl_port: str
    i2c_ids: list[str] = field(default_factory=list)


# Camera map for THIS rig. serial -> GMSL port is verified (from the Orbbec SDK
# ``uid``). The i2c_ids are the surgical-reset targets; verify/auto-fill with
# ``--derive`` before relying on Tier 1. Branch i2c@0 == i2c-9, i2c@1 == i2c-10;
# each camera owns one contiguous group of four g300 addresses.
CAMERAS: dict[str, Camera] = {
    "cam_high": Camera(
        name="cam_high", serial="CV3H4600001E", gmsl_port="gmsl2-3",
        i2c_ids=["10-0066", "10-0067", "10-0068", "10-0069"],
    ),
    "cam_low": Camera(
        name="cam_low", serial="CV3H46000031", gmsl_port="gmsl2-4",
        i2c_ids=["10-006a", "10-006b", "10-006c", "10-006d"],
    ),
    "cam_right_wrist": Camera(
        name="cam_right_wrist", serial="CV34361000HP", gmsl_port="gmsl2-1",
        i2c_ids=["9-0066", "9-0067", "9-0068", "9-0069"],
    ),
    "cam_left_wrist": Camera(
        name="cam_left_wrist", serial="CV34361000J3", gmsl_port="gmsl2-2",
        i2c_ids=["9-006a", "9-006b", "9-006c", "9-006d"],
    ),
}

# All g300 i2c groups, used by --derive to try candidates against a down camera.
ALL_I2C_GROUPS: list[list[str]] = [
    ["9-0066", "9-0067", "9-0068", "9-0069"],
    ["9-006a", "9-006b", "9-006c", "9-006d"],
    ["10-0066", "10-0067", "10-0068", "10-0069"],
    ["10-006a", "10-006b", "10-006c", "10-006d"],
]


def snapshot_hash(base_url: str, serial: str, timeout: float = 5.0) -> str | None:
    """Return an MD5 of the camera's current snapshot, or None on error."""
    url = f"{base_url.rstrip('/')}/snapshot/{serial}?t={time.time_ns()}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gmsl-watchdog"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return hashlib.md5(resp.read()).hexdigest()
    except Exception as exc:  # noqa: BLE001
        _LOGGER.debug("snapshot %s failed: %s", serial, exc)
        return None


def sdk_serials(container: str = "ur-camera-streamer") -> list[str] | None:
    """Ground-truth: serials the Orbbec SDK currently enumerates in the streamer.

    Returns None if the query can't run (e.g. container not present). Used by
    --derive to confirm a camera actually came back at the bus level.
    """
    code = (
        "from pyorbbecsdk import Context;"
        "dl=Context().query_devices();"
        "print('\\n'.join(dl.get_device_by_index(i).get_device_info()"
        ".get_serial_number() for i in range(dl.get_count())))"
    )
    try:
        out = subprocess.run(
            ["docker", "exec", container, "python3", "-c", code],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.debug("sdk_serials failed: %s", exc)
        return None
    if out.returncode != 0:
        _LOGGER.debug("sdk_serials rc=%s err=%s", out.returncode, out.stderr.strip())
        return None
    return [s.strip() for s in out.stdout.splitlines() if s.strip()]


def _write_sysfs(path: Path, value: str) -> bool:
    try:
        path.write_text(value)
        return True
    except Exception as exc:  # noqa: BLE001
        _LOGGER.debug("write %s <- %s failed: %s", path, value, exc)
        return False


def g300_rebind(i2c_ids: list[str], settle: float = 1.5, dry_run: bool = False) -> bool:
    """Unbind then bind a camera's g300 I2C addresses (surgical, single-camera).

    Returns True if every address was re-bound. Does NOT touch the other
    cameras' addresses, so siblings on the same deserializer keep streaming.
    """
    if not i2c_ids:
        _LOGGER.error("no i2c_ids to rebind (mapping not derived?)")
        return False
    unbind = G300_DRIVER / "unbind"
    bind = G300_DRIVER / "bind"
    if not unbind.exists() or not bind.exists():
        _LOGGER.error("g300 driver sysfs not found at %s (driver loaded?)", G300_DRIVER)
        return False

    _LOGGER.info("surgical rebind of %s", " ".join(i2c_ids))
    if dry_run:
        _LOGGER.info("[dry-run] would unbind/bind %s", " ".join(i2c_ids))
        return True

    # Unbind in reverse so the primary (g2m0) address goes last/first cleanly.
    for dev in reversed(i2c_ids):
        if (G300_DRIVER / dev).exists():
            _write_sysfs(unbind, dev)
    time.sleep(settle)
    ok = True
    for dev in i2c_ids:
        if not _write_sysfs(bind, dev):
            ok = False
    time.sleep(settle)
    if ok:
        _LOGGER.info("rebind complete for %s", " ".join(i2c_ids))
    else:
        _LOGGER.warning("rebind incomplete for %s", " ".join(i2c_ids))
    return ok


def full_reload(dry_run: bool = False) -> bool:
    """Last resort: reconnect.sh reloads g300/max9296/max96712 (resets ALL cams)."""
    if not RECONNECT_SH.exists():
        _LOGGER.error("reconnect.sh not found at %s", RECONNECT_SH)
        return False
    _LOGGER.warning("FULL driver reload via %s (all cameras reset)", RECONNECT_SH.name)
    if dry_run:
        _LOGGER.info("[dry-run] would run %s", RECONNECT_SH)
        return True
    try:
        subprocess.run(["bash", str(RECONNECT_SH)], check=True, timeout=120)
        return True
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error("full reload failed: %s", exc)
        return False


def recover_camera(cam: Camera, max_surgical_fails: int, surgical_fails: int,
                   dry_run: bool = False) -> tuple[bool, int]:
    """Attempt to recover one camera. Returns (recovered, new_surgical_fail_count).

    Tier 1 (surgical) is tried first. After ``max_surgical_fails`` consecutive
    surgical failures, escalate to Tier 2 (full reload) and reset the counter.
    """
    if surgical_fails < max_surgical_fails:
        ok = g300_rebind(cam.i2c_ids, dry_run=dry_run)
        if ok:
            return True, 0
        return False, surgical_fails + 1

    _LOGGER.warning(
        "camera '%s' failed %d surgical attempts; escalating to full reload",
        cam.name, surgical_fails,
    )
    ok = full_reload(dry_run=dry_run)
    return ok, 0


def monitor(base_url: str, interval: float, stale: float, cooldown: float,
            max_surgical_fails: int, dry_run: bool) -> None:
    """Poll snapshot freshness per camera and recover any that go stale."""
    last_hash: dict[str, str | None] = {}
    last_change: dict[str, float] = {}
    last_recover: dict[str, float] = {}
    surgical_fails: dict[str, int] = {n: 0 for n in CAMERAS}
    now = time.monotonic()
    for cam in CAMERAS.values():
        last_change[cam.name] = now
        last_recover[cam.name] = 0.0

    _LOGGER.info("watchdog started: %d cameras, stale=%.0fs interval=%.1fs%s",
                 len(CAMERAS), stale, interval, " [dry-run]" if dry_run else "")
    while True:
        now = time.monotonic()
        for cam in CAMERAS.values():
            h = snapshot_hash(base_url, cam.serial)
            if h is None:
                # Treat an unreachable snapshot like a frozen one (no fresh frame).
                pass
            elif h != last_hash.get(cam.name):
                last_hash[cam.name] = h
                last_change[cam.name] = now
                if surgical_fails[cam.name]:
                    _LOGGER.info("camera '%s' is live again", cam.name)
                surgical_fails[cam.name] = 0
                continue

            age = now - last_change[cam.name]
            if age < stale:
                continue
            if now - last_recover[cam.name] < cooldown:
                continue

            _LOGGER.warning("camera '%s' (%s) stale %.1fs -> recover",
                            cam.name, cam.serial, age)
            recovered, surgical_fails[cam.name] = recover_camera(
                cam, max_surgical_fails, surgical_fails[cam.name], dry_run,
            )
            last_recover[cam.name] = now
            # Reset the freshness clock so we give the link time to retrain
            # before judging it stale again.
            last_change[cam.name] = now
            if recovered:
                _LOGGER.info("recovery issued for '%s'; awaiting fresh frames", cam.name)
        time.sleep(interval)


def derive(base_url: str, dry_run: bool) -> None:
    """Confirm/auto-fill the serial->i2c mapping using a currently-down camera.

    Finds a camera the SDK no longer enumerates (dropped off the bus), then
    tries each unassigned i2c group in turn, checking after each whether the
    missing serial reappears in the SDK device list. Safe: it only rebinds the
    one group, observes, and reports which group restored the down camera.
    """
    serials_present = sdk_serials()
    if serials_present is None:
        _LOGGER.error("cannot query SDK (is the streamer container running?)")
        return
    down = [c for c in CAMERAS.values() if c.serial not in serials_present]
    if not down:
        _LOGGER.info("all cameras present; nothing to derive. Current map:")
        for c in CAMERAS.values():
            _LOGGER.info("  %-16s %s %s -> %s", c.name, c.serial, c.gmsl_port,
                         " ".join(c.i2c_ids))
        return

    target = down[0]
    _LOGGER.info("down camera: %s (%s, %s). Trying i2c groups to identify its link.",
                 target.name, target.serial, target.gmsl_port)
    for group in ALL_I2C_GROUPS:
        _LOGGER.info("trying group %s", " ".join(group))
        if not g300_rebind(group, dry_run=dry_run):
            continue
        time.sleep(3.0)
        present = sdk_serials() or []
        if target.serial in present:
            _LOGGER.info("MATCH: %s (%s) recovered via i2c group %s",
                         target.name, target.serial, " ".join(group))
            _LOGGER.info("Set CAMERAS['%s'].i2c_ids = %s", target.name, group)
            return
    _LOGGER.warning("no single group recovered %s; a full reload may be required",
                    target.name)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000",
                   help="UrCameraStreamer base URL (default %(default)s)")
    p.add_argument("--interval", type=float, default=2.0,
                   help="poll period seconds (default %(default)s)")
    p.add_argument("--stale", type=float, default=8.0,
                   help="seconds of frozen frames before recovery (default %(default)s)")
    p.add_argument("--cooldown", type=float, default=30.0,
                   help="min seconds between recovery attempts per camera")
    p.add_argument("--max-surgical-fails", type=int, default=3,
                   help="surgical attempts before full reload (default %(default)s)")
    p.add_argument("--recover", metavar="NAME",
                   help="one-shot: surgically recover this camera and exit")
    p.add_argument("--derive", action="store_true",
                   help="confirm serial->i2c mapping using a down camera and exit")
    p.add_argument("--list", action="store_true",
                   help="print the camera map and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="log actions without touching the hardware")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )

    if args.list:
        for c in CAMERAS.values():
            print(f"{c.name:16s} serial={c.serial} {c.gmsl_port} "
                  f"i2c={' '.join(c.i2c_ids) or '(unset)'}")
        return 0

    if args.derive:
        derive(args.base_url, args.dry_run)
        return 0

    if args.recover:
        cam = CAMERAS.get(args.recover)
        if cam is None:
            # Allow recovery by serial too.
            cam = next((c for c in CAMERAS.values() if c.serial == args.recover), None)
        if cam is None:
            _LOGGER.error("unknown camera '%s' (known: %s)",
                          args.recover, ", ".join(CAMERAS))
            return 2
        ok = g300_rebind(cam.i2c_ids, dry_run=args.dry_run)
        return 0 if ok else 1

    try:
        monitor(args.base_url, args.interval, args.stale, args.cooldown,
                args.max_surgical_fails, args.dry_run)
    except KeyboardInterrupt:
        _LOGGER.info("watchdog stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
