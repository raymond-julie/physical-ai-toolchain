# GMSL driver + per-camera watchdog (MIC-733-AO)

Kernel modules and helpers for the Orbbec G300 / Gemini GMSL cameras on the
Advantech MIC-733-AO carrier, plus a watchdog that recovers a **single** dropped
camera without disturbing the others. This is host-side tooling for the
[`../`](../README.md) UrDualRecorder rig.

> [!WARNING]
> The kernel build artifacts (`*.ko` modules and the `*.dtbo` device-tree
> overlay) are platform-specific, built on the target host against its exact
> kernel (`5.15.148-tegra`), and are intentionally **not committed** to this
> repository. Build them on the target, then run `copy_to_target.sh` to install
> them. Only the host scripts and the watchdog are tracked here.

## Files

| File                 | Purpose                                                                                                                              |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `gmsl_watchdog.py`   | Per-camera watchdog: detects a dropped camera and reconnects **just that one**, falling back to `reconnect.sh` only as a last resort. |
| `reconnect.sh`       | **Full** driver reload (`rmmod`/`insmod` of `g300` + `obc_max9296` + `obc_max96712`). Resets **every** camera on the bus.             |
| `copy_to_target.sh`  | Install the modules/overlay into `/lib/modules/$(uname -r)` and enable the overlay. Run once on the target after a kernel update.     |
| `*.ko`, `*.dtbo`     | Prebuilt kernel modules + device-tree overlay (built on the target host; not committed — see warning above).                          |

## Why a per-camera watchdog

A GMSL link can drop mid-run (`Device response with bad magic, magic=0x0`). When
that happens the Orbbec SDK can no longer recover that camera in software, and
the camera streamer keeps serving its last cached frame — the feed looks "up"
(HTTP 200) but is frozen. `reconnect.sh` fixes it but resets all four cameras.

`gmsl_watchdog.py` instead re-trains only the failed link, so the other cameras
keep streaming.

## How recovery works

The cameras enumerate behind a `pca9546` I2C mux (`i2c-9` / `i2c-10`); each
branch has a `max9296` deserializer and two cameras, and each camera exposes
four `g300` virtual I2C addresses (`g2m0..g2m3`). A single camera is reset by
unbinding/binding just its four addresses on the `g300` driver:

```bash
echo 10-006a > /sys/bus/i2c/drivers/g300/unbind   # ... for each of the 4 ids
echo 10-006a > /sys/bus/i2c/drivers/g300/bind
```

The bridge from a camera to its addresses is: **serial → GMSL port (`gmsl2-N`,
the SDK `uid`) → four `g300` I2C ids**. That map lives in the `CAMERAS` dict at
the top of `gmsl_watchdog.py`.

Recovery is tiered (least disruptive first):

1. Surgical — unbind/bind only the failed camera's four `g300` addresses.
2. Full reload — `reconnect.sh` (resets all cameras), only after
   `--max-surgical-fails` consecutive surgical attempts fail.

Detection uses the streamer's HTTP snapshots (no device contention): a camera
whose snapshot bytes are unchanged for `--stale` seconds is treated as down.

## Usage

Run as **root on the host** (sysfs writes + `rmmod`/`insmod`):

```bash
sudo ./gmsl_watchdog.py --list            # show the serial -> i2c map
sudo ./gmsl_watchdog.py --derive          # confirm the map using a down camera
sudo ./gmsl_watchdog.py --recover cam_low # one-shot surgical recover, then exit
sudo ./gmsl_watchdog.py                    # monitor + auto-recover (foreground)
sudo ./gmsl_watchdog.py --dry-run          # log actions without touching hardware
```

Key flags: `--base-url` (streamer, default `http://127.0.0.1:8000`), `--stale`
(frozen seconds before recovery, default 8), `--interval`, `--cooldown`,
`--max-surgical-fails`.

### Confirm the I2C mapping first

The serial → `gmsl2-N` mapping is read from the SDK and is reliable; the four
`g300` ids per camera should be confirmed once with `--derive`. With a camera
currently down, `--derive` rebinds each candidate group and checks whether the
missing serial reappears in the SDK device list, then prints the correct
`i2c_ids` for that camera. Update the `CAMERAS` dict if it differs.

## Run as a service (optional)

```ini
# /etc/systemd/system/gmsl-watchdog.service
[Unit]
Description=GMSL per-camera watchdog
After=docker.service
Wants=docker.service

[Service]
ExecStart=/usr/bin/python3 /opt/ur-dual-recorder/gmsl-driver-mic-733ao/gmsl_watchdog.py
Restart=always
User=root

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now gmsl-watchdog.service
```

Set `ExecStart` to wherever this directory is deployed on the target host.

## Limitation

After the kernel link is restored, the streamer must reopen the recovered
camera. Its capture thread retries on a stall, but a thread wedged inside a
native SDK call may not pick the camera back up without a streamer restart.
For fully unattended recovery, add a bounded timeout around the streamer's
device open/reopen so a wedged thread self-heals — see
[`../../../camera_streamer/`](../../../camera_streamer/README.md).
