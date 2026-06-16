#!/usr/bin/env bash
# Full GMSL driver reload: reset every camera on the MIC-733-AO bus by reloading
# g300 + obc_max9296 + obc_max96712. Last-resort recovery used by gmsl_watchdog.py
# only after surgical per-camera rebinds fail.
#
# `set -e` is intentionally omitted: rmmod tolerates a partially loaded module
# stack (an already-unloaded module returns non-zero), and aborting mid-reload
# would leave the bus down. -u/-o pipefail are safe (no unset vars, no pipes).
set -uo pipefail

if [ -e /sys/module/g300 ]; then
	echo "Files g300 has loaded,unload g300"
	sudo rmmod g300
	sudo rmmod obc_max9296
	sudo rmmod obc_max96712
	echo "reconnect g300"
	sudo insmod /lib/modules/5.15.148-tegra/updates/drivers/media/i2c/obc_max9296.ko
	sudo insmod /lib/modules/5.15.148-tegra/updates/drivers/media/i2c/obc_max96712.ko
	sudo insmod /lib/modules/5.15.148-tegra/updates/drivers/media/i2c/g300.ko
else
	echo "connect g300"
	sudo insmod /lib/modules/5.15.148-tegra/updates/drivers/media/i2c/obc_max9296.ko
	sudo insmod /lib/modules/5.15.148-tegra/updates/drivers/media/i2c/obc_max96712.ko
	sudo insmod /lib/modules/5.15.148-tegra/updates/drivers/media/i2c/g300.ko
fi
