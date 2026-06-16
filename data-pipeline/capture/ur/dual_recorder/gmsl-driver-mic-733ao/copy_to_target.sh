#!/usr/bin/env bash
# Install the prebuilt GMSL kernel modules + device-tree overlay into
# /lib/modules/$(uname -r) and enable the Orbbec camera overlay. Run once on the
# target host after building the modules there, or after a kernel update.
#
# The referenced *.ko / *.dtbo are platform-specific build artifacts produced on
# the target and are intentionally not committed (see README.md). `set -e` makes
# the installer fail fast if any module is missing or a copy fails.
set -euo pipefail

sudo cp tegra234-camera-g300-overlay.dtbo /boot/

## copy tegra-camera.ko file ##
tegra_camera_dir=/lib/modules/$(uname -r)/updates/drivers/media/platform/tegra/camera
if [ ! -f $tegra_camera_dir/tegra-camera.ko.orig ];then
    echo "bakckup tegra-camera.ko"
    sudo cp $tegra_camera_dir/tegra-camera.ko $tegra_camera_dir/tegra-camera.ko.orig
fi
sudo cp tegra-camera.ko $tegra_camera_dir

# enable metadata:
## copy videodev.ko file ##
videodev_dir=/lib/modules/$(uname -r)/kernel/drivers/media/v4l2-core
if [ ! -f $videodev_dir/videodev.ko.orig ];then
    echo "bakckup videodev.ko"
    sudo cp $videodev_dir/videodev.ko $videodev_dir/videodev.ko.orig
fi
sudo cp videodev.ko $videodev_dir

## copy capture-ivc.ko file ##
capture_ivc_dir=/lib/modules/$(uname -r)/updates/drivers/platform/tegra/rtcpu
if [ ! -f $capture_ivc_dir/capture-ivc.ko.orig ];then
    echo "bakckup capture-ivc.ko"
    sudo cp $capture_ivc_dir/capture-ivc.ko $capture_ivc_dir/capture-ivc.ko.orig
fi
sudo cp capture-ivc.ko $capture_ivc_dir

## copy nvhost-nvcsi-t194.ko file ##
nvhost_nvcsi_t194_dir=/lib/modules/$(uname -r)/updates/drivers/video/tegra/host/nvcsi/
if [ ! -f $nvhost_nvcsi_t194_dir/nvhost-nvcsi-t194.ko.orig ];then
    echo "bakckup nvhost-nvcsi-t194.ko"
    sudo cp $nvhost_nvcsi_t194_dir/nvhost-nvcsi-t194.ko $nvhost_nvcsi_t194_dir/nvhost-nvcsi-t194.ko.orig
fi
sudo cp nvhost-nvcsi-t194.ko $nvhost_nvcsi_t194_dir

sudo cp obc_max9296.ko /lib/modules/$(uname -r)/updates/drivers/media/i2c/
sudo cp obc_max96712.ko /lib/modules/$(uname -r)/updates/drivers/media/i2c/
sudo cp g300.ko /lib/modules/$(uname -r)/updates/drivers/media/i2c/

sudo /opt/nvidia/jetson-io/config-by-hardware.py -n 2="Jetson Orbbec Camera G335Lg"

sudo depmod -a
