"""Control ROS 2 bag recording (start/stop) via a supervised subprocess."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from collections.abc import Sequence
from datetime import datetime

_LOGGER = logging.getLogger(__name__)


class RosBagRecorder:
    """Manage a ``ros2 bag record`` subprocess with clean start/stop control."""

    def __init__(self, output_dir: str = "./recordings_raw", bag_name: str | None = None) -> None:
        self.output_dir = output_dir
        self.bag_name = bag_name
        self.process: subprocess.Popen[bytes] | None = None
        self.is_recording = False
        os.makedirs(self.output_dir, exist_ok=True)

    def start_recording(
        self,
        topics: Sequence[str] | None = None,
        all_topics: bool = False,
        compression: str | None = None,
    ) -> bool:
        """Start a recording. Returns True when the subprocess launched."""
        if self.is_recording:
            _LOGGER.warning("Recording is already in progress")
            return False

        if self.bag_name:
            bag_file = self.bag_name
        else:
            bag_file = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        bag_path = os.path.join(self.output_dir, bag_file)

        cmd = ["ros2", "bag", "record"]
        if all_topics:
            cmd.append("-a")
        elif topics:
            cmd.extend(topics)
        else:
            _LOGGER.error("Must specify topics or set all_topics=True")
            return False
        if compression:
            cmd.extend(["--compression-mode", "file", "--compression-format", compression])
        cmd.extend(["-o", bag_path])

        try:
            # New process group so SIGINT can be delivered to the whole tree.
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError:
            _LOGGER.error("ros2 bag command not found. Is ROS 2 installed and sourced?")
            return False
        except OSError as exc:
            _LOGGER.error("Error starting recording: %s", exc)
            return False

        self.is_recording = True
        _LOGGER.info("Started recording to %s (cmd: %s)", bag_path, " ".join(cmd))
        return True

    def stop_recording(self) -> bool:
        """Stop the active recording. Returns True when it stopped cleanly."""
        if not self.is_recording or self.process is None:
            _LOGGER.warning("No recording in progress")
            return False

        try:
            # SIGINT to the process group mirrors a Ctrl+C on the terminal.
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _LOGGER.warning("Process did not stop gracefully, forcing termination")
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=2)
        except (OSError, subprocess.SubprocessError) as exc:
            _LOGGER.error("Error stopping recording: %s", exc)
            return False

        self.is_recording = False
        self.process = None
        _LOGGER.info("Recording stopped successfully")
        return True

    def is_recording_active(self) -> bool:
        return self.is_recording

    def __del__(self) -> None:
        if self.is_recording:
            _LOGGER.info("Stopping recording on cleanup")
            self.stop_recording()
