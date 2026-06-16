#!/usr/bin/env python3
"""Hold open a Nova motion-group session on each controller so the
controller-state stream keeps publishing on NATS.

Nova v2 only publishes `nova.v2.cells.<cell>.controllers.<ctrl>.state`
while at least one client has a motion-group session open against the
controller. The episode recorder is a *passive* NATS subscriber — it
does NOT open a session itself — so without something like this script
(or the Nova web UI) running, the recorder sees no joint state and
every episode is discarded as "0 frames".

Usage:
    pip install --user wandelbots-nova
    # Adjust NOVA_HOST / NOVA_ACCESS_TOKEN as needed (or use ~/.nova).
    python3 keep_nova_session_alive.py
"""

from __future__ import annotations

import asyncio
import os
import signal

from nova import Nova

CELL = os.environ.get("NOVA_CELL", "cell")
CONTROLLERS = os.environ.get("NOVA_CONTROLLERS", "ur5-left,ur-right").split(",")


async def _hold(controller_id: str) -> None:
    """Acquire one motion-group on `controller_id` and idle forever."""
    async with Nova() as nova:  # reads ~/.nova / env for host + token
        cell = nova.cell(CELL)
        ctrl = await cell.controller(controller_id)
        # Take the first motion-group on this controller and keep the
        # async-context-manager open. That alone is enough to make Nova
        # start publishing controller state on NATS.
        mg = (await ctrl.activated_motion_groups())[0]
        async with mg:
            print(f"[{controller_id}] motion-group session active: {mg.motion_group_id}")
            while True:
                await asyncio.sleep(3600)


async def main() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    tasks = [asyncio.create_task(_hold(c.strip())) for c in CONTROLLERS if c.strip()]
    await stop.wait()
    for t in tasks:
        t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
