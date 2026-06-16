"""Setup script for editable install.

We do not require installation — the launcher adds the source dir to
PYTHONPATH and invokes the nodes via ``python -m``. This file is
provided so ``pip install -e .`` also works for developers who prefer
that workflow.
"""

from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="episode_recorder",
    version="0.2.0",
    description="Vendor-agnostic multi-robot LeRobot dataset recorder.",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "episode_recorder": [
            "web/templates/*.html",
            "web/static/*",
        ],
    },
    python_requires=">=3.10",
    # Runtime deps live in requirements.txt / install_dependencies.sh;
    # listing them here would force pip to resolve ROS-shipped wheels.
    install_requires=[],
    entry_points={
        "console_scripts": [
            "robot_reader=episode_recorder.nodes.robot_reader:main",
            "episode_recorder=episode_recorder.nodes.episode_recorder:main",
            "trigger_tool_io=episode_recorder.nodes.trigger_tool_io:main",
            "trigger_gui=episode_recorder.nodes.trigger_gui:main",
        ],
    },
)
