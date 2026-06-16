"""Optional editable install: ``pip install -e .``"""

from setuptools import find_packages, setup

setup(
    name="ur_dual_recorder",
    version="0.1.0",
    description="Dual leader/follower UR teleoperation recorder",
    packages=find_packages(include=["ur_dual_recorder", "ur_dual_recorder.*"]),
    include_package_data=True,
    package_data={"ur_dual_recorder.web": ["templates/*.html", "static/*"]},
    python_requires=">=3.10",
    install_requires=[
        "pyyaml",
        "numpy",
        "opencv-python",
        "ur_rtde",
        "flask",
    ],
    entry_points={
        "console_scripts": [
            "ur-dual-recorder=ur_dual_recorder.__main__:main",
        ],
    },
)
