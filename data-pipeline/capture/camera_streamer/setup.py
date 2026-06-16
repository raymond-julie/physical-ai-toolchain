from setuptools import find_packages, setup

setup(
    name="camera_streamer",
    version="0.1.0",
    description="Host connected Orbbec cameras as shareable MJPEG links on the LAN.",
    packages=find_packages(),
    include_package_data=True,
    package_data={"camera_streamer": ["templates/*.html", "static/*"]},
    python_requires=">=3.10",
    install_requires=[
        "flask>=2.0",
        "opencv-python-headless>=4.5",
        "numpy>=1.21",
        "PyYAML>=5.4",
    ],
    entry_points={
        "console_scripts": [
            "camera-streamer=camera_streamer.__main__:main",
        ],
    },
)
