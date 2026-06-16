# Camera Stream URCap

A PolyScope 5 (e-Series) URCap that shows the **live camera feed on the teach
pendant**. It consumes the MJPEG streams published by
[UrCameraStreamer](../../camera_streamer/README.md) and renders them in an
Installation screen (and a header-bar toolbar popup), so an operator can watch
any camera without leaving PolyScope.

> [!NOTE]
> This is the **only Java / Maven / OSGi** component in the data-capture
> pipeline. It is built separately with the UR URCap SDK (see Prerequisites) and
> is not part of the Python `data-pipeline` package. The `com.trainmybot.*`
> package name is retained verbatim as the documented bundle identifier.

## How it works

- The URCap registers a single Installation node under
  `Installation → URCaps → Camera Stream`, plus a toolbar button in the
  PolyScope header bar that opens the same live feed from any screen.
- The view embeds a Swing panel that opens the streamer's
  `multipart/x-mixed-replace` endpoint, slices each JPEG frame out of the HTTP
  body (by `0xFFD8` / `0xFFD9` markers), decodes it, and paints it live.
- The connection auto-reconnects with backoff if the streamer restarts or the
  network drops.
- The configured Streamer URL and Camera id are persisted in the robot
  installation, so they survive restarts.

The stream URL is built as `http://<host>:<port>/stream/<cameraId>`. Leave the
camera id blank to use the URL field verbatim (handy for pasting a full stream
URL).

## Project layout

```text
urcap/
├── pom.xml                                  # Maven aggregator / parent
├── build.sh                                 # Convenience build wrapper
└── com.trainmybot.camerastream.impl/
    ├── pom.xml                              # OSGi bundle + urcap packaging
    └── src/main/java/com/trainmybot/camerastream/impl/
        ├── Activator.java                                       # OSGi entry point
        ├── CameraStreamInstallationNodeService.java            # Node service
        ├── CameraStreamInstallationNodeContribution.java       # Persisted config + logic
        ├── CameraStreamInstallationNodeView.java               # Swing UI
        ├── CameraStreamToolbarService.java                     # Header-bar button
        ├── CameraStreamToolbarContribution.java                # Toolbar popup
        └── MjpegStreamPanel.java                               # MJPEG reader + renderer
```

## Prerequisites

Building a URCap needs JDK 8, Maven 3, and the UR URCap SDK (it provides the
`com.ur.urcap:api` artifact, which is not on Maven Central).

Install the toolchain:

```bash
sudo apt-get install -y openjdk-8-jdk maven
```

Get the SDK from Universal Robots (signed in):
`Download Center → PolyScope 5 SW → URCap SDK` (e.g. `sdk-1.18.0.zip`). Unpack
it; it contains an `artifacts/api/<version>/` folder with the API jars and an
`install.sh` that registers them into your local `~/.m2`.

Two ways to make the API artifact resolvable:

- Point the build at the SDK (recommended — `build.sh` seeds `~/.m2` for you):

  ```bash
  URCAP_SDK_DIR=/path/to/unpacked/sdk ./build.sh
  ```

- Or install one API version manually (matches `<api.version>` in
  [pom.xml](pom.xml), default `1.3.0`):

  ```bash
  mvn install:install-file \
    -Dfile=/path/to/sdk/artifacts/api/1.3.0/com.ur.urcap.api-1.3.0.jar \
    -DgroupId=com.ur.urcap -DartifactId=api -Dversion=1.3.0 -Dpackaging=jar
  ```

> [!NOTE]
> The project uses API `1.3.0` on purpose: it is the oldest version exposing
> everything the URCap needs, which gives the widest PolyScope 5.x
> compatibility — it installs on `5.25` and earlier e-Series controllers. Bump
> `<api.version>` only if you start using newer API features.

## Build

```bash
cd urcap
./build.sh                       # if the api artifact is already in ~/.m2
# or
URCAP_SDK_DIR=/path/to/sdk ./build.sh
# -> com.trainmybot.camerastream.impl/target/com.trainmybot.camerastream.impl-1.0.0.urcap
```

`build.sh` forces JDK 8 (URCaps must target Java 8 even if the system default is
newer), optionally seeds the SDK artifacts, then runs `mvn clean install`. The
`.urcap` is the OSGi bundle jar (built by `maven-bundle-plugin`) copied to the
`.urcap` extension.

## Install on the robot

1. Copy the generated `*.urcap` file to a USB stick.
2. On the pendant: `Settings → System → URCaps → +`, select the file.
3. Restart PolyScope when prompted.
4. Open `Installation → URCaps → Camera Stream`.

## Use

1. Make sure UrCameraStreamer is running and reachable from the robot
   controller's network (default port `8000`).
2. Tap Streamer URL and enter the streamer host, e.g.
   `http://192.168.1.20:8000`.
3. Tap Camera id and enter a camera serial (e.g. `CV3H4600001E`). Leave it blank
   if the URL field already contains a full `/stream/<id>` URL.
4. Tap Apply. The live feed appears; Status shows the connection state. Use
   Reconnect to force a fresh connection.

To list available camera ids, open the streamer dashboard at
`http://<host>:8000/` or query `http://<host>:8000/api/cameras`.

## Notes

- The pendant and the streamer host must be on the same reachable network. From
  the controller, `http://127.0.0.1` will not work unless the streamer runs on
  the controller itself.
- One physical camera cannot be opened by two processes at once, but many
  viewers can share one stream — the URCap is just another MJPEG viewer, so it
  adds no load on the camera device.
- There is no authentication on the streamer; run it on a trusted network only.
