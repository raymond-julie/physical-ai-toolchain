# blob_sync

Uploads encoded UR recorder sessions to an Azure Blob Storage container. A session uploads only **after its camera videos finish encoding**, so partially written datasets are never sent.

## 📋 Prerequisites

- Python 3.12+
- An Azure Blob Storage container and a container-level SAS URL with write access

## 🚀 Quick Start

```bash
cd data-pipeline/capture/upload/blob_sync
cp config.example.yaml config.yaml
# edit config.yaml: set container_url to your container SAS URL
./run.sh --check    # validate config + container access
./run.sh --once     # upload all ready sessions, then exit
./run.sh            # watch mode: poll and upload as sessions complete
```

> [!NOTE]
> `config.yaml` is git-ignored because the SAS URL is a secret. Set
> `BLOB_SYNC_CONTAINER_URL` in the environment instead to keep it out of the file.

## 🧠 Readiness Logic

Each recording lives under `recordings_lerobot/session_<timestamp>/`. The recorder writes the parquet data first and encodes the camera `.mp4` videos last (using temporary `tmp*` directories). A session uploads only when:

1. `meta/info.json` exists,
2. `data/` contains a `.parquet` file,
3. `videos/` contains at least one `.mp4` (configurable), and
4. no file under the session changed for `settle_seconds` (quiescence — proves the encoder finished).

After a successful upload the session gets an `.uploaded` marker file so it is never re-sent. Local files are kept.

## ⚙️ Configuration

| Key | Default | Meaning |
|-----|---------|---------|
| `source_dir` | — | Directory holding `session_*` datasets. |
| `container_url` | — | Container-level SAS URL (or `BLOB_SYNC_CONTAINER_URL`). |
| `blob_prefix` | `""` | Virtual folder prefix inside the container. |
| `settle_seconds` | `60` | Quiescence window proving encoding finished. |
| `poll_interval_seconds` | `30` | Rescan interval in watch mode. |
| `require_videos` | `true` | Require encoded `.mp4`s before uploading. |
| `exclude_globs` | `tmp*`, `*.tmp`, `.uploaded` | Names skipped during upload. |

Blobs are written as `<blob_prefix>/<session_name>/<relative_path>`, e.g. `ur_dual_recorder/session_20260603_192318/videos/observation.images.cam_high/chunk-000/file-000.mp4`.

## 🐳 Container

```bash
docker build -t blob_sync .
docker run --rm \
  -v /abs/path/to/recordings_lerobot:/data/recordings_lerobot \
  -v "$PWD/config.yaml:/app/config.yaml:ro" \
  blob_sync --once
```

Set `source_dir: /data/recordings_lerobot` in the mounted `config.yaml`, or pass `-e BLOB_SYNC_CONTAINER_URL=...` to keep the SAS URL out of the image. `build-and-deploy.sh` builds and pushes to an Azure Container Registry (override `REGISTRY`, `IMAGE`, `TAG`).

## 🧪 Tests

```bash
cd data-pipeline/capture/upload/blob_sync
python -m pytest tests
```

Tests stub the Azure SDK, so no cloud credentials or hardware are needed.
