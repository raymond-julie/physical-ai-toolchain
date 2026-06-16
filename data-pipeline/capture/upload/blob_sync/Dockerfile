# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY blob_sync ./blob_sync

# Mount the recordings dir and provide config at runtime:
#   docker run --rm \
#     -v /path/to/recordings_lerobot:/data/recordings_lerobot \
#     -v /path/to/config.yaml:/app/config.yaml:ro \
#     blob_sync
#
# Or pass the SAS URL via env instead of baking it into the image:
#   -e BLOB_SYNC_CONTAINER_URL="https://...?<sas>"
ENTRYPOINT ["python", "-m", "blob_sync", "--config", "/app/config.yaml"]
