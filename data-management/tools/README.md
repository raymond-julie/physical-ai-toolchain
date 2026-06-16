# Data Management Tools

CLI tools for dataset operations in the Physical AI Toolchain.

## 📦 Available Tools

| Tool                                                | Purpose                                                      |
|-----------------------------------------------------|--------------------------------------------------------------|
| `blob_path_validator.py`                            | Validate blob paths against storage naming conventions       |
| `convert_lerobot_v3_to_v2.py`                       | Convert a LeRobot v3.0 dataset to the GR00T-flavored v2.0     |
| `merge_lerobot_sessions.py`                         | Merge multiple `session_*` LeRobot v2.1 folders into one      |
| [`lerobot-converter/`](lerobot-converter/README.md) | Convert ROS 2 MCAP recordings to LeRobot v2.1 + inspect CLIs  |

The converter and merge tools have standalone runtime dependencies. Install them
per directory:

```bash
python -m pip install -r requirements.txt                   # convert/merge
python -m pip install -r lerobot-converter/requirements.txt # MCAP converter
```

### Convert a LeRobot v3.0 dataset to v2.0

`convert_lerobot_v3_to_v2.py` splits sharded v3.0 parquet/mp4 files into the
per-episode v2.0 layout GR00T expects. The `meta/modality.json` defaults to a
UR10e single-arm 7-DoF, 2-camera embodiment; pass `--modality-config PATH` to
supply a JSON spec for a different embodiment.

```bash
python convert_lerobot_v3_to_v2.py --src /data/run-v3 --dst /data/run-v2
python convert_lerobot_v3_to_v2.py --src /data/run-v3 --dst /data/run-v2 \
    --modality-config dual_arm_modality.json
```

### Merge LeRobot v2.1 sessions

`merge_lerobot_sessions.py` combines `session_*` folders into one dataset,
re-indexing episodes and frames globally and rebuilding `meta/`:

```bash
python merge_lerobot_sessions.py ./downloads ./combined-sessions
```

## 🧪 Tests

Behavior tests for the tools live in [tests](tests) and run from the repository
root without datasets or ffmpeg:

```bash
uv run pytest data-management/tools/tests -v
```

## 📋 Planned Tools

| Tool       | Purpose                                                                 |
|------------|-------------------------------------------------------------------------|
| `filter`   | Select episodes matching criteria (task, success rate, metadata fields) |
| `split`    | Partition datasets into train/validation/test sets                      |
| `validate` | Check dataset integrity, schema compliance, and completeness            |

See [../specifications/dataset-curation.specification.md](../specifications/dataset-curation.specification.md)
for the full Filter, Split, Merge, Convert, and Validate contracts.


> [!NOTE]
> Replace all example IP placeholders (for example, 192.168.1.x) with the actual robot IP addresses for your environment before running.
