"""CLI entrypoint for blob_sync.

Examples:
    python -m blob_sync --config config.yaml          # watch (default)
    python -m blob_sync --config config.yaml --once   # one-shot
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import BlobSyncConfigError, load_config
from .sync import sync_once, watch
from .uploader import BlobUploader

_LOGGER = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blob_sync",
        description="Sync encoded UR recorder sessions to Azure Blob Storage.",
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Path to the YAML config file (default: ./config.yaml).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Scan once, upload all ready sessions, then exit.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate config and container access, then exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested blob_sync mode."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        cfg = load_config(args.config)
    except BlobSyncConfigError as exc:
        _LOGGER.error("%s", exc)
        return 2

    uploader = BlobUploader(cfg.container_url, cfg.blob_prefix)

    if args.check:
        try:
            uploader.check_access()
        except Exception as exc:
            _LOGGER.error("Container access check failed: %s", exc)
            return 1
        _LOGGER.info("Container access OK: %s", cfg.container_url_redacted)
        return 0

    if args.once:
        uploaded = sync_once(cfg, uploader)
        _LOGGER.info("Done. Uploaded %d session(s).", uploaded)
        return 0

    try:
        watch(cfg, uploader)
    except KeyboardInterrupt:
        _LOGGER.info("Interrupted; exiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
