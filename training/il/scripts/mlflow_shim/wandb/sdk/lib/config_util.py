"""Stub of ``wandb.sdk.lib.config_util`` providing only ``ConfigError``.

HuggingFace's ``transformers.integrations.integration_utils.WandbCallback``
imports this symbol unconditionally to use as an ``except`` clause. Our
in-shim ``_Config`` never raises ``ConfigError``, so this stub merely needs
to expose the name as an ``Exception`` subclass for the import to succeed.
"""

from __future__ import annotations


class ConfigError(Exception):
    """Stand-in for wandb's ConfigError; never raised by the shim's _Config."""
