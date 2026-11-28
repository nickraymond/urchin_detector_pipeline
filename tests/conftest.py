"""Lead call (lead 2, 2026-11-28): no test may start ONNX Runtime with its telemetry on. ONNX Runtime's official
builds send telemetry to Microsoft by default (its Privacy.md); `ORT_DISABLE_TELEMETRY=1` before it starts stops it.
Set here, before any test module is imported, as well as in `app/__init__.py`."""

import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
