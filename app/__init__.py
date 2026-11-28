"""The urchin_detector_pipeline app.

Lead call (lead 2, 2026-11-28; QE 11's card 9 check): ONNX Runtime's official builds send telemetry to Microsoft by
default (its own Privacy.md). `ORT_DISABLE_TELEMETRY=1`, set before ONNX Runtime starts, stops the uploader, its events
and its device id. Every app process imports this package first (the web app, every `python -m app.…` worker and
tool step), and every child process inherits the switch from here. Data stays on the laptop (SPEC, hard constraints).
"""

import os

ORT_OFF = {"ORT_DISABLE_TELEMETRY": "1"}    # also named in every env the app builds for a child
os.environ.update(ORT_OFF)
