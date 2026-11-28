"""Build 2, S9: evaluate a trained model on a test set (DESIGN §Planned for S9, D90-D94).

    run.py      the `evaluate` job kind: what can be evaluated, the test set and its hash, where results go
    worker.py   the job's worker: each test frame at the model's eval scale, in tiles, merged across seams
    yolox.py    the YOLOX predictor (best.pt's EMA weights, inference only, in the app's environment)

The tiling itself is app/train/tiles.py, shared with the trainer's validation (S8 Fixes 2).
"""
