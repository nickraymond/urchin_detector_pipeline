"""Build 2, "what runs where" (S6): convert a model to 8-bit and put it through each camera's compiler.

    convert.py          the `convert` job kind and where its files live (app environment, no torch)
    convert_worker.py   the job's worker: 8-bit step, then the camera's compiler, then fit.json
    to_int8.py          runs inside a tool environment (D51): PyTorch -> ONNX -> 8-bit TFLite
    toolchain/          the tool environments' pinned requirements
"""
