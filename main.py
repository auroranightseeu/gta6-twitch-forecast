"""The harness runs python3 /bot/main.py. This shim delegates to the validated round-5 model file; the round-4 exact-anchor main.py is preserved in git history at commit 0110be67."""

import importlib.util
from pathlib import Path

_MODEL_PATH = Path(__file__).resolve().with_name("huber-bias-corrected.py")
_SPEC = importlib.util.spec_from_file_location("_round5_model", _MODEL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load model: {_MODEL_PATH}")
_MODEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODEL)

forecast_case = _MODEL.forecast_case
LEVELS = _MODEL.LEVELS
main = _MODEL.main

if __name__ == "__main__":
    main()
