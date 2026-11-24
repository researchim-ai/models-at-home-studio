"""Run processor-only VLM checks in the selected GPU environment."""
import contextlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.paths import add_runtime_path
add_runtime_path()

if __name__ == '__main__':
    config = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        if config.get('training_backend') == 'unsloth':
            import unsloth
        from backend.services.vlm_preflight import inspect
        result = inspect(config)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
