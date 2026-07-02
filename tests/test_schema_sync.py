"""Guard against drift between the two copies of the vision wire schema.

zero/vision/schemas.py (Pi side) and server/vision/shared/schemas.py (GPU side)
are mirrored by hand with a "keep in sync" comment — this test makes the sync
check mechanical: same classes, same field names per class.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("Detection", "AnalyzeRequest", "SceneFact", "AnalyzeResponse")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wire_schemas_have_same_fields():
    edge = _load(ROOT / "zero" / "vision" / "schemas.py", "edge_schemas")
    server = _load(ROOT / "server" / "vision" / "shared" / "schemas.py", "server_schemas")
    for cls_name in CLASSES:
        edge_cls = getattr(edge, cls_name, None)
        server_cls = getattr(server, cls_name, None)
        assert edge_cls is not None, f"{cls_name} missing on the Pi side"
        assert server_cls is not None, f"{cls_name} missing on the server side"
        assert set(edge_cls.model_fields) == set(server_cls.model_fields), (
            f"{cls_name} fields drifted between zero/vision/schemas.py and "
            f"server/vision/shared/schemas.py"
        )
