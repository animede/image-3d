import io
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("IMAGE3D_GENERATOR", "mock")


@pytest.fixture()
def tmp_data_dir(tmp_path, monkeypatch):
    """各テストで独立した data/ ディレクトリを使う(ジョブ永続化の副作用を分離)。"""
    from server import config

    data_dir = tmp_path / "data"
    jobs_dir = data_dir / "jobs"
    jobs_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "JOBS_DIR", jobs_dir)
    return data_dir


def make_test_png_bytes(size=(128, 128), color=(200, 50, 50)) -> bytes:
    """テスト用のPNG画像バイト列を生成する。"""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
