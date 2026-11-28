"""TRACKER S8: shrunk copies of outside images (samples._load with a cache folder). Synthetic JPEGs only."""

import os

import numpy as np
from PIL import Image

from app.train import samples


def photo(path, w=1200, h=900):
    rng = np.random.default_rng(3)
    a = (np.linspace(0, 255, w)[None, :, None] * np.ones((h, 1, 3))).astype(np.uint8)
    a[100:300, 200:400] = rng.integers(0, 255, (200, 200, 3))
    Image.fromarray(a).save(path, format="JPEG", quality=95)
    return str(path)


def opened(monkeypatch):
    seen = []
    real = Image.open

    def spy(fp, *a, **k):
        seen.append(os.path.basename(str(fp)))
        return real(fp, *a, **k)
    monkeypatch.setattr(samples.Image, "open", spy)
    return seen


def test_a_copy_is_made_once_then_read_instead_of_the_original(tmp_path, monkeypatch):
    src, cache = photo(tmp_path / "big.JPG"), tmp_path / "cache"
    direct = samples._load(src, (300, 225))
    seen = opened(monkeypatch)
    first = samples._load(src, (300, 225), str(cache))
    copies = sorted(p.name for p in cache.iterdir())
    assert seen == ["big.JPG"] and len(copies) == 1 and copies[0].endswith("_600x450.jpg")
    seen.clear()
    again = samples._load(src, (300, 225), str(cache))
    smaller = samples._load(src, (150, 112), str(cache))
    assert seen == [copies[0], copies[0]]                       # the original was not decoded again
    assert again.shape == first.shape == direct.shape == (225, 300, 3) and smaller.shape == (112, 150, 3)
    assert np.abs(again.astype(int) - direct.astype(int)).mean() < 3      # the same picture, give or take JPEG


def test_a_larger_request_reads_the_original_and_a_changed_original_gets_new_copies(tmp_path, monkeypatch):
    src, cache = photo(tmp_path / "big.JPG"), tmp_path / "cache"
    samples._load(src, (300, 225), str(cache))
    seen = opened(monkeypatch)
    assert samples._load(src, (800, 600), str(cache)).shape == (600, 800, 3)
    assert seen == ["big.JPG"]                                  # 600 x 450 is too small for it
    assert len(list(cache.iterdir())) == 1                      # 1600 px would be the original's own size: no copy
    old_key = samples.cache_key(src)
    os.utime(src, ns=(1, 1))
    assert samples.cache_key(src) != old_key and samples._cached(src, (300, 225), str(cache)) is None


def test_no_cache_folder_means_no_copies_and_sidecars_are_ignored(tmp_path):
    src, cache = photo(tmp_path / "big.JPG"), tmp_path / "cache"
    samples._load(src, (300, 225))
    assert not cache.exists()
    cache.mkdir()
    (cache / f"._{samples.cache_key(src)}_600x450.jpg").write_bytes(b"\0\x05AppleDouble")
    assert samples._cached(src, (300, 225), str(cache)) is None
    assert samples._load(src, (300, 225), str(cache)).shape == (225, 300, 3)


def test_make_uses_the_cache_named_by_the_sample(tmp_path):
    src = photo(tmp_path / "big.JPG")
    s = {"image": src, "width": 1200, "height": 900, "boxes": [[0, 210, 110, 180, 180]], "ignore": [],
         "cache": str(tmp_path / "cache")}
    samples.make(s, 128, 0, 0, 0)
    assert len(list((tmp_path / "cache").iterdir())) == 1
