"""The in-memory projections against the authors' recipe: MM-PCQA's ``get_projections.py`` with
the six-face turns of GMS-3DQA's README, run on the same cloud written to a PLY file."""

import runpy
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from PIL import Image

from lmm_pcqa_projections import TURNS_DEG, projections

_RECIPE = Path(__file__).resolve().parent.parent / "MM-PCQA" / "utils" / "get_projections.py"
# The two lines of the recipe GMS-3DQA's README replaces to render six faces instead of four.
_FOUR_FACES = (
    "rotate_para = [[0,0],[90*interval,0],[90*interval,0],[90*interval,0]]\n"
    "    for i in range(4):"
)
_SIX_FACES = (
    "rotate_para = [[0,0],[90*interval,0],[90*interval,0],[90*interval,0],[0,90*interval],"
    "[0,180*interval]]\n    for i in range(6):"
)
_POINTS = 20_000
_BYTE = 255
_SPREAD = (1.0, 0.5, 0.25)


@pytest.fixture
def cloud() -> tuple[np.ndarray, np.ndarray]:
    """A seeded ellipsoidal cloud with colours on the byte grid, which a PLY file keeps."""
    rng = np.random.default_rng(2026)
    points = rng.normal(size=(_POINTS, 3)) * _SPREAD
    colours = rng.integers(0, _BYTE + 1, size=(_POINTS, 3)) / _BYTE
    cloud = (points, colours)
    return cloud


@pytest.mark.skipif(not _RECIPE.is_file(), reason="needs the MM-PCQA fork beside this one")
def test_projections_equal_the_authors_recipe(
    cloud: tuple[np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    """Every view equals, bit for bit, the PNG file the recipe writes for the same cloud."""
    points, colours = cloud
    recipe = _RECIPE.read_text()
    assert recipe.count(_FOUR_FACES) == 1
    script = tmp_path / _RECIPE.name
    script.write_text(recipe.replace(_FOUR_FACES, _SIX_FACES))
    ply = tmp_path / "cloud.ply"
    written = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    written.colors = o3d.utility.Vector3dVector(colours)
    o3d.io.write_point_cloud(ply, written)
    runpy.run_path(str(script))["camera_rotation"]("ply", str(ply), str(tmp_path))
    views = projections(points, colours)
    files = [Image.open(tmp_path / f"{i}.png").convert("RGB") for i in range(len(TURNS_DEG))]
    assert all(np.array_equal(view, np.asarray(file)) for view, file in zip(views, files))
