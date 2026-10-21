"""The six projections LMM-PCQA scores, rendered in memory as the authors render them.

The recipe is MM-PCQA's ``utils/get_projections.py`` (Open3D's legacy visualiser, a hidden
1080 × 1920 window, the camera turned before each capture, the white background cropped away)
with the six-face turns the authors give in GMS-3DQA's README::

    turns = (0, 0), (90°, 0), (90°, 0), (90°, 0), (0, 90°), (0, 180°),

each applied to the view the previous one left, at 5.82 pixels of drag per degree. A view is
the RGB image the release's ``cv2.imwrite`` stores and its ``PIL`` reader reads back; PNG is
lossless, so the round trip through files is left out.
"""

import cv2
import numpy as np
import open3d as o3d

# MM-PCQA's window, width by height, and its drag per degree of turn.
_WIDTH, _HEIGHT = 1080, 1920
_PIXELS_PER_DEGREE = 5.82
# GMS-3DQA's six-face turns in degrees: three quarter turns about the vertical, then two
# vertical turns.
TURNS_DEG = ((0, 0), (90, 0), (90, 0), (90, 0), (0, 90), (0, 180))
# The background's grey level, and the colour range of a captured buffer.
_WHITE = 255
_BUFFER_SCALE = 255


def projections(points: np.ndarray, colours: np.ndarray) -> list[np.ndarray]:
    """The six cropped views of a cloud, RGB ``uint8``, in the authors' order.

    Args:
        points: Coordinates, ``N × 3``.
        colours: Colours in 0..1, ``N × 3``.

    Raises:
        ValueError: If a view shows no point, as the release's crop then finds no bounds.
    """
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(colours)
    visualiser = o3d.visualization.Visualizer()
    visualiser.create_window(visible=False, width=_WIDTH, height=_HEIGHT)
    visualiser.add_geometry(cloud)
    control = visualiser.get_view_control()
    views = []
    for turn_x, turn_y in TURNS_DEG:
        control.rotate(turn_x * _PIXELS_PER_DEGREE, turn_y * _PIXELS_PER_DEGREE)
        visualiser.poll_events()
        visualiser.update_renderer()
        buffer = np.asarray(visualiser.capture_screen_float_buffer(True))
        bgr = cv2.cvtColor((buffer * _BUFFER_SCALE).astype(np.uint8), cv2.COLOR_RGB2BGR)
        views.append(np.ascontiguousarray(background_crop(bgr)[..., ::-1]))
    visualiser.destroy_window()
    return views


def background_crop(image: np.ndarray) -> np.ndarray:
    """MM-PCQA's crop of a BGR view to the rows and columns whose mean grey is not white.

    The release searches the end of each range from index 0 backwards (``row[-0]`` is the
    first row), so a view not white in its first row or column keeps its last ones.

    Args:
        image: The captured view, BGR ``uint8``.

    Raises:
        ValueError: If every row is white.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    col_a, col_b = _bounds(np.mean(grey, axis=0))
    row_a, row_b = _bounds(np.mean(grey, axis=1))
    cropped = image[row_a:row_b, col_a:col_b, :]
    return cropped


def _bounds(profile: np.ndarray) -> tuple[int, int]:
    """The release's start and end of the non-white range of a mean-grey profile.

    Args:
        profile: Mean grey of every row or column.

    Raises:
        ValueError: If the whole profile is white.
    """
    shown = [i for i in range(len(profile)) if profile[i] != _WHITE]
    if not shown:
        raise ValueError("the view shows no point")
    ends = [len(profile) - i for i in range(len(profile)) if profile[-i] != _WHITE]
    bounds = (shown[0], ends[0])
    return bounds
