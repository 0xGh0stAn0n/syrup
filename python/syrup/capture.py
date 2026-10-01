"""Live frames from a window on screen (Windows only), for sessions.

    session = syrup.ops.track_moving_regions.session()
    for frame in syrup.capture.window("Notepad"):
        print(session(frame))
"""

from . import Image, _call, _native
from .errors import InputError


def windows():
    """Titles of the windows that can be captured; empty off Windows."""
    return _call(_native.list_windows)


def window(title, *, frames=None):
    """Frames of the first window whose title contains `title`, as
    syrup.Image, until it closes or `frames` have been taken. Raises
    InputError if no window matches, and DependencyError off Windows."""
    taken = 0
    while frames is None or taken < frames:
        try:
            data, width, height, channels = _call(_native.capture_window, title)
        except InputError:
            if taken == 0:
                raise
            return
        yield Image(data, width, height, channels)
        taken += 1
