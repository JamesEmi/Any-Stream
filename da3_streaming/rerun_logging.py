"""
Rerun visualisation logging for any_streaming_rt.

Confines the rerun dependency to this module, so importers of any_streaming_rt
do not need it installed. When rerun is missing, build_logger() returns a
NullRerunLogger implementing the same interface, so a mistyped call is an
AttributeError at the call site rather than silently dropped output.
"""

import sys

try:
    import rerun as rr

    HAS_RERUN = True
except ImportError:
    rr = None
    HAS_RERUN = False

import numpy as np


class RerunLogger:
    """Logs to a live rerun recording."""

    @staticmethod
    def add_cli_args(parser):
        """Add rerun's --rr-* / --save flags. Must run before parse_args()."""
        rr.script_add_args(parser)

    @staticmethod
    def setup(args, app_name: str):
        rr.script_setup(args, app_name)

    def view_coordinates_rdf(self, path: str):
        rr.log(path, rr.ViewCoordinates.RDF, static=True)

    def set_time(self, timeline: str, sequence: int):
        rr.set_time(timeline, sequence=int(sequence))

    def pointcloud(self, path: str, positions, colors):
        rr.log(path, rr.Points3D(positions=positions, colors=colors))

    def trajectory(self, path: str, positions, color, static: bool = False,
                   line: bool = True, radius: float = None):
        """Log positions as uniformly-coloured points plus a connecting strip.

        The strip goes to f"{path}_line" and is skipped for fewer than 2 points.
        Set line=False for marker-only entities; radius adds per-point radii.
        """
        positions = np.asarray(positions)
        n = len(positions)

        kwargs = {"positions": positions,
                  "colors": np.full((n, 3), color, dtype=np.uint8)}
        if radius is not None:
            kwargs["radii"] = np.full(n, radius, dtype=np.float32)
        rr.log(path, rr.Points3D(**kwargs), static=static)

        if line and n >= 2:
            rr.log(
                f"{path}_line",
                rr.LineStrips3D([positions], colors=[list(color)]),
                static=static,
            )

    def image(self, path: str, img):
        rr.log(path, rr.Image(img))

    def depth_image(self, path: str, arr):
        rr.log(path, rr.DepthImage(np.asarray(arr, dtype=np.float32)))

    def scalar(self, path: str, value: float):
        rr.log(path, rr.Scalars(float(value)))

    def camera_transform(self, path: str, translation, mat3x3):
        rr.log(path, rr.Transform3D(translation=translation, mat3x3=mat3x3))

    def pinhole(self, path: str, image_from_camera, width: int, height: int,
                image_plane_distance: float = 1.0):
        rr.log(
            path,
            rr.Pinhole(
                image_from_camera=np.asarray(image_from_camera, dtype=np.float32),
                height=int(height),
                width=int(width),
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=image_plane_distance,
            ),
        )


class NullRerunLogger(RerunLogger):
    """No-op implementation used when rerun is not installed."""

    @staticmethod
    def add_cli_args(parser):
        pass

    @staticmethod
    def setup(args, app_name: str):
        pass

    def view_coordinates_rdf(self, path: str):
        pass

    def set_time(self, timeline: str, sequence: int):
        pass

    def pointcloud(self, path: str, positions, colors):
        pass

    def trajectory(self, path: str, positions, color, static: bool = False,
                   line: bool = True, radius: float = None):
        pass

    def image(self, path: str, img):
        pass

    def depth_image(self, path: str, arr):
        pass

    def scalar(self, path: str, value: float):
        pass

    def camera_transform(self, path: str, translation, mat3x3):
        pass

    def pinhole(self, path: str, image_from_camera, width: int, height: int,
                image_plane_distance: float = 1.0):
        pass


def build_logger() -> RerunLogger:
    """Return a live logger if rerun is importable, else a no-op one."""
    if HAS_RERUN:
        return RerunLogger()
    print(
        "[rerun_logging] rerun is not installed; visualisation logging is "
        "disabled. Install with: pip install rerun-sdk",
        file=sys.stderr,
    )
    return NullRerunLogger()
