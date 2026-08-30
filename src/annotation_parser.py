"""Parse CAMELYON16 ASAP annotations and measure patch overlap.

Coordinates in CAMELYON16 ASAP XML files are treated as level-0 WSI pixel
coordinates. A patch is represented as ``(x, y, width, height)`` in the same
coordinate system, where ``(x, y)`` is its upper-left corner.

This module stores polygon vertices only. It never rasterizes a WSI or creates
a full-resolution mask. Polygon/rectangle overlap is computed by clipping each
annotation polygon to the patch rectangle and measuring the clipped polygon's
area.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import openslide


Point = Tuple[float, float]
Patch = Tuple[float, float, float, float]
_EPSILON = 1e-9


@dataclass(frozen=True)
class AnnotationPolygon:
    """One ASAP annotation contour in level-0 WSI coordinates."""

    name: str
    annotation_type: str
    points: Tuple[Point, ...]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return ``(min_x, min_y, max_x, max_y)`` for this contour."""
        xs = [point[0] for point in self.points]
        ys = [point[1] for point in self.points]
        return min(xs), min(ys), max(xs), max(ys)


def _local_name(tag: str) -> str:
    """Return an XML tag name without an optional namespace."""
    return tag.rsplit("}", 1)[-1]


def _coordinate_sort_key(element: ET.Element) -> int:
    """Sort coordinate elements by ASAP's optional Order attribute."""
    try:
        return int(element.attrib.get("Order", "0"))
    except ValueError:
        return 0


def parse_annotation_xml(xml_path: Path) -> List[AnnotationPolygon]:
    """Parse all usable ASAP ``Annotation`` contours from ``xml_path``.

    Each ``Annotation`` containing at least three valid ``Coordinate``
    elements becomes one polygon. The XML ``Type`` (including ``Polygon`` or
    ``Spline``) is retained as metadata; both are interpreted as closed tumor
    contours for geometric verification.

    Raises:
        FileNotFoundError: If ``xml_path`` does not exist.
        ValueError: If the XML is malformed or contains no usable contours.
    """
    xml_path = Path(xml_path)
    if not xml_path.is_file():
        raise FileNotFoundError(f"Annotation XML not found: {xml_path}")

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Malformed annotation XML: {xml_path}") from exc

    polygons: List[AnnotationPolygon] = []
    for annotation in root.iter():
        if _local_name(annotation.tag) != "Annotation":
            continue

        coordinates = next(
            (
                child
                for child in annotation
                if _local_name(child.tag) == "Coordinates"
            ),
            None,
        )
        if coordinates is None:
            continue

        points: List[Point] = []
        coordinate_elements = sorted(
            (
                child
                for child in coordinates
                if _local_name(child.tag) == "Coordinate"
            ),
            key=_coordinate_sort_key,
        )
        for coordinate in coordinate_elements:
            try:
                points.append(
                    (float(coordinate.attrib["X"]), float(coordinate.attrib["Y"])))
            except (KeyError, TypeError, ValueError):
                continue

        if len(points) >= 3:
            polygons.append(
                AnnotationPolygon(
                    name=annotation.attrib.get("Name", ""),
                    annotation_type=annotation.attrib.get("Type", ""),
                    points=tuple(points),
                )
            )

    if not polygons:
        raise ValueError(f"No usable annotation polygons found in: {xml_path}")
    return polygons


def _signed_area(points: Sequence[Point]) -> float:
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def polygon_area(points: Sequence[Point]) -> float:
    """Return the non-negative area of a simple polygon."""
    if len(points) < 3:
        return 0.0
    return abs(_signed_area(points))


def _clip_polygon(
    polygon: Sequence[Point],
    inside,
    intersection,
) -> List[Point]:
    """Clip a polygon against one rectangle edge."""
    if not polygon:
        return []

    clipped: List[Point] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            clipped.append(intersection(previous, current))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    return clipped


def _rectangle_intersection_area(
    polygon: Sequence[Point],
    rectangle: Tuple[float, float, float, float],
) -> float:
    """Return polygon area inside an axis-aligned rectangle."""
    left, top, right, bottom = rectangle
    clipped: List[Point] = list(polygon)

    def vertical_intersection(first: Point, second: Point, x: float) -> Point:
        delta_x = second[0] - first[0]
        if abs(delta_x) <= _EPSILON:
            return x, first[1]
        ratio = (x - first[0]) / delta_x
        return x, first[1] + ratio * (second[1] - first[1])

    def horizontal_intersection(first: Point, second: Point, y: float) -> Point:
        delta_y = second[1] - first[1]
        if abs(delta_y) <= _EPSILON:
            return first[0], y
        ratio = (y - first[1]) / delta_y
        return first[0] + ratio * (second[0] - first[0]), y

    clipped = _clip_polygon(
        clipped,
        lambda point: point[0] >= left - _EPSILON,
        lambda first, second: vertical_intersection(first, second, left),
    )
    clipped = _clip_polygon(
        clipped,
        lambda point: point[0] <= right + _EPSILON,
        lambda first, second: vertical_intersection(first, second, right),
    )
    clipped = _clip_polygon(
        clipped,
        lambda point: point[1] >= top - _EPSILON,
        lambda first, second: horizontal_intersection(first, second, top),
    )
    clipped = _clip_polygon(
        clipped,
        lambda point: point[1] <= bottom + _EPSILON,
        lambda first, second: horizontal_intersection(first, second, bottom),
    )
    return polygon_area(clipped)


def patch_tumor_overlap_percentage(
    patch: Patch, polygons: Iterable[AnnotationPolygon]
) -> float:
    """Return the percentage of patch area covered by tumor contours.

    Overlap from multiple polygons is summed. CAMELYON16 contours are
    expected to represent non-overlapping lesion regions; overlapping contours
    would therefore be counted more than once and should be validated before
    using this value for labels.
    """
    x, y, width, height = patch
    if width <= 0 or height <= 0:
        raise ValueError("Patch width and height must be positive")

    rectangle = (x, y, x + width, y + height)
    patch_area = width * height
    overlap_area = 0.0
    for polygon in polygons:
        min_x, min_y, max_x, max_y = polygon.bounds
        if max_x < rectangle[0] or min_x > rectangle[2]:
            continue
        if max_y < rectangle[1] or min_y > rectangle[3]:
            continue
        overlap_area += _rectangle_intersection_area(polygon.points, rectangle)

    return min(100.0, max(0.0, overlap_area / patch_area * 100.0))


def patch_intersects_tumor(
    patch: Patch, polygons: Iterable[AnnotationPolygon]
) -> bool:
    """Return whether a patch has positive-area intersection with any contour."""
    return patch_tumor_overlap_percentage(patch, polygons) > _EPSILON


def classify_patch_overlap(
    patch: Patch,
    polygons: Iterable[AnnotationPolygon],
    tumor_threshold: float = 50.0,
) -> Optional[int]:
    """Apply the initial policy: tumor=1, normal=0, ambiguous=None."""
    overlap = patch_tumor_overlap_percentage(patch, polygons)
    if overlap >= tumor_threshold:
        return 1
    if overlap <= _EPSILON:
        return 0
    return None


def _verification_paths(base_dir: Path) -> Tuple[Path, Path]:
    """Select the first downloaded tumor WSI and its stem-matched XML."""
    tumor_dir = base_dir / "data" / "RAW" / "camelyon16" / "tumor"
    annotation_dir = base_dir / "data" / "RAW" / "camelyon16" / "annotations"
    wsi_path = sorted(tumor_dir.glob("*.tif"))[0]
    xml_path = annotation_dir / f"{wsi_path.stem}.xml"
    return wsi_path, xml_path


def verify_one_tumor_wsi(base_dir: Path) -> None:
    """Parse and report one real downloaded tumor WSI/XML pair."""
    wsi_path, xml_path = _verification_paths(base_dir)
    polygons = parse_annotation_xml(xml_path)
    with openslide.OpenSlide(str(wsi_path)) as slide:
        width, height = slide.dimensions

    all_points = [point for polygon in polygons for point in polygon.points]
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    within_wsi = (
        min_x >= 0 and min_y >= 0 and max_x <= width and max_y <= height
    )
    types = ", ".join(sorted({polygon.annotation_type or "unspecified" for polygon in polygons}))

    print(f"WSI filename                 : {wsi_path.name}")
    print(f"XML filename                 : {xml_path.name}")
    print(f"Annotation polygon count     : {len(polygons)}")
    print(f"Annotation types             : {types}")
    print(f"WSI level-0 dimensions       : {width} x {height}")
    print(f"Minimum X coordinate         : {min_x:.3f}")
    print(f"Maximum X coordinate         : {max_x:.3f}")
    print(f"Minimum Y coordinate         : {min_y:.3f}")
    print(f"Maximum Y coordinate         : {max_y:.3f}")
    print(f"Coordinates within WSI       : {'YES' if within_wsi else 'NO'}")

    if not within_wsi:
        raise ValueError("Annotation coordinates extend outside WSI level-0 dimensions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wsi",
        type=Path,
        help="Optional tumor WSI path; defaults to the first local tumor TIFF.",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        help="Optional matching ASAP XML path; defaults to the WSI stem in annotations/.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    if args.wsi or args.xml:
        if not args.wsi or not args.xml:
            parser.error("--wsi and --xml must be supplied together")
        wsi_path, xml_path = args.wsi, args.xml
        polygons = parse_annotation_xml(xml_path)
        with openslide.OpenSlide(str(wsi_path)) as slide:
            width, height = slide.dimensions
        points = [point for polygon in polygons for point in polygon.points]
        print(f"WSI filename                 : {wsi_path.name}")
        print(f"XML filename                 : {xml_path.name}")
        print(f"Annotation polygon count     : {len(polygons)}")
        print(f"WSI level-0 dimensions       : {width} x {height}")
        print(f"Minimum X coordinate         : {min(point[0] for point in points):.3f}")
        print(f"Maximum X coordinate         : {max(point[0] for point in points):.3f}")
        print(f"Minimum Y coordinate         : {min(point[1] for point in points):.3f}")
        print(f"Maximum Y coordinate         : {max(point[1] for point in points):.3f}")
        within_wsi = all(
            0 <= point[0] <= width and 0 <= point[1] <= height for point in points
        )
        print(f"Coordinates within WSI       : {'YES' if within_wsi else 'NO'}")
        return

    verify_one_tumor_wsi(base_dir)


if __name__ == "__main__":
    main()
