#!/usr/bin/env python3
"""Compact and simplify a rock_export.py result with exact pixel checks (CPU)."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from core import bbox, canonical_image, decode, file_sha, read_json
from PIL import Image
from scipy import ndimage as ndi
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union


def require(condition, message):
    if not condition:
        raise ValueError(message)


def polygon_parts(geometry):
    """Preserve all polygon components, including those in geometry collections."""
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        return [p for child in geometry.geoms for p in polygon_parts(child)]
    raise ValueError(f"Unexpected geometry: {geometry.geom_type}")


def validate(doc, data, derived, owner):
    """Check the serialized polygons against PNG, RLE and ownership, not contours."""
    h, w = data["imageHeight"], data["imageWidth"]
    require(owner.shape == (h, w), "Ownership dimensions mismatch")
    require(np.issubdtype(owner.dtype, np.integer), "Ownership must use integer group IDs")
    annotations = {a["group_id"]: a for a in data["annotations"]}
    require(len(annotations) == len(data["annotations"]), "Duplicate annotation group IDs")
    require(all(isinstance(gid, int) and gid > 0 for gid in annotations), "Invalid group ID")
    raster = np.zeros((h, w), dtype=owner.dtype)
    grouped = defaultdict(list)
    for shape in doc["shapes"]:
        gid = shape["group_id"]
        require(
            gid in annotations and shape["shape_type"] == "polygon", "Unknown group or shape type"
        )
        points = np.asarray(shape["points"], dtype=float)
        require(points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3, "Invalid points")
        require(
            np.isfinite(points).all() and np.equal(points, np.floor(points)).all(),
            "Expected finite integer pixel-cell edges",
        )
        require(((points >= 0) & (points <= [w, h])).all(), "Polygon outside image")
        polygon = Polygon(points)
        require(
            polygon.is_valid and not polygon.is_empty and polygon.area > 0,
            "Invalid or empty polygon",
        )
        require(not polygon.interiors, "Polygon has interior rings")
        x0, y0, x1, y1 = map(int, polygon.bounds)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        local = contains_xy(polygon, xx + 0.5, yy + 0.5)
        require(local.any(), "Polygon has no mask pixels")
        region = raster[y0:y1, x0:x1]
        require(not region[local].any(), "Polygons share pixels")
        region[local] = gid
        grouped[gid].append(polygon)
    require(set(grouped) == set(annotations), "Missing annotation polygons")
    require(np.array_equal(raster, owner), "Polygon/ownership round trip failed")
    geometries = []
    for gid, annotation in annotations.items():
        mask = raster == gid
        target = np.asarray(Image.open(derived / annotation["mask_file"])) > 0
        require(np.array_equal(mask, target), f"PNG mask mismatch: {annotation['id']}")
        require(np.array_equal(mask, decode(annotation["segmentation"])), "RLE mask mismatch")
        require(
            mask.any()
            and int(mask.sum()) == annotation["area"]
            and bbox(mask) == annotation["bbox"],
            "Mask area or box mismatch",
        )
        require(not (ndi.binary_fill_holes(mask) & (owner == 0)).any(), "Enclosed unassigned void")
        geometry = unary_union(grouped[gid])
        for other in geometries:
            require(geometry.intersection(other).area == 0, "Positive-area polygon intersection")
        geometries.append(geometry)


def vertex_count(pieces):
    return sum(len(p.exterior.coords) - 1 for p in pieces)


def raster_pieces(pieces, shape):
    mask = np.zeros(shape, dtype=bool)
    for polygon in pieces:
        x0, y0, x1, y1 = map(int, polygon.bounds)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= contains_xy(polygon, xx + 0.5, yy + 0.5)
    return mask


def simplify_groups(groups, owner, tolerance, skip=()):
    """Back off Douglas-Peucker tolerance until exact pixels and separation survive."""
    require(
        np.isfinite(tolerance) and tolerance >= 0,
        "Simplification tolerance must be finite and nonnegative",
    )
    current = {gid: unary_union(pieces) for gid, pieces in groups.items()}
    result, records = {}, []
    for gid, pieces in groups.items():
        before = vertex_count(pieces)
        accepted, used, rejections = pieces, None, []
        if tolerance > 0 and gid not in skip:
            levels = [tolerance / (2**i) for i in range(5)] + [0.0]
            for level in levels:
                candidates = [p.simplify(level, preserve_topology=True) for p in pieces]
                if not all(
                    p.geom_type == "Polygon"
                    and p.is_valid
                    and not p.is_empty
                    and p.area > 0
                    and not p.interiors
                    for p in candidates
                ):
                    rejections.append({"tolerance": level, "reason": "invalid_or_nonsolid"})
                    continue
                if vertex_count(candidates) >= before:
                    continue
                if not np.array_equal(raster_pieces(candidates, owner.shape), owner == gid):
                    rejections.append({"tolerance": level, "reason": "changes_mask_pixels"})
                    continue
                geometry = unary_union(candidates)
                if sum(p.area for p in candidates) != geometry.area:
                    rejections.append({"tolerance": level, "reason": "overlapping_components"})
                    continue
                if any(
                    geometry.intersection(other).area > 0
                    for other_gid, other in current.items()
                    if other_gid != gid
                ):
                    rejections.append({"tolerance": level, "reason": "intersects_other_rock"})
                    continue
                accepted, used = candidates, level
                current[gid] = geometry
                break
        result[gid] = accepted
        records.append(
            {
                "group_id": gid,
                "vertices_before": before,
                "vertices_after": vertex_count(accepted),
                "accepted_tolerance": used,
                "rejections": rejections,
                "rectangle_fallback": gid in skip,
            }
        )
    return result, {
        "requested_tolerance_pixels": tolerance,
        "method": "Douglas-Peucker, topology preserved; exact pixel and inter-rock geometry gates",
        "vertices_before": sum(r["vertices_before"] for r in records),
        "vertices_after": sum(r["vertices_after"] for r in records),
        "groups": records,
    }


def convert(
    derived, original, output=None, *, label="rocks", overwrite=False, simplify_tolerance=0.5
):
    """Read a derived export without modifying it; write LabelMe and a validation sidecar."""
    derived, original = Path(derived).resolve(), Path(original).resolve()
    output = Path(output).resolve() if output else original.with_suffix(".json")
    report_path = output.with_suffix(".validation.json")
    require(isinstance(label, str) and bool(label.strip()), "Label must be nonempty")
    require(output.suffix.lower() == ".json", "Output must be a JSON file")
    require(output.parent.is_dir(), "Output parent directory must exist")
    protected = {original, *(p.resolve() for p in derived.rglob("*") if p.is_file())}
    require(
        output not in protected and report_path not in protected,
        "Cannot overwrite source artifacts",
    )
    require(
        overwrite or not (output.exists() or report_path.exists()),
        "Output or validation sidecar exists; choose a new path or pass --overwrite",
    )
    data = read_json(derived / "annotations.json")
    owner = np.load(derived / "ownership.npy", allow_pickle=False)
    canonical = canonical_image(original, data.get("provenance", {}).get("orientation", "raw"))
    reference = canonical_image(derived / data["imagePath"], "raw")
    require(
        np.array_equal(np.asarray(canonical), np.asarray(reference)),
        "Original image pixels mismatch",
    )
    require(
        canonical.size == (data["imageWidth"], data["imageHeight"]), "Image dimensions mismatch"
    )
    grouped = defaultdict(list)
    for shape in data["shapes"]:
        grouped[shape["group_id"]].append(shape)
    shapes, fallback, pieces_by_group = [], [], {}
    for annotation in data["annotations"]:
        gid = annotation["group_id"]
        require(grouped[gid], "Missing source rectangle pieces")
        old = [Polygon(s["points"]) for s in grouped[gid]]
        require(all(p.is_valid and not p.is_empty for p in old), "Invalid source polygons")
        pieces = polygon_parts(unary_union(old))
        if any(p.interiors for p in pieces):
            # An enclosed neighboring rock must remain separate; do not fill across it.
            pieces = old
            fallback.append(gid)
        pieces_by_group[gid] = pieces
    require(set(grouped) == set(pieces_by_group), "Unaccounted source polygon group")
    pieces_by_group, simplification = simplify_groups(
        pieces_by_group, owner, simplify_tolerance, fallback
    )
    for annotation in data["annotations"]:
        gid = annotation["group_id"]
        for polygon in pieces_by_group[gid]:
            shapes.append(
                {
                    "label": label,
                    "points": [list(p) for p in list(polygon.exterior.coords)[:-1]],
                    "group_id": gid,
                    "description": annotation["id"] + "; native pixels; model proposal",
                    "shape_type": "polygon",
                    "flags": {},
                }
            )
    doc = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.relpath(original, output.parent),
        "imageData": None,
        "imageHeight": data["imageHeight"],
        "imageWidth": data["imageWidth"],
    }
    # Validate the JSON representation, including fallback pieces, before writing any output.
    serialized = json.dumps(doc, indent=2, allow_nan=False) + "\n"
    validate(json.loads(serialized), data, derived, owner)
    with output.open("w" if overwrite else "x") as stream:
        stream.write(serialized)
    validate(read_json(output), data, derived, owner)
    report = {
        "image": str(original),
        "labelme": str(output),
        "label": label,
        "derived_export": str(derived),
        "instances": len(data["annotations"]),
        "polygon_pieces": len(shapes),
        "original_rectangle_pieces": len(data["shapes"]),
        "rectangle_fallback_groups": fallback,
        "simplification": simplification,
        "mask_roundtrip": "exact at pixel centers",
        "shared_pixels": 0,
        "positive_area_intersections": 0,
        "interior_rings": 0,
        "enclosed_unassigned_voids": 0,
        "status": data["status"],
        "source_annotations_sha256": file_sha(derived / "annotations.json"),
        "source_ownership_sha256": file_sha(derived / "ownership.npy"),
        "original_image_sha256": file_sha(original),
        "labelme_sha256": file_sha(output),
        "converter_sha256": file_sha(Path(__file__)),
    }
    with report_path.open("w" if overwrite else "x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("derived", type=Path, help="Final rock_export.py output directory")
    parser.add_argument(
        "original", type=Path, help="Unchanged source image to reference in LabelMe"
    )
    parser.add_argument("--out", type=Path, help="Defaults to ORIGINAL with a .json extension")
    parser.add_argument("--label", default="rocks", help="LabelMe label (default: rocks)")
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=0.5,
        help="Maximum simplification tolerance in native pixels (default: 0.5; 0 disables); exact mask agreement required",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output and validation sidecar"
    )
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                convert(
                    args.derived,
                    args.original,
                    args.out,
                    label=args.label,
                    overwrite=args.overwrite,
                    simplify_tolerance=args.simplify_tolerance,
                ),
                indent=2,
            )
        )
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
