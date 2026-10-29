#!/usr/bin/env python3
"""Export reviewed rock identities using exact, hole-free rectangle polygons."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
from core import bbox, canonical_image, decode, digest, encode, file_sha, read_json, write_json
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from segment import relation_flags

POLICY = {
    "version": 1,
    "ownership": "greatest interior distance; lexical target ID breaks ties",
    "polygons": "vertically coalesced horizontal pixel-run rectangles",
    "coordinates": "pixel cell edges; rasterize at pixel centers",
    "holes": "fill source-union holes only outside other source support; then fill only unassigned enclosed pixels",
}


def rectangles(mask):
    """Return disjoint (x0,x1,y0,y1) rectangles, including single-pixel parts."""
    active, result = {}, []
    for y in range(mask.shape[0] + 1):
        row = mask[y] if y < mask.shape[0] else np.zeros(mask.shape[1], bool)
        change = np.diff(np.r_[False, row, False].astype(np.int8))
        runs = set(
            zip(
                np.flatnonzero(change == 1).tolist(),
                np.flatnonzero(change == -1).tolist(),
                strict=False,
            )
        )
        for run in list(active):
            if run not in runs:
                result.append((*run, active.pop(run), y))
        for run in sorted(runs):
            active.setdefault(run, y)
    return result


def decisions_for(doc, ids, image_sha, source_sha):
    if doc.get("schema") != "rock-derived-decisions/1":
        raise ValueError("Expected rock-derived-decisions/1")
    if doc.get("image_sha256") != image_sha or doc.get("source_proposals_sha256") != source_sha:
        raise ValueError("Decision image/source hash mismatch")
    if not str(doc.get("reviewer", "")).strip():
        raise ValueError("Reviewer identity is required")
    merged, excluded, reviewed = {}, {}, {}
    for decision in doc.get("decisions", []):
        action, sources = decision.get("action"), decision.get("source_ids", [])
        if action not in {"merge", "exclude", "keep", "uncertain"}:
            raise ValueError("Unsupported decision action")
        if (
            not isinstance(sources, list)
            or not sources
            or not str(decision.get("reason", "")).strip()
        ):
            raise ValueError("Every decision needs source_ids and a reason")
        target = decision.get("target_id")
        if action == "merge" and target not in ids:
            raise ValueError("Unknown merge target")
        for source in sources:
            if source not in ids or source in reviewed:
                raise ValueError("Unknown or repeated decision source")
            reviewed[source] = decision
            if action == "merge":
                if source == target:
                    raise ValueError("Self-merge")
                merged[source] = target
            elif action == "exclude":
                excluded[source] = decision["reason"]
    if any(t in merged or t in excluded for t in merged.values()):
        raise ValueError("Merge targets must survive; normalize chains first")
    dispositions = doc.get("empty_dispositions", {})
    for source, decision in dispositions.items():
        if (
            source not in ids
            or decision.get("outcome") not in {"duplicate", "erroneous", "unresolved"}
            or not str(decision.get("reason", "")).strip()
        ):
            raise ValueError("Invalid empty disposition")
        if decision.get("outcome") == "duplicate" and (
            decision.get("target_id") not in ids or decision["target_id"] == source
        ):
            raise ValueError("Empty duplicate needs another existing target")
    return merged, excluded, reviewed, dispositions


def run(image_path, proposals_path, decisions_path, out, orientation="raw"):
    out = Path(out)
    if out.exists():
        raise ValueError("Output must be a new directory")
    image = canonical_image(image_path, orientation)
    with tempfile.TemporaryDirectory() as t:
        canonical = Path(t) / "original.png"
        image.save(canonical)
        image_sha = file_sha(canonical)
    payload = read_json(proposals_path)
    if payload.get("canonical_sha256", payload.get("image_sha256")) != image_sha:
        raise ValueError("Proposals do not match canonical image/orientation")
    source_sha, objects = file_sha(proposals_path), payload["masks"]
    ids = [o["id"] for o in objects]
    if not ids or len(ids) != len(set(ids)) or not all(isinstance(i, str) for i in ids):
        raise ValueError("Source IDs must be unique nonempty strings")
    if any(o.get("label", "rock") != "rock" for o in objects):
        raise ValueError("Separate non-rock labels first")
    h, w = image.height, image.width
    original = {}
    for obj in objects:
        m = decode(obj["rle"])
        if (
            m.shape != (h, w)
            or not m.any()
            or obj.get("area") != int(m.sum())
            or obj.get("bbox") != bbox(m)
        ):
            raise ValueError("Source mask dimensions, area or bbox mismatch")
        original[obj["id"]] = m
    doc = read_json(decisions_path)
    merged, excluded, reviewed, dispositions = decisions_for(doc, set(ids), image_sha, source_sha)
    targets = sorted(set(ids) - set(merged) - set(excluded))
    numbers = {t: n for n, t in enumerate(targets, 1)}
    groups = {t: [i for i in ids if merged.get(i, i) == t and i not in excluded] for t in targets}
    source_support = np.zeros((h, w), bool)
    for source in ids:
        if source not in excluded:
            source_support |= original[source]
    owner = np.zeros((h, w), np.int32)
    best = np.zeros((h, w), float)
    cover = np.zeros((h, w), np.int32)
    edits = {
        "union_hole_fill_pixels": {},
        "initial_source_hole_fill_pixels": {},
        "merge_created_hole_fill_pixels": {},
        "post_ownership_hole_fill_pixels": {},
    }
    for target in targets:
        m = np.logical_or.reduce([original[i] for i in groups[target]])
        filled = m | (ndi.binary_fill_holes(m) & ~source_support)
        added = filled & ~m
        baseline = (
            np.logical_or.reduce([ndi.binary_fill_holes(original[i]) for i in groups[target]])
            if len(groups[target]) > 1
            else filled
        )
        edits["initial_source_hole_fill_pixels"][target] = int((added & baseline).sum())
        edits["merge_created_hole_fill_pixels"][target] = int((added & ~baseline).sum())
        edits["union_hole_fill_pixels"][target] = int(added.sum())
        cover += filled
        x, y, bw, bh = bbox(filled)
        region = np.s_[y : y + bh, x : x + bw]
        local = filled[region]
        d = ndi.distance_transform_edt(np.pad(local, 1))[1:-1, 1:-1]
        win = local & (d > best[region])
        owner[region][win] = numbers[target]
        best[region][win] = d[win]
    for target in targets:
        add = ndi.binary_fill_holes(owner == numbers[target]) & (owner == 0)
        owner[add] = numbers[target]
        edits["post_ownership_hole_fill_pixels"][target] = int(add.sum())
    edits["contested_pixels"] = int((cover > 1).sum())
    # Review flags always use original overlapping proposals, not the final ownership map.
    flag_objects = [dict(o, label="rock") for o in objects]
    flag_ids = [
        o["id"]
        for o, f in zip(objects, relation_flags(flag_objects, None), strict=False)
        if "inside_larger_same_label_proposal" in f
    ]
    pending = [i for i in flag_ids if i not in reviewed and i not in merged.values()]
    areas = {t: int((owner == numbers[t]).sum()) for t in targets}
    ledger = []
    for source in ids:
        target = merged.get(source, source)
        number = numbers.get(target, 0)
        area = areas.get(target, 0)
        status = (
            "excluded"
            if source in excluded
            else "merged"
            if source in merged
            else "retained"
            if area
            else "emptied_by_ownership"
        )
        row = {
            "source_id": source,
            "status": status,
            "target_id": None if source in excluded else target,
            "source_area": int(original[source].sum()),
            "final_target_area": area,
            "source_score": next(o.get("score") for o in objects if o["id"] == source),
            "source_rle_sha256": digest(next(o["rle"] for o in objects if o["id"] == source)),
            "source_pixels_retained_in_target": int((original[source] & (owner == number)).sum())
            if number
            else 0,
            "review_decision": reviewed.get(source),
        }
        if not area and source not in excluded:
            winners, counts = np.unique(owner[original[source]], return_counts=True)
            row["winning_owners"] = {
                targets[int(n) - 1]: int(count)
                for n, count in zip(winners, counts, strict=False)
                if n
            }
            row["empty_disposition"] = dispositions.get(source)
            if source not in dispositions or dispositions[source]["outcome"] == "unresolved":
                pending.append(source)
        ledger.append(row)
    for source, disposition in dispositions.items():
        if next(r for r in ledger if r["source_id"] == source)["status"] != "emptied_by_ownership":
            raise ValueError("Empty dispositions must describe actually emptied targets")
        if disposition["outcome"] == "duplicate" and not areas.get(disposition["target_id"], 0):
            raise ValueError("Empty duplicate target must survive")
    pending = [
        i for i in pending if i not in dispositions or dispositions[i]["outcome"] == "unresolved"
    ]
    pending += [i for i, d in reviewed.items() if d["action"] == "uncertain"]
    pending = sorted(set(pending))
    state = (
        "review_draft"
        if pending or doc.get("touching_regions_reviewed") is not True
        else "reviewed_model_proposals"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".rock-export-", dir=out.parent))
    try:
        image.save(stage / "original.png")
        (stage / "masks").mkdir()
        shapes, annotations, labels = [], [], []
        polygon_owner = np.zeros_like(owner)
        rgb = np.array(image)
        preview = rgb.copy()
        for target in targets:
            number = numbers[target]
            m = owner == number
            if not m.any():
                continue
            if (ndi.binary_fill_holes(m) & (owner == 0)).any():
                raise ValueError("Enclosed unassigned void remains")
            pieces = rectangles(m)
            raster = np.zeros_like(m)
            for x0, x1, y0, y1 in pieces:
                if x0 >= x1 or y0 >= y1 or polygon_owner[y0:y1, x0:x1].any():
                    raise ValueError("Polygon intersection")
                polygon_owner[y0:y1, x0:x1] = number
                raster[y0:y1, x0:x1] = True
                shapes.append(
                    {
                        "label": "rock",
                        "group_id": number,
                        "shape_type": "polygon",
                        "flags": {},
                        "description": target + "; pixel cell edges",
                        "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                    }
                )
            if not np.array_equal(raster, m):
                raise ValueError("Polygon round trip failed")
            encoded = encode(m)
            if not np.array_equal(decode(encoded), m):
                raise ValueError("RLE round trip failed")
            mask_path = stage / "masks" / f"{number:06}.png"
            Image.fromarray(m.astype(np.uint8) * 255).save(mask_path)
            if not np.array_equal(np.array(Image.open(mask_path)) > 0, m):
                raise ValueError("PNG round trip failed")
            annotations.append(
                {
                    "id": target,
                    "group_id": number,
                    "source_ids": groups[target],
                    "mask_file": f"masks/{number:06}.png",
                    "area": int(m.sum()),
                    "bbox": bbox(m),
                    "segmentation": encoded,
                    "polygon_pieces": len(pieces),
                    "source_scores": {
                        i: next(o.get("score") for o in objects if o["id"] == i)
                        for i in groups[target]
                    },
                    "status": "proposal",
                }
            )
            color = np.random.default_rng(int(digest(target)[:8], 16)).integers(50, 256, 3)
            preview[m] = (rgb[m] * 0.57 + color * 0.43).astype(np.uint8)
            preview[m & ~ndi.binary_erosion(m)] = color
            x, y, bw, bh = bbox(m)
            d = ndi.distance_transform_edt(np.pad(m[y : y + bh, x : x + bw], 1))
            cy, cx = np.unravel_index(d.argmax(), d.shape)
            labels.append((x + int(cx) - 1, y + int(cy) - 1, target))
        if not np.array_equal(polygon_owner, owner):
            raise ValueError("Ownership round trip failed")
        im = Image.fromarray(preview)
        draw = ImageDraw.Draw(im)
        for x, y, target in labels:
            draw.text(
                (x, y),
                target.removeprefix("mask-").lstrip("0") or "0",
                fill="white",
                stroke_width=1,
                stroke_fill="black",
                anchor="mm",
            )
        im.save(stage / "preview.png")
        checks = {
            "input_proposals": len(ids),
            "surviving_instances": len(annotations),
            "polygon_pieces": len(shapes),
            "shared_pixels": 0,
            "positive_area_polygon_intersections": 0,
            "interior_rings": 0,
            "enclosed_unassigned_voids": 0,
            "polygon_mask_roundtrip": "exact",
            "png_rle_roundtrip": "exact",
            "nested_flagged_source_ids": flag_ids,
            "unresolved_review_ids": pending,
            "status": state,
        }
        provenance = {
            "image_sha256": image_sha,
            "source_proposals_sha256": source_sha,
            "decisions_sha256": file_sha(decisions_path),
            "exporter_sha256": file_sha(Path(__file__)),
            "policy": POLICY,
            "policy_sha256": digest(POLICY),
            "orientation": orientation,
            "model": payload.get("model", payload.get("selection_runtime")),
            "note": "Rectangle pieces share group_id; counts are unaccepted proposals.",
        }
        result = {
            "version": "5.5.0",
            "flags": {},
            "imagePath": "original.png",
            "imageData": None,
            "imageWidth": w,
            "imageHeight": h,
            "shapes": shapes,
            "annotations": annotations,
            "status": state,
            "provenance": provenance,
            "validation": checks,
        }
        write_json(stage / "annotations.json", result)
        write_json(stage / "ledger.json", ledger)
        write_json(stage / "validation.json", checks)
        write_json(stage / "geometry-edits.json", edits)
        write_json(stage / "decisions.json", doc)
        if len(read_json(stage / "annotations.json")["annotations"]) != len(annotations):
            raise ValueError("Saved annotation mismatch")
        np.save(stage / "ownership.npy", owner, allow_pickle=False)
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return checks


def main():
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument(
        "--proposals",
        type=Path,
        required=True,
        help="Full-workflow proposals.json or direct sam.py result JSON",
    )
    p.add_argument("--decisions", type=Path, help="Hash-bound rock-derived-decisions/1 file")
    p.add_argument("--out", type=Path, help="New directory; never overwrites earlier exports")
    p.add_argument(
        "--init-decisions",
        type=Path,
        help="Create a hash-bound review template instead of exporting",
    )
    p.add_argument("--orientation", choices=["raw", "exif"], default="raw")
    a = p.parse_args()
    try:
        if a.init_decisions:
            if a.decisions or a.out:
                raise ValueError("Use --init-decisions without --decisions or --out")
            if a.init_decisions.exists():
                raise ValueError("Decision template already exists")
            image = canonical_image(a.image, a.orientation)
            with tempfile.TemporaryDirectory() as t:
                path = Path(t) / "original.png"
                image.save(path)
                sha = file_sha(path)
            source = read_json(a.proposals)
            if source.get("canonical_sha256", source.get("image_sha256")) != sha:
                raise ValueError("Proposals do not match canonical image/orientation")
            if any(o.get("label", "rock") != "rock" for o in source["masks"]):
                raise ValueError("Separate non-rock labels first")
            queue = [
                o["id"]
                for o, f in zip(
                    source["masks"],
                    relation_flags([dict(o, label="rock") for o in source["masks"]], None),
                    strict=False,
                )
                if "inside_larger_same_label_proposal" in f
            ]
            template = {
                "schema": "rock-derived-decisions/1",
                "image_sha256": sha,
                "source_proposals_sha256": file_sha(a.proposals),
                "reviewer": "",
                "touching_regions_reviewed": False,
                "decisions": [],
                "empty_dispositions": {},
                "nested_review_queue": queue,
            }
            write_json(a.init_decisions, template)
            print(
                json.dumps(
                    {
                        "template": str(a.init_decisions),
                        "nested_review_candidates": len(queue),
                        "next": "Set reviewer and record visual decisions; review queue entries are not merge instructions.",
                    },
                    indent=2,
                )
            )
        else:
            if not a.decisions or not a.out:
                raise ValueError("Export requires --decisions and --out")
            print(
                json.dumps(run(a.image, a.proposals, a.decisions, a.out, a.orientation), indent=2)
            )
    except (ValueError, KeyError) as e:
        p.exit(2, f"error: {e}\n")


if __name__ == "__main__":
    main()
