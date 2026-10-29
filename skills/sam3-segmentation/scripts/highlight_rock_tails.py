#!/usr/bin/env python3
"""Highlight narrow protrusion candidates; never edit the source annotations.

Requires numpy, scipy, Pillow and pycocotools (SAM skill base environment).
Input: rock_export annotations JSON with annotations[].segmentation COCO RLE.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pycocotools import mask as coco
from scipy import ndimage as ndi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("annotations", type=Path)
    p.add_argument("--image", type=Path)
    p.add_argument("--out", type=Path, required=True, help="New output directory")
    p.add_argument("--radius", type=int, default=4, help="Opening disk radius in native pixels")
    p.add_argument("--min-area", type=int, default=4)
    p.add_argument("--max-fraction", type=float, default=0.08)
    p.add_argument(
        "--min-depth", type=float, help="Minimum extension beyond opened body; default 1.5 * radius"
    )
    args = p.parse_args()
    if args.radius < 1 or args.min_area < 1 or not 0 < args.max_fraction < 1:
        p.error("radius/min-area must be positive and max-fraction between zero and one")
    depth_gate = args.min_depth if args.min_depth is not None else args.radius * 1.5
    if depth_gate <= 0:
        p.error("min-depth must be positive")
    doc = json.loads(args.annotations.read_text())
    image_path = args.image or args.annotations.parent / doc["imagePath"]
    source = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    w, h = source.size
    if (w, h) != (doc["imageWidth"], doc["imageHeight"]):
        raise ValueError("Image and annotation dimensions differ")
    yy, xx = np.ogrid[-args.radius : args.radius + 1, -args.radius : args.radius + 1]
    disk = xx * xx + yy * yy <= args.radius**2
    rgb = np.asarray(source).copy()
    labels, flags = [], []
    for ann in doc["annotations"]:
        mask = coco.decode(ann["segmentation"]).astype(bool)
        if mask.shape != (h, w):
            raise ValueError("Mask dimensions differ")
        # Padding avoids eroding genuine image-edge silhouettes artificially.
        padded = np.pad(mask, args.radius, mode="edge")
        opened = ndi.binary_opening(padded, structure=disk)[
            args.radius : -args.radius, args.radius : -args.radius
        ]
        if not opened.any():
            continue  # No substantial body: cannot assess a tail reliably.
        lost, count = ndi.label(mask & ~opened)
        distance = ndi.distance_transform_edt(~opened)
        regions = []
        for number, sl in enumerate(ndi.find_objects(lost), 1):
            if sl is None:
                continue
            local = lost[sl] == number
            area = int(local.sum())
            depth = float(distance[sl][local].max())
            if area < args.min_area or area > mask.sum() * args.max_fraction or depth < depth_gate:
                continue
            y0, x0 = sl[0].start, sl[1].start
            rle = coco.encode(np.asfortranarray(local.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            regions.append(
                {
                    "bbox": [x0, y0, local.shape[1], local.shape[0]],
                    "area": area,
                    "extension_depth_px": round(depth, 2),
                    "local_segmentation": rle,
                }
            )
            rgb[sl][local] = [255, 35, 35]
        if regions:
            labels.append(
                (
                    min(r["bbox"][0] for r in regions),
                    min(r["bbox"][1] for r in regions),
                    str(int(ann["id"].split("-")[-1])),
                )
            )
            edge = mask & ~ndi.binary_erosion(mask)
            rgb[edge] = [255, 220, 0]
            # Restore red on flagged boundary pixels too.
            for region in regions:
                x, y, bw, bh = region["bbox"]
                local = coco.decode(region["local_segmentation"]).astype(bool)
                rgb[y : y + bh, x : x + bw][local] = [255, 35, 35]
            flags.append({"id": ann["id"], "status": "needs_visual_review", "regions": regions})
    args.out.mkdir(parents=True, exist_ok=False)
    preview = Image.fromarray(rgb)
    draw = ImageDraw.Draw(preview)
    for x, y, identifier in labels:
        draw.text(
            (max(0, x - 4), max(0, y - 15)),
            identifier,
            fill="white",
            stroke_width=1,
            stroke_fill="black",
        )
    preview.save(args.out / "preview.png")
    report = {
        "method": "disk opening residual with small-area and extension-depth gates",
        "note": "Red = suspected tail; yellow = flagged rock boundary. Heuristic flags only; genuine pointed rocks can trigger. Source annotations unchanged.",
        "annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "parameters": {
            "radius": args.radius,
            "min_area": args.min_area,
            "max_fraction": args.max_fraction,
            "min_depth": depth_gate,
        },
        "flagged_instances": len(flags),
        "flags": flags,
    }
    (args.out / "flags.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"flagged_instances": len(flags), "ids": [f["id"] for f in flags], "out": str(args.out)}
        )
    )


if __name__ == "__main__":
    main()
