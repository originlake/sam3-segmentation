"""Geometry and source-preservation checks for the CPU LabelMe converter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest

import numpy as np
from compact_labelme import convert
from core import bbox, encode, file_sha, read_json, write_json
from PIL import Image
from rock_export import run


class CompactLabelMeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "source.png"
        Image.new("RGB", (20, 20), "gray").save(self.image)
        self.derived = self.root / "derived"
        self.output = self.root / "output.json"

    def fixture(self, masks):
        proposals, decisions = self.root / "proposals.json", self.root / "decisions.json"
        write_json(
            proposals,
            {
                "canonical_sha256": file_sha(self.image),
                "masks": [
                    {
                        "id": key,
                        "label": "rock",
                        "rle": encode(mask),
                        "area": int(mask.sum()),
                        "bbox": bbox(mask),
                        "score": 0.9,
                    }
                    for key, mask in masks.items()
                ],
            },
        )
        write_json(
            decisions,
            {
                "schema": "rock-derived-decisions/1",
                "image_sha256": file_sha(self.image),
                "source_proposals_sha256": file_sha(proposals),
                "reviewer": "synthetic-test",
                "touching_regions_reviewed": True,
                "decisions": [],
            },
        )
        run(self.image, proposals, decisions, self.derived)

    def test_compaction_preserves_disconnected_slivers_and_corner_contacts(self):
        a = np.zeros((20, 20), bool)
        a[2:8, 2:8] = True
        a[8:11, 2:5] = True
        a[8, 8] = True
        a[15, 1:19] = True
        a[0, 19] = True
        b = np.zeros_like(a)
        b[2:8, 8:13] = True
        self.fixture({"a": a, "b": b})
        before = {p: file_sha(p) for p in self.derived.rglob("*") if p.is_file()}
        report = convert(self.derived, self.image, self.output)
        doc = read_json(self.output)
        self.assertEqual(report["instances"], 2)
        self.assertEqual(report["mask_roundtrip"], "exact at pixel centers")
        self.assertEqual({s["label"] for s in doc["shapes"]}, {"rocks"})
        self.assertEqual(report["rectangle_fallback_groups"], [])
        self.assertLess(report["polygon_pieces"], report["original_rectangle_pieces"])
        self.assertEqual({p: file_sha(p) for p in before}, before)
        self.assertTrue(self.output.with_suffix(".validation.json").exists())

    def test_enclosed_neighbor_uses_solid_fallback_without_pixel_loss(self):
        a = np.zeros((20, 20), bool)
        a[2:15, 2:15] = True
        a[5:12, 5:12] = False
        b = np.zeros_like(a)
        b[5:12, 5:12] = True
        self.fixture({"a": a, "b": b})
        report = convert(self.derived, self.image, self.output)
        self.assertEqual(report["rectangle_fallback_groups"], [1])
        self.assertEqual(report["interior_rings"], 0)
        self.assertEqual(report["positive_area_intersections"], 0)

    def test_simplification_reduces_vertices_and_backs_off_for_exact_pixels(self):
        a = np.zeros((20, 20), bool)
        for y in range(2, 16):
            a[y, 2 : y + 3] = True
        self.fixture({"a": a})
        report = convert(self.derived, self.image, self.output, simplify_tolerance=4)
        details = report["simplification"]
        self.assertLess(details["vertices_after"], details["vertices_before"])
        self.assertTrue(
            any(
                r["reason"] == "changes_mask_pixels"
                for g in details["groups"]
                for r in g["rejections"]
            )
        )
        self.assertEqual(report["mask_roundtrip"], "exact at pixel centers")
        exact = self.root / "without-simplification.json"
        disabled = convert(self.derived, self.image, exact, simplify_tolerance=0)
        self.assertEqual(
            disabled["simplification"]["vertices_before"],
            disabled["simplification"]["vertices_after"],
        )

    def test_simplification_preserves_continuous_separation_and_enclosed_neighbor(self):
        a = np.zeros((20, 20), bool)
        for y in range(2, 16):
            a[y, 2 : y + 3] = True
        b = ~a
        self.fixture({"a": a, "b": b})
        report = convert(self.derived, self.image, self.output, simplify_tolerance=4)
        self.assertEqual(report["positive_area_intersections"], 0)
        self.assertEqual(report["shared_pixels"], 0)
        self.assertEqual(report["interior_rings"], 0)
        self.assertIn(2, report["rectangle_fallback_groups"])

    def test_invalid_simplification_tolerance_rejected_before_output(self):
        a = np.zeros((20, 20), bool)
        a[2:8, 2:8] = True
        self.fixture({"a": a})
        for tolerance in [-1, float("nan"), float("inf")]:
            with self.assertRaisesRegex(ValueError, "tolerance"):
                convert(self.derived, self.image, self.output, simplify_tolerance=tolerance)
            self.assertFalse(self.output.exists())

    def test_source_pixel_mismatch_rejected_before_output(self):
        a = np.zeros((20, 20), bool)
        a[2:8, 2:8] = True
        self.fixture({"a": a})
        Image.new("RGB", (20, 20), "black").save(self.image)
        with self.assertRaisesRegex(ValueError, "pixels mismatch"):
            convert(self.derived, self.image, self.output)
        self.assertFalse(self.output.exists())

    def test_inconsistent_saved_png_or_ownership_rejected(self):
        a = np.zeros((20, 20), bool)
        a[2:8, 2:8] = True
        self.fixture({"a": a})
        mask_path = self.derived / "masks/000001.png"
        Image.new("L", (20, 20), 0).save(mask_path)
        with self.assertRaisesRegex(ValueError, "PNG mask mismatch"):
            convert(self.derived, self.image, self.output)
        self.assertFalse(self.output.exists())
        Image.fromarray(a.astype("uint8") * 255).save(mask_path)
        owner = np.load(self.derived / "ownership.npy")
        owner[0, 0] = 1
        np.save(self.derived / "ownership.npy", owner)
        with self.assertRaisesRegex(ValueError, "ownership round trip"):
            convert(self.derived, self.image, self.output)
        self.assertFalse(self.output.exists())

    def test_output_reference_label_override_and_overwrite_guard(self):
        a = np.zeros((20, 20), bool)
        a[2:8, 2:8] = True
        self.fixture({"a": a})
        folder = self.root / "out"
        folder.mkdir()
        output = folder / "source.json"
        convert(self.derived, self.image, output, label="stone")
        doc = read_json(output)
        self.assertEqual((output.parent / doc["imagePath"]).resolve(), self.image)
        self.assertEqual({s["label"] for s in doc["shapes"]}, {"stone"})
        before = file_sha(output)
        with self.assertRaisesRegex(ValueError, "exists"):
            convert(self.derived, self.image, output)
        self.assertEqual(file_sha(output), before)
        convert(self.derived, self.image, output, overwrite=True)
        self.assertEqual({s["label"] for s in read_json(output)["shapes"]}, {"rocks"})
        with self.assertRaisesRegex(ValueError, "source artifacts"):
            convert(self.derived, self.image, self.derived / "annotations.json", overwrite=True)


if __name__ == "__main__":
    unittest.main()
