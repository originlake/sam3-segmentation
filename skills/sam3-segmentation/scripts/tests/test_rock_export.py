"""CPU integration checks for saved rock exports; no model or user photo needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import tempfile
import unittest

import numpy as np
from core import bbox, decode, encode, file_sha, read_json, write_json
from PIL import Image
from rock_export import decisions_for, rectangles, run


class RockExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        Image.new("RGB", (20, 20), "gray").save(self.image)
        self.source = self.root / "proposals.json"
        self.decisions = self.root / "decisions.json"

    def fixture(self, masks, decisions=(), **extra):
        objects = [
            {
                "id": i,
                "label": "rock",
                "area": int(m.sum()),
                "bbox": bbox(m),
                "rle": encode(m),
                "score": 0.6,
            }
            for i, m in masks.items()
        ]
        write_json(self.source, {"canonical_sha256": file_sha(self.image), "masks": objects})
        doc = {
            "schema": "rock-derived-decisions/1",
            "image_sha256": file_sha(self.image),
            "source_proposals_sha256": file_sha(self.source),
            "reviewer": "synthetic-test",
            "decisions": list(decisions),
            **extra,
        }
        write_json(self.decisions, doc)
        return doc

    def verify_saved(self, out):
        saved = read_json(out / "annotations.json")
        owner = np.load(out / "ownership.npy")
        raster = np.zeros_like(owner)
        for s in saved["shapes"]:
            (x0, y0), (x1, _), (_, y1), _ = s["points"]
            self.assertLess(x0, x1)
            self.assertLess(y0, y1)
            self.assertFalse(raster[y0:y1, x0:x1].any())
            raster[y0:y1, x0:x1] = s["group_id"]
        np.testing.assert_array_equal(owner, raster)
        for a in saved["annotations"]:
            m = owner == a["group_id"]
            np.testing.assert_array_equal(decode(a["segmentation"]), m)
            np.testing.assert_array_equal(np.array(Image.open(out / a["mask_file"])) > 0, m)
            self.assertEqual(int(m.sum()), a["area"])
            self.assertEqual(bbox(m), a["bbox"])
            self.assertNotIn("score", a)
        return saved

    def test_rectangles_preserve_holes_slivers_corners_and_components(self):
        m = np.zeros((20, 20), bool)
        m[2:12, 2:12] = True
        m[4:8, 4:8] = False
        m[12, 12] = True
        m[18, :] = True
        m[0, 19] = True
        r = np.zeros_like(m, dtype=int)
        for x0, x1, y0, y1 in rectangles(m):
            r[y0:y1, x0:x1] += 1
        np.testing.assert_array_equal(r, m.astype(int))

    def test_enclosed_neighbor_and_saved_geometry(self):
        ring = np.zeros((20, 20), bool)
        ring[2:15, 2:15] = True
        ring[5:12, 5:12] = False
        center = np.zeros_like(ring)
        center[5:12, 5:12] = True
        self.fixture({"a": ring, "b": center})
        run(self.image, self.source, self.decisions, self.root / "out")
        saved = self.verify_saved(self.root / "out")
        self.assertEqual(len(saved["annotations"]), 2)
        self.assertTrue((np.load(self.root / "out/ownership.npy")[5:12, 5:12] == 2).all())

    def test_merge_closes_hole_and_preserves_provenance(self):
        a = np.zeros((20, 20), bool)
        a[3:12, 3:12] = True
        a[4:11, 4:11] = False
        a[3, 6:9] = False
        b = np.zeros_like(a)
        b[3, 6:9] = True
        self.fixture(
            {"a": a, "b": b},
            [{"action": "merge", "source_ids": ["b"], "target_id": "a", "reason": "same rock"}],
            touching_regions_reviewed=True,
        )
        before = file_sha(self.source)
        check = run(self.image, self.source, self.decisions, self.root / "out")
        saved = self.verify_saved(self.root / "out")
        self.assertEqual(check["surviving_instances"], 1)
        self.assertEqual(saved["annotations"][0]["area"], 81)
        self.assertEqual(set(saved["annotations"][0]["source_scores"]), {"a", "b"})
        self.assertEqual(
            read_json(self.root / "out/geometry-edits.json")["union_hole_fill_pixels"]["a"], 49
        )
        self.assertEqual(
            read_json(self.root / "out/geometry-edits.json")["merge_created_hole_fill_pixels"]["a"],
            49,
        )
        self.assertEqual(file_sha(self.source), before)
        self.assertEqual(
            [r["status"] for r in read_json(self.root / "out/ledger.json")], ["retained", "merged"]
        )
        with self.assertRaises(ValueError):
            run(self.image, self.source, self.decisions, self.root / "out")

    def test_empty_proposals_require_disposition(self):
        m = np.zeros((20, 20), bool)
        m[2:8, 2:8] = True
        doc = self.fixture({"a": m, "b": m}, touching_regions_reviewed=True)
        check = run(self.image, self.source, self.decisions, self.root / "draft")
        self.assertEqual(check["status"], "review_draft")
        self.assertIn("b", check["unresolved_review_ids"])
        rows = read_json(self.root / "draft/ledger.json")
        self.assertEqual(rows[1]["status"], "emptied_by_ownership")
        doc["empty_dispositions"] = {
            "b": {"outcome": "duplicate", "target_id": "a", "reason": "identical reviewed proposal"}
        }
        write_json(self.decisions, doc)
        check = run(self.image, self.source, self.decisions, self.root / "reviewed")
        self.assertEqual(check["status"], "reviewed_model_proposals")
        self.verify_saved(self.root / "reviewed")

    def test_invalid_decisions_and_source_mismatch(self):
        m = np.zeros((20, 20), bool)
        m[1:3, 1:3] = True
        doc = self.fixture({"a": m, "b": m, "c": m})
        for decisions in [
            [{"action": "merge", "source_ids": ["a"], "target_id": "a", "reason": "bad"}],
            [
                {"action": "merge", "source_ids": ["a"], "target_id": "b", "reason": "bad"},
                {"action": "merge", "source_ids": ["b"], "target_id": "c", "reason": "bad"},
            ],
            [
                {"action": "merge", "source_ids": ["a"], "target_id": "b", "reason": "bad"},
                {"action": "exclude", "source_ids": ["b"], "reason": "bad"},
            ],
            [{"action": "keep", "source_ids": ["z"], "reason": "bad"}],
        ]:
            bad = copy.deepcopy(doc)
            bad["decisions"] = decisions
            with self.assertRaises(ValueError):
                decisions_for(
                    bad, {"a", "b", "c"}, doc["image_sha256"], doc["source_proposals_sha256"]
                )
        doc["image_sha256"] = "wrong"
        write_json(self.decisions, doc)
        with self.assertRaises(ValueError):
            run(self.image, self.source, self.decisions, self.root / "bad")
        self.assertFalse((self.root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
