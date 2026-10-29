# Individual-rock export policy

The user's default requirement is mutually exclusive individual rocks with solid,
hole-free polygons. Apply it to rock annotations in both direct and full modes,
unless the user explicitly changes the policy. This does not flatten unrelated
parent/part categories into one label map.

## Derive the deliverable

### Review physical identity before geometry cleanup

Inspect every `inside_larger_same_label_proposal` flag against the native source
and its larger proposal. Also inspect touching rock regions across the image for
duplicate surfaces or fragments of one rock; unflagged proposals can have the
same error. A few representative close-ups do not complete this review.

The review flag triggers when at least **25% of the smaller mask's pixels**
overlap a larger same-label proposal. Evaluate this on the original overlapping
proposals before exclusive ownership removes the overlap. This broad review
threshold does not change automatic deduplication or authorize automatic merges.
Touching fragments with little overlap, such as 128 beside 136 in `2.jpg`, still
need visual inspection; lowering the threshold cannot flag every identity error.

Decide whether the apparent boundary separates physical rocks or only faces,
shadows, or texture on one rock. Do not automatically merge every nested or
touching pair: real rocks can overlap or occlude one another. Record each flagged
case as distinct, duplicate/part, or uncertain, with the inspected IDs and reason.
For a duplicate/part, retain one physical identity and preserve supported visible
pixels through a recorded merge in the derived export. Preserve original proposal
IDs and masks in the raw records; use the full workflow's decision command for
proposal exclusions when applicable. Do not edit `selected.json` or `refined.json`.

Complete this identity review before assigning contested pixels. Exclusive pixel
ownership can carve a part proposal out of its parent, leaving a false split that
passes all overlap and hole checks. Reinspect the final preview after cleanup.
If review remains incomplete, label the output as a review draft and state the
unreviewed scope; do not imply that geometry validation completed identity review.

Observed failure: in `2.jpg`, proposal 425 covered the lower face of rock 410 and
was already flagged as nested. Overlap cleanup preserved both labels as exclusive
regions, hiding the duplicate identity from format checks. Native-image review
supported merging 425 into 410. IDs describe this example, not a general merge rule.

### Record derived merges explicitly

Use the CPU-only command `rock_export.py` through the skill launcher. It accepts
full-workflow `export/proposals.json` or a direct `sam.py` result JSON containing
rock masks. Match the source orientation (`raw` or `exif`):

```sh
sh "$S/scripts/run" base rock_export.py \
  --image /path/photo.jpg --proposals /path/run/export/proposals.json \
  --orientation raw --init-decisions /path/rock-decisions.json
# Set reviewer and record inspected decisions in rock-decisions.json.
sh "$S/scripts/run" base rock_export.py \
  --image /path/photo.jpg --proposals /path/run/export/proposals.json \
  --orientation raw --decisions /path/rock-decisions.json --out /path/new-rock-export
```

Initialization supplies actual image/source hashes and a nested-review queue at
the current flag threshold; a queue entry is not a decision. Export outputs
`annotations.json` (LabelMe polygons plus matching RLE), `preview.png`, native
`masks/`, `ownership.npy`, `ledger.json`, `geometry-edits.json`, `decisions.json`
and `validation.json`. It copies the canonical source as `original.png`.
The output directory must be new. Earlier annotations and raw files are preserved.

Supported decision actions are `merge`, `exclude`, `keep`, and `uncertain`, all
using `source_ids`. `merge` also needs `target_id`. Set `touching_regions_reviewed`
to true only after that visual review is complete. Inspect `ledger.json` for
`emptied_by_ownership` rows; record dispositions under `empty_dispositions`, e.g.:

```json
{"mask-00002": {"outcome": "duplicate", "target_id": "mask-00001", "reason": "Same visible rock, checked against source"}}
```

Allowed outcomes are `duplicate`, `erroneous`, and `unresolved`; duplicates need a
surviving target. Regenerate to a new directory after updating decisions. Unreviewed
nested cases, uncertain decisions, missing empty dispositions or incomplete touching
review leave the output marked `review_draft`. Even `reviewed_model_proposals`
does not mean human acceptance or measured accuracy. The command checks record
consistency; it cannot verify the truth of a reviewer's reason or completeness claim.

This exporter recomputes ownership from source unions each time, avoiding stale
map reuse. Merged identities preserve per-source scores without an aggregate
confidence. Its exact rectangle polygons can have a high piece count; pieces with
the same `group_id` are one rock. Run CPU regression checks with:

```sh
sh "$S/scripts/run" base python "$S/scripts/tests/test_rock_export.py"
```

`segment.py decide` supports `exclude`, `uncertain`, and `keep`; it does **not**
implement merges. Keep merges in a separate derived decision file. Do not pass a
new `merge` action to that command or modify its output files to simulate one.
For example (replace both hashes with actual SHA256 values):

```json
{
  "schema": "rock-derived-decisions/1",
  "image_sha256": "CANONICAL_IMAGE_SHA256",
  "source_proposals_sha256": "SOURCE_PROPOSALS_FILE_SHA256",
  "reviewer": "reviewer identity",
  "decisions": [
    {
      "action": "merge",
      "source_ids": ["mask-00425"],
      "target_id": "mask-00410",
      "reason": "Lower face of the same visible rock",
      "uncertainty": "medium"
    }
  ]
}
```

Validate image/source hashes, existing IDs, nonempty reasons and unique source
assignments. Reject self-merges, cycles and targets that are excluded or also
merged away; normalize a reviewed chain to one surviving target explicitly.
Union the source and target masks before ownership assignment. For an incremental
edit, reassign source-owned pixels to the target, retain unrelated ownership, and
record that the edit used the existing ownership map. Keep the surviving target
ID and all contributing source IDs; do not attribute the combined mask's confidence
to the target's original score. Preserve individual source scores as provenance.

Bind any cached ownership map to the canonical image, proposal file, decision file,
and geometry-policy hashes. A filename alone is insufficient to reuse it after edits.

### Highlight thin tails before delivery

Geometry checks do not detect a narrow mask extension leaking onto background.
After generating a derived rock export, run the CPU-only highlighter and inspect
each flagged region against the canonical source pixels:

```sh
sh "$S/scripts/run" base highlight_rock_tails.py \
  /path/new-rock-export/annotations.json \
  --radius 4 --out /path/new-tail-review
```

The input is the derived `rock_export.py` JSON with
`annotations[].segmentation` COCO RLE, not raw direct-call JSON or polygon-only
LabelMe files. The image defaults to `imagePath` relative to that JSON; use
`--image /path/original.png` to override it with the same canonical pixels.
The output directory must be new. Outputs are `preview.png` (red suspected
tails, yellow flagged rock boundaries, source rock IDs) and `flags.json`
(candidate pixel regions as crop-local RLE with native bounding-box origins,
areas, extension depths, input hashes and settings). Source annotations and
masks remain unchanged.

The detector uses a disk opening to estimate a thicker body, then flags small
connected residual regions extending sufficiently far beyond it. Defaults are
`--radius 4`, `--min-area 4`, `--max-fraction 0.08` of the rock area and
`--min-depth 6` pixels (default depth is 1.5 times radius). Radius is in native
pixels: adjust it to the suspected tail width and image scale, not a universal
rock constant. This heuristic can also flag detached specks and genuine pointed
corners; wide leakage and masks with no surviving thick body may be missed.
Absence of flags does not certify a clean boundary.

Inspect every flag before deciding to remove pixels. Do not replace the full
mask with its opened version or automatically delete all highlighted regions.
Keep uncertain protrusions marked for review. Record confirmed removals with
source IDs, exact removed pixels, reason and input hashes in a separate derived
edit record, preserving raw proposals. After any removal, regenerate all masks,
polygons, areas, boxes and previews from the edited ownership map; repeat hole,
overlap and exact polygon/mask checks, then rerun the tail highlighter. The
highlighter itself implements no removal or regeneration command.

Observed example: at the defaults, the upper-right noisy tail of rock 41 in
`3.jpg` was highlighted. Five of that image's 172 proposals were flagged; this
is a regression example, not an accuracy benchmark or expected flag count.

### Apply the geometry policy

1. Preserve the original model masks and review records. Apply the following
   geometry edits to a separate derived export, never by editing `selected.json`
   or `refined.json`.
2. Fill enclosed background voids inside each rock mask. Keep the boundary
   supported by the visible rock; do not bridge open gaps between separate rocks.
3. Assign every contested pixel to exactly one rock. Use a recorded, deterministic
   ownership rule and visually inspect touching boundaries. One workable rule is
   greatest distance inside the candidate mask, with original confidence breaking
   ties between candidates from the same model head. This is a geometry heuristic,
   not proof of the correct physical boundary.
4. Recheck holes after ownership assignment. Fill remaining enclosed **unassigned**
   pixels without taking pixels from another rock. If another rock occupies an
   enclosed region, partition the surrounding geometry into solid simple polygons
   rather than filling over that rock. Give all pieces of the same rock one
   LabelMe `group_id`; polygon-piece count is not rock-instance count.
5. Regenerate all delivered masks/RLE, polygons, areas, boxes and previews from
   the final ownership map. Record hole filling and overlap resolution as export
   edits; do not silently attach original RLE to changed polygons.

After **every merge, exclusion or ownership edit**, repeat hole handling and
regeneration. A union can close a formerly open gap: merging 128 and 168 into 136
did this in `2.jpg`. Fill only enclosed unassigned pixels; never overwrite another
rock. Record added pixels separately for initial filling, ownership repair and
post-merge filling, rather than reporting only the first-stage count.

### Preserve exact polygon geometry

Use a native coordinate convention explicitly: pixel `(x,y)` occupies
`[x,x+1] × [y,y+1]`, and rasterization tests its center `(x+0.5,y+0.5)`.
Construct geometry from pixel cells or contiguous row runs. Avoid contour
simplification unless its saved polygons still reproduce every final mask pixel.

When splitting around an enclosed neighboring rock, preserve every polygonal
component returned by intersection, including polygons nested in a geometry
collection. Do not silently discard a result because it is not a single Polygon
or MultiPolygon. An initial triangle/intersection implementation lost pixels in
this task; the round-trip check caught it. Use only an exporter validated on
disconnected regions, one-pixel slivers, enclosed neighbors and corner contacts.

If a contour/splitting exporter fails, a correctness-first fallback is one
rectangle for each contiguous horizontal run of foreground pixels, optionally
coalescing identical runs on adjacent rows. Each rectangle is a solid simple
polygon; use one `group_id` for all rectangles belonging to the same rock. This
can produce many pieces, so disclose the piece count separately from instance
count. Verify saved polygon-to-mask agreement even with this fallback. Never
bypass a mismatch or substitute original RLE for altered geometry.

### Compact LabelMe delivery

Before final LabelMe export, ask for the exact shape label name unless the user
has already supplied it in the current task or session. Do not assume that the
requested object category is the desired label spelling or singular/plural form.
Continue independent segmentation and review while awaiting the label, then pass
it explicitly with `--label`. Reuse an already specified name without asking again.

After final identity, geometry and tail review, use the CPU-only converter to
turn the derived export's rectangle pieces into compact LabelMe polygons:

```sh
sh "$S/scripts/run" base compact_labelme.py \
  /path/final-rock-export /path/photo.jpg --out /path/photo.json --label "CONFIRMED_LABEL"
```

The original image must have exactly the canonical pixels used by the export,
under its recorded orientation policy. The output references that image using a
path relative to the JSON; `imageData` is null. Omitting `--out` writes beside the
original with the same stem. Existing output or validation sidecars are refused
unless `--overwrite` is explicit. Source export artifacts are never overwritten.

The CLI fallback is `rocks`, but agents must obtain the user's label and pass
`--label NAME` explicitly rather than silently relying on that fallback. This changes
only the delivered shape label, not raw proposal labels, text prompts or instance
identities. Each rock keeps its `group_id` and source ID in the description.
Disconnected pieces remain separate polygons with the same group ID.

The converter uses Shapely (included in the locked base environment) to union
pixel-run rectangles, then apply conservative Douglas-Peucker simplification.
The default maximum tolerance is **0.5 native pixels**; change it with
`--simplify-tolerance PIXELS`, or set `--simplify-tolerance 0` to retain the exact
union boundaries without this extra step. No coordinate rounding is applied.

A proposed simplified boundary is retained only when it reduces vertex count,
keeps every component valid, solid and nonempty, reproduces the same native mask
pixels exactly, and has zero positive-area intersection with other rocks.
Topological preservation for one polygon alone is insufficient: neighboring-rock
geometry is checked as well. Each group tries the requested tolerance, four
successive halvings and finally zero tolerance (removing only redundant collinear
vertices). If no candidate passes, the original boundary is retained. Accepted
groups are checked against both previously simplified and remaining original
neighbors, so later groups cannot introduce an overlap. Simplification does not
fill gaps, change masks, remove small components or join separate rock identities.

If the union encloses a neighboring rock and has an interior ring, that group's
original solid rectangle pieces are retained and extra simplification is skipped.
Every polygon component, including those in geometry collections, is preserved.
Piece count can therefore exceed rock count; inspect `rectangle_fallback_groups`
in the report. The `simplification` record gives requested and accepted tolerances,
vertex counts before/after and rejected candidates with reasons for each group.

Before delivery it checks the serialized and saved polygons against the derived
ownership map, PNG masks, RLE, areas and boxes using pixel-center rasterization.
It rejects shared pixels, positive-area intersections, invalid polygons, interior
rings and enclosed unassigned voids. The adjacent `photo.validation.json` records
counts, fallback groups, input/output hashes and the inherited review status.
Neither compaction nor relabeling upgrades a review draft to accepted ground truth.
The derived export, raw proposals and review records remain unchanged. Continue
using the derived `annotations.json`, not the polygon-only LabelMe file, for the
tail highlighter.

Run CPU geometry and source-preservation regression checks with:

```sh
sh "$S/scripts/run" base python "$S/scripts/tests/test_compact_labelme.py"
```

### Account for disappearing proposals

Produce a source-to-output ledger covering **every input proposal**: retained,
merged into a named survivor, explicitly excluded with reason, or emptied by
ownership assignment. Include source and final pixel areas and surviving IDs.
Count proposals before cleanup, surviving physical-identity proposals and polygon
pieces separately. An emptied mask is not automatically a reviewed duplicate.

Before delivery, inspect each ownership-emptied proposal against the winning
owners and source pixels. Record whether it was a duplicate, an erroneous mask,
or a potentially lost distinct rock. Restore/reassign a distinct rock when the
source supports it; otherwise report the unresolved loss in the review draft.
Do not silently drop it because the exporter skips empty masks. Three proposals
lost all pixels during the initial `2.jpg` cleanup without a disposition; that
was incomplete accounting, not evidence that three duplicate rocks were removed.

## Validate the saved output

- Identity review is recorded separately from geometry validation: every nested
  flag has a disposition, touching regions have been inspected, and unresolved
  physical-instance ambiguities are reported.
- Merge records and the complete source-to-output ledger agree with saved output
  IDs; no ownership-emptied source disappears without a reviewed disposition.
- All post-edit hole repairs are included in edit counts, and merged masks retain
  contributing source identities/scores rather than an inherited confidence claim.
- Each labeled pixel belongs to at most one rock.
- Different rocks have zero positive-area polygon intersection; shared boundary
  edges are allowed.
- Every polygon is valid, nonempty and has no interior rings. No rock encloses
  unassigned background pixels.
- Rasterized polygons match their corresponding final instance masks exactly
  under the recorded native coordinate convention. RLE, areas, boxes, image
  dimensions and image reference agree with those same masks.
- Inspect the regenerated preview and report remaining model uncertainty.
- Inspect the thin-tail highlights against source pixels; record retained
  uncertainties and any confirmed derived removals separately from geometry checks.

These are annotation-output requirements, not a change to the raw model's
capabilities or a claim of human-verified segmentation accuracy.
