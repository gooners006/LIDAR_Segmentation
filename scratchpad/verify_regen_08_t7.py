"""T7 (delegate_brief_2026_08_02.md) verification: output/08 vs output/08_regen.

Checks that detection/tracking artifacts are byte-identical between the old
(2026-08-01, fixed-prior/pre-#38) output/08 and the regenerated output/08_regen
(per-car length estimate + #38 RNG fix). Only completed clouds and tracks.json
metadata are allowed to differ (as in the #35 follow-on regen check).

Reports:
- track-set equality (track_id, first_frame, last_frame, point_count,
  raw_point_count, class, centroid_history)
- md5 match for every *_partial.ply (single-frame raw completion input)
- md5 match for every <track_id>.ply belonging to a track NOT completed in
  EITHER version (raw passthrough output; should be untouched by the regen)
- fallback-frequency count (length_estimate_source == "fallback") among
  completed tracks in the new run
"""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_DIR = os.path.join(ROOT, "output", "08")
NEW_DIR = os.path.join(ROOT, "output", "08_regen")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tracks(d):
    with open(os.path.join(d, "tracks.json")) as f:
        meta = json.load(f)
    return {t["track_id"]: t for t in meta["tracks"]}


def main():
    old_tracks = load_tracks(OLD_DIR)
    new_tracks = load_tracks(NEW_DIR)

    old_ids, new_ids = set(old_tracks), set(new_tracks)
    print(f"old tracks: {len(old_ids)}, new tracks: {len(new_ids)}")
    if old_ids != new_ids:
        print("TRACK SET MISMATCH:")
        print("  only in old:", sorted(old_ids - new_ids)[:20])
        print("  only in new:", sorted(new_ids - old_ids)[:20])
    else:
        print("Track set identical (same track_ids).")

    id_fields = ["first_frame", "last_frame", "point_count", "raw_point_count",
                 "class", "centroid_history"]
    field_mismatches = []
    for tid in sorted(old_ids & new_ids):
        o, n = old_tracks[tid], new_tracks[tid]
        for field in id_fields:
            if o.get(field) != n.get(field):
                field_mismatches.append((tid, field))
    print(f"Field mismatches (should be 0): {len(field_mismatches)}")
    if field_mismatches:
        print("  sample:", field_mismatches[:10])

    # Partial-ply md5 comparison (should always match: single-frame raw input
    # is a function of detection/tracking only, not completion).
    partial_mismatches, partial_missing = [], []
    old_obj_dir = os.path.join(OLD_DIR, "objects")
    new_obj_dir = os.path.join(NEW_DIR, "objects")
    old_partials = {f for f in os.listdir(old_obj_dir) if f.endswith("_partial.ply")}
    new_partials = {f for f in os.listdir(new_obj_dir) if f.endswith("_partial.ply")}
    if old_partials != new_partials:
        print("PARTIAL FILE SET MISMATCH:")
        print("  only in old:", sorted(old_partials - new_partials)[:20])
        print("  only in new:", sorted(new_partials - old_partials)[:20])
    for fname in sorted(old_partials & new_partials):
        o_md5 = md5(os.path.join(old_obj_dir, fname))
        n_md5 = md5(os.path.join(new_obj_dir, fname))
        if o_md5 != n_md5:
            partial_mismatches.append(fname)
    print(f"_partial.ply compared: {len(old_partials & new_partials)}, "
          f"mismatches: {len(partial_mismatches)}")
    if partial_mismatches:
        print("  sample:", partial_mismatches[:10])

    # For tracks NOT completed in either version, the main <track_id>.ply
    # should be the untouched raw passthrough -> must match byte-for-byte.
    raw_mismatches = []
    n_checked = 0
    for tid in sorted(old_ids & new_ids):
        o, n = old_tracks[tid], new_tracks[tid]
        if o.get("completed") or n.get("completed"):
            continue  # completed clouds are allowed to differ
        fname = f"{tid}.ply"
        o_path, n_path = os.path.join(old_obj_dir, fname), os.path.join(new_obj_dir, fname)
        if not (os.path.exists(o_path) and os.path.exists(n_path)):
            continue
        n_checked += 1
        if md5(o_path) != md5(n_path):
            raw_mismatches.append(tid)
    print(f"Non-completed track .ply compared: {n_checked}, mismatches: {len(raw_mismatches)}")
    if raw_mismatches:
        print("  sample:", raw_mismatches[:10])

    # Fallback-frequency count (answers #40 caveat (b))
    completed_new = [t for t in new_tracks.values() if t.get("completed")]
    with_source = [t for t in completed_new if "length_estimate_source" in t]
    fallback = [t for t in with_source if t["length_estimate_source"] == "fallback"]
    print(f"\nCompleted tracks (new): {len(completed_new)}")
    print(f"  with length_estimate_source recorded: {len(with_source)}")
    print(f"  fallback (< COMPLETION_LENGTH_MIN_FRAMES gate-passed frames): "
          f"{len(fallback)} ({100 * len(fallback) / max(1, len(with_source)):.1f}%)")

    regression = bool(field_mismatches or partial_mismatches or raw_mismatches
                       or old_ids != new_ids)
    print(f"\nREGRESSION DETECTED: {regression}")


if __name__ == "__main__":
    main()
