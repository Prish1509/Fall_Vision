"""
Train / validation / test split files (Member 2, Step 12b).

Which video goes where:
  Le2i, CAUCAFall, MCFD   OmniFall's official cross-subject (cs) splits:
                          train -> train, val -> val, test -> test_staged
                          (cross-subject = the same person never appears in two splits)
  GMDCSA24                Subject 1 -> train, Subject 2 -> val, Subjects 3-4 -> test_gmdcsa24, matching the
                          rule experiments (rules were tuned on S1-2 and reported on S3-4)
  OOPS                    ALL videos -> test_oops (real-world falls, never trained or tuned on)
  Own videos              added later from splits/own_videos_labels.csv -> test_own

Writes:
  splits/videos.csv     one row per video: dataset, path, split, fall_segments, seconds
  splits/summary.md     counts per dataset and split

Segment labels (fall / fallen / walk ...) stay in labels/omnifall/<dataset>.csv and are joined on
(dataset, path) later (Step 18).

Idea used (cite in the report): OmniFall's staged-to-wild protocol and its cs splits
(Schneider et al., arXiv 2505.19889).

Usage:
  python src/make_splits.py --labels labels/omnifall --out splits
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

HUB = "simplexsigil2/omnifall"
STAGED = ["le2i", "caucafall", "mcfd"]
SPLIT_RENAME = {"train": "train", "val": "val", "test": "test_staged"}
ORDER = ["train", "val", "test_staged", "test_gmdcsa24", "test_oops", "test_own"]
FALL = 1     # OmniFall class id for "fall"


def read_rows(path, need=None):
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        if need and need not in (rd.fieldnames or []):
            raise SystemExit(f"{path}: expected a '{need}' column, got {rd.fieldnames}")
        return list(rd)


def hub_file(name, cache):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HUB, name, repo_type="dataset", local_dir=cache)


def main():
    ap = argparse.ArgumentParser(description="Make train / val / test split files")
    ap.add_argument("--labels", default="labels/omnifall", help="folder with <dataset>.csv label files")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--cache", default="/tmp/omnifall_hub", help="where Hub split files are downloaded")
    ap.add_argument("--split-dir", default=None,
                    help="use local copies of the Hub split files (folder with cs/<dataset>/*.csv)")
    a = ap.parse_args()

    # ---- per-video info from the label files ----
    info = {}                                   # (dataset, path) -> dict
    for fname in sorted(os.listdir(a.labels)):
        if not fname.endswith(".csv"):
            continue
        for r in read_rows(os.path.join(a.labels, fname)):
            key = (r["dataset"], r["path"])
            d = info.setdefault(key, dict(subject=r["subject"], fall_segments=0, seconds=0.0))
            d["fall_segments"] += int(r["label"]) == FALL
            d["seconds"] = max(d["seconds"], float(r["end"]))
    datasets_present = {k[0] for k in info}
    print("Label files cover:", ", ".join(sorted(datasets_present)))

    split_of = {}

    def assign(key, split):
        old = split_of.get(key)
        if old is not None and old != split:
            # one video in two splits would leak; keep the stricter (later in ORDER) one
            split = max(old, split, key=ORDER.index)
            print(f"  note: {key[0]}/{key[1]} listed in both {old} and {split}; kept {split}")
        split_of[key] = split

    # ---- staged datasets: official cross-subject splits ----
    for ds in STAGED:
        if ds not in datasets_present:
            print(f"  {ds}: no label file, skipped")
            continue
        for part, ours in SPLIT_RENAME.items():
            name = f"splits/cs/{ds}/{part}.csv"
            p = os.path.join(a.split_dir, "cs", ds, f"{part}.csv") if a.split_dir else hub_file(name, a.cache)
            rows = read_rows(p, need="path")
            for path in {r["path"] for r in rows}:
                if (ds, path) in info:
                    assign((ds, path), ours)

        left = [k for k in info if k[0] == ds and k not in split_of]
        if left:
            print(f"  {ds}: {len(left)} videos are in no official split, so they are not used")

    # ---- GMDCSA24: same subjects as the rule experiments ----
    gm = [k for k in info if k[0].lower() == "gmdcsa24"]
    for k in gm:
        assign(k, {"1": "train", "2": "val"}.get(info[k]["subject"], "test_gmdcsa24"))
    if not gm:
        print("  GMDCSA24: no label file, skipped")

    # ---- OOPS: everything is test ----
    for k in info:
        if k[0] == "OOPS":
            assign(k, "test_oops")

    # ---- own videos (Step 13), if the sheet has real rows ----
    own_csv = os.path.join(a.out, "own_videos_labels.csv")
    if os.path.exists(own_csv):
        for r in read_rows(own_csv):
            if r.get("file") and "example" not in r.get("notes", ""):
                key = ("own", os.path.splitext(r["file"])[0])
                info[key] = dict(subject=r.get("member", ""), seconds=0.0,
                                 fall_segments=int(r.get("type", "") == "fall"))
                assign(key, "test_own")

    # ---- write ----
    os.makedirs(a.out, exist_ok=True)
    out_csv = os.path.join(a.out, "videos.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "path", "split", "subject", "fall_segments", "seconds"])
        for (ds, path), sp in sorted(split_of.items(), key=lambda kv: (ORDER.index(kv[1]), kv[0])):
            d = info[(ds, path)]
            w.writerow([ds, path, sp, d["subject"], d["fall_segments"], round(d["seconds"], 2)])

    counts = Counter((ds, sp) for (ds, _), sp in split_of.items())
    falls = defaultdict(int)
    for (ds, path), sp in split_of.items():
        falls[(ds, sp)] += info[(ds, path)]["fall_segments"] > 0
    unused = sorted({k[0] for k in info} - {k[0] for k in split_of})

    lines = ["# Splits (Step 12b)", "",
             "Videos per dataset and split (videos containing at least one fall in brackets).",
             "OOPS and own videos are test only: never used for training or tuning.", "",
             "| Dataset | " + " | ".join(ORDER) + " |", "|---|" + "---|" * len(ORDER)]
    for ds in sorted({ds for ds, _ in counts}):
        cells = [f"{counts[(ds, sp)]} ({falls[(ds, sp)]})" if counts[(ds, sp)] else "-" for sp in ORDER]
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    tot = [sum(counts[(ds, sp)] for ds, _ in counts.keys() if _ == sp) for sp in ORDER]
    lines.append("| **total** | " + " | ".join(str(t) if t else "-" for t in tot) + " |")
    if unused:
        lines += ["", f"Label files not used for any split (no videos downloaded): {', '.join(unused)}"]
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out_csv} ({len(split_of)} videos) and {os.path.join(a.out, 'summary.md')}")


if __name__ == "__main__":
    main()
