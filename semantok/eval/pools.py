"""Fixed evaluation pools: which clips, where to read them, and what conditions them.

The clip lists in ``semantok/data/splits`` are the exact pools behind the paper numbers. FID and FVD
depend on the sample size and on the pool, so change a list only on purpose.

Kinetics-600 videos are expected at ``<root>/<label>/<youtube_id>_<start:06d>_<end:06d>.mp4`` (the
layout of the standard Kinetics downloader, validation split). uCO3D is read from the user's copy of the
official release: video paths and captions come from ``<root>/metadata.sqlite`` (table
``sequence_annots``), so no uCO3D annotation ships with this repository.
"""
import csv
import os
import sqlite3

from ..data.video import load_kinetics_clip, load_uco3d_clip

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _lines(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def k600_classes():
    """The 597 Kinetics-600 classes of our copy, in AR class-id order."""
    return _lines(os.path.join(DATA, "k600_classes.txt"))


class K600Pool:
    """``<label>/<file>.mp4`` clips. ``cond(i)`` is the class id, ``text(i)`` the class name (the
    ViCLIP prompt, used verbatim)."""

    dataset = "k600"

    def __init__(self, root, name="k600_val_bank_2048"):
        self.root, self.name = root, name
        self.ids = _lines(os.path.join(DATA, "splits", f"{name}.txt"))
        self.classes = k600_classes()
        self._cid = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.ids)

    def label(self, i):
        return self.ids[i].split("/", 1)[0]

    def cond(self, i):
        return self._cid[self.label(i)]

    def text(self, i):
        return self.label(i)

    def load(self, i):
        return load_kinetics_clip(os.path.join(self.root, self.ids[i]))


class UCO3DPool:
    """``<category>/<sequence_name>`` clips. ``cond(i)`` is the long caption (umT5 input);
    ``text(i)`` the ViCLIP prompt (long caption, else short caption)."""

    dataset = "uco3d"

    def __init__(self, root, name, index_csv=None):
        """``index_csv`` (columns category, sequence_name, video_path, short_caption, long_caption)
        replaces the official ``metadata.sqlite`` when given."""
        self.root, self.name = root, name
        self.ids = _lines(os.path.join(DATA, "splits", f"{name}.txt"))
        want = set(self.ids)
        rows = _read_index_csv(index_csv) if index_csv else _read_uco3d_metadata(root)
        self._rows = {k: r for k, r in rows if k in want}
        missing = [c for c in self.ids if c not in self._rows]
        if missing:
            raise KeyError(f"{len(missing)} clips of {name} are missing from the uCO3D metadata, "
                           f"e.g. {missing[:3]}")

    def __len__(self):
        return len(self.ids)

    def cond(self, i):
        return self._rows[self.ids[i]]["long_caption"]

    def text(self, i):
        r = self._rows[self.ids[i]]
        return r.get("long_caption") or r.get("short_caption") or ""

    def load(self, i):
        return load_uco3d_clip(os.path.join(self.root, self._rows[self.ids[i]]["video_path"]))


def _read_uco3d_metadata(root):
    """(``category/sequence_name``, row) from the official ``metadata.sqlite``."""
    path = os.path.join(root, "metadata.sqlite")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found: --uco3d-root must point at the official uCO3D download")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cur = con.execute("SELECT category, sequence_name, _video_path, _short_caption_text, _caption_text "
                          "FROM sequence_annots")
        for cat, seq, video, short, long in cur:
            yield f"{cat}/{seq}", {"video_path": video, "short_caption": short or "",
                                   "long_caption": long or ""}
    finally:
        con.close()


def _read_index_csv(path):
    with open(path) as f:
        for r in csv.DictReader(f):
            yield f"{r['category']}/{r['sequence_name']}", r
