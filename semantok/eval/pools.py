"""Fixed evaluation pools: which clips, where to read them, and what conditions them.

The clip lists in ``semantok/data/splits`` are the exact pools behind the paper numbers. FID and FVD
depend on the sample size and on the pool, so change a list only on purpose.

Kinetics-600 videos are expected at ``<root>/<label>/<youtube_id>_<start:06d>_<end:06d>.mp4`` (the
layout of the standard Kinetics downloader, validation split). uCO3D videos are read from
``<root>/<video_path>``, with ``video_path`` and the captions taken from an index CSV that has the
columns ``category, sequence_name, video_path, short_caption, long_caption``.
"""
import csv
import os

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

    def __init__(self, root, index_csv, name):
        self.root, self.name = root, name
        self.ids = _lines(os.path.join(DATA, "splits", f"{name}.txt"))
        want = set(self.ids)
        self._rows = {}
        with open(index_csv) as f:
            for r in csv.DictReader(f):
                key = f"{r['category']}/{r['sequence_name']}"
                if key in want:
                    self._rows[key] = r
        missing = [c for c in self.ids if c not in self._rows]
        if missing:
            raise KeyError(f"{len(missing)} clips of {name} are not in {index_csv}, e.g. {missing[:3]}")

    def __len__(self):
        return len(self.ids)

    def cond(self, i):
        return self._rows[self.ids[i]]["long_caption"]

    def text(self, i):
        r = self._rows[self.ids[i]]
        return r.get("long_caption") or r.get("short_caption") or ""

    def load(self, i):
        return load_uco3d_clip(os.path.join(self.root, self._rows[self.ids[i]]["video_path"]))
