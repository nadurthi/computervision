"""Evaluation: pipeline events against hand labels, and the skill rule against expert .dvw files.

Hand labels: a CSV with columns  rally, t, team, number, skill
  rally  - rally index in your labelling (any integer)
  t      - contact time in seconds (video time)
  team   - '*' home or 'a' away  (or 'near'/'far')
  number - shirt number, empty if unknown
  skill  - S R E A B D F
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .skills import rallies_from_dvw, score_rule_on_rallies


@dataclass
class EventScore:
    labels: int
    predicted: int
    matched: int
    skill_correct: int
    number_known: int
    number_correct: int
    rallies: int
    rallies_all_right: int

    @property
    def recall(self):
        return self.matched / self.labels if self.labels else float("nan")

    @property
    def precision(self):
        return self.matched / self.predicted if self.predicted else float("nan")

    @property
    def skill_acc(self):
        return self.skill_correct / self.matched if self.matched else float("nan")

    def as_dict(self):
        d = dict(self.__dict__)
        d.update(recall=round(self.recall, 4), precision=round(self.precision, 4), skill_acc=round(self.skill_acc, 4))
        return d


def load_labels(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["t"] = float(r["t"])
        r["rally"] = int(r["rally"])
        r["number"] = int(r["number"]) if r.get("number") not in (None, "") else None
    return rows


def score_events(pred: list[dict], labels: list[dict], tol_s: float = 0.3) -> EventScore:
    """pred: rows from events.json. Greedy one-to-one matching in time within tol_s."""
    used = set()
    matched, skill_ok, num_known, num_ok = 0, 0, 0, 0
    per_rally_ok: dict[int, bool] = {r["rally"]: True for r in labels}
    for lab in sorted(labels, key=lambda r: r["t"]):
        best, best_dt = None, tol_s
        for i, p in enumerate(pred):
            if i in used:
                continue
            dt = abs(p["t"] - lab["t"])
            if dt <= best_dt:
                best, best_dt = i, dt
        if best is None:
            per_rally_ok[lab["rally"]] = False
            continue
        used.add(best)
        p = pred[best]
        matched += 1
        if p["skill"] == lab["skill"]:
            skill_ok += 1
        else:
            per_rally_ok[lab["rally"]] = False
        if p.get("number") is not None:
            num_known += 1
            num_ok += int(lab["number"] is not None and p["number"] == lab["number"])
    if labels:
        lo, hi = min(r["t"] for r in labels) - tol_s, max(r["t"] for r in labels) + tol_s
        n_pred = sum(1 for p in pred if lo <= p["t"] <= hi)
    else:
        n_pred = len(pred)
    return EventScore(len(labels), n_pred, matched, skill_ok, num_known, num_ok,
                      len(per_rally_ok), sum(per_rally_ok.values()))


def score_rule_on_dvw_files(paths: list[str], use_block_flag: bool = True) -> dict:
    import datavolley as dv

    rallies = []
    for p in paths:
        rallies += rallies_from_dvw(dv.read_dv(p))
    s = score_rule_on_rallies(rallies, use_block_flag)
    return {
        "files": len(paths), "rallies": s.rallies, "touches": s.total,
        "acc_all": round(s.acc(), 4), "acc_excl_free_ball": round(s.acc(("F",)), 4),
        "acc_free_ball": round(s.class_acc("F"), 4),
        "per_skill": {k: {"n": s.per_class[k], "acc": round(s.class_acc(k), 4)} for k in "SREABDF"},
        "rallies_all_right": s.rallies_all_right,
        "top_confusions": [[a, b, n] for (a, b), n in s.confusion.most_common() if a != b][:8],
    }
