"""Skill from touch order: the rule that turns a rally's touches into S / R / E / A / B / D / F.

Input per touch is only what video can give without knowing the skill: which team touched it,
the time, and whether the touch was at the net (a block candidate). Labels are assigned in one
pass over the rally, and each touch's label depends on the labels the rule already gave earlier
touches (never on ground truth), so errors propagate exactly as they would on video.

  - first touch of the rally                                   -> S
  - first touch of a possession, incoming ball was a serve      -> R
  - at the net, just after an opponent attack                   -> B  (does not count as a touch)
  - first touch after an opponent attack                        -> D
  - first touch after a block rebound                           -> A if it goes straight back over, else D
  - first touch after any other ball over (overpass, easy ball) -> A if it goes straight back over, else F
  - later touches: sends the ball over, or the third touch      -> A ; otherwise E

These choices were checked against expert files: e.g. a second touch that crosses the net is
scouted A (setter dump) 140 times and E 26 times in the 7 files bundled with py-datavolley.

The free-ball line is the structural weak spot: a free ball sent over on the third touch looks
exactly like an attack in the touch sequence, so the receiving touch is called D, not F.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

SKILLS = ("S", "R", "E", "A", "B", "D", "F")


@dataclass
class TouchIn:
    team: str                 # any two distinct values, e.g. "*"/"a" or "near"/"far"
    t: float = 0.0            # seconds
    at_net: bool = False      # touch within the net band (block candidate)


def label_rally(touches: Sequence[TouchIn], block_window_s: float = 0.6) -> list[str]:
    labels: list[str] = []
    poss_team: str | None = None
    poss_count = 0             # touches in the current possession, blocks excluded
    last_other: dict[str, tuple[str, float]] = {}   # team -> (label, time) of the last touch by that team
    for i, tc in enumerate(touches):
        nxt = touches[i + 1] if i + 1 < len(touches) else None
        sends_over = nxt is not None and nxt.team != tc.team
        if i == 0:
            lab = "S"
            poss_team, poss_count = tc.team, 0
        elif tc.team != poss_team:
            # ball came over from the other side: start of a new possession
            other = poss_team
            incoming, t_in = last_other.get(other, ("S", tc.t))
            if incoming == "A" and tc.at_net and tc.t - t_in <= block_window_s:
                lab = "B"
                poss_team, poss_count = tc.team, 0
            elif incoming == "S":
                lab = "R"
            elif incoming == "A":
                lab = "D"
            elif incoming == "B":
                lab = "A" if sends_over else "D"
            else:
                # overpass or easy ball: hit straight back over, or straight to the floor = attack
                lab = "A" if (sends_over or nxt is None) else "F"
            if lab != "B":
                poss_team, poss_count = tc.team, 1
        else:
            prev_own = labels[-1]
            if prev_own == "B" and poss_count == 0:
                lab = "D"             # first counted touch after own block = dig (block cover)
                poss_count = 1
            else:
                poss_count += 1
                if sends_over or poss_count >= 3 or nxt is None and poss_count >= 2:
                    lab = "A"
                else:
                    lab = "E"
        labels.append(lab)
        last_other[tc.team] = (lab, tc.t)
    return labels


# ---------------------------------------------------------------------------------------------
# Evaluating the rule on expert DataVolley files: perfect touch sequences, so any error is the rule's
# ---------------------------------------------------------------------------------------------

_DV_LETTER = {"Serve": "S", "Reception": "R", "Set": "E", "Attack": "A",
              "Block": "B", "Dig": "D", "Freeball": "F"}


def rallies_from_dvw(plays: Iterable[dict]) -> list[list[dict]]:
    """Group scouted touches by rally, in file order, dropping non-touch rows.

    DataVolley codes a reception/dig/free ball that goes straight to an attacker twice (R then a
    phantom E by the same player). Video sees one touch, so the phantom E is dropped.
    """
    by_rally: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for p in plays:
        letter = _DV_LETTER.get(p.get("skill") or "")
        if letter is None or p.get("rally_number") is None:
            continue
        key = (p.get("set_number"), p.get("rally_number"))
        if key not in by_rally:
            by_rally[key] = []
            order.append(key)
        team = (p.get("code") or "?")[0]
        rows = by_rally[key]
        if (letter == "E" and rows and rows[-1]["letter"] in ("R", "D", "F")
                and rows[-1]["team"] == team and rows[-1]["number"] == p.get("player_number")):
            continue
        rows.append({"letter": letter, "team": team, "number": p.get("player_number"),
                     "line": p.get("file_line_number")})
    return [by_rally[k] for k in order]


@dataclass
class RuleScore:
    total: int = 0
    correct: int = 0
    per_class: Counter = field(default_factory=Counter)
    per_class_correct: Counter = field(default_factory=Counter)
    confusion: Counter = field(default_factory=Counter)
    rallies: int = 0
    rallies_all_right: int = 0

    def acc(self, exclude: tuple[str, ...] = ()) -> float:
        n = sum(v for k, v in self.per_class.items() if k not in exclude)
        c = sum(v for k, v in self.per_class_correct.items() if k not in exclude)
        return c / n if n else float("nan")

    def class_acc(self, k: str) -> float:
        return self.per_class_correct[k] / self.per_class[k] if self.per_class[k] else float("nan")


def score_rule_on_rallies(rallies: list[list[dict]], use_block_flag: bool = True) -> RuleScore:
    """use_block_flag: treat 'touch at the net' as observable (video sees the player at the net).
    In expert files the only reliable proxy is the scouted B itself, so this is optimistic for blocks;
    set False to see how the rule does with no net information at all."""
    sc = RuleScore()
    for rows in rallies:
        if not rows:
            continue
        touches = [TouchIn(team=r["team"], t=float(i) * 0.4, at_net=use_block_flag and r["letter"] == "B")
                   for i, r in enumerate(rows)]
        pred = label_rally(touches, block_window_s=1.0)
        all_ok = True
        for r, p in zip(rows, pred):
            sc.total += 1
            sc.per_class[r["letter"]] += 1
            sc.confusion[(r["letter"], p)] += 1
            if r["letter"] == p:
                sc.correct += 1
                sc.per_class_correct[r["letter"]] += 1
            else:
                all_ok = False
        sc.rallies += 1
        sc.rallies_all_right += int(all_ok)
    return sc
