"""Write events as a DataVolley .dvw file.

Only the sections and columns that readers rely on are filled. Scout rows follow the layout of
DataVolley 4 files (semicolon-separated):

    0 code ; 1 point phase ; 2 attack phase ; 3 - ; 4 start coord ; 5 mid coord ; 6 end coord ;
    7 time hh.mm.ss ; 8 set ; 9 home setter pos ; 10 away setter pos ; 11 - ; 12 video seconds ;
    13 - ; 14..19 home rotation ; 20..25 away rotation

Code: team ('*' home, 'a' away), 2-digit shirt number ('00' unknown), skill letter, skill type
('H' high/default), evaluation. The pipeline does not judge quality yet, so evaluation defaults to
'+' for continued play and '#'/'=' only where the rally's last touch decides the point.
Coordinates are DataVolley's grid index (see court_to_dv_coord), mapped from court metres.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

from .rally import Rally

UNKNOWN = "00"


def court_to_dv_coord(x_m: float, y_m: float, home_at_y0: bool = True) -> str:
    """Court metres -> DataVolley coordinate index (4 digits).

    Inverse of datavolley.dv_index2xy: idx = (row - 1) * 100 + col, with
    x_dv = 0.5 + 3 (col - 10.5) / 80 and y_dv = 0.5 + 6 (row - 10.5) / 81, where the court spans
    x_dv 0.5..3.5 and y_dv 0.5..6.5 (net at 3.5). DataVolley draws the home team in the lower half,
    so if the home team is at our Y = 18 end the court is rotated by 180 degrees first.
    """
    if not home_at_y0:
        x_m, y_m = 9.0 - x_m, 18.0 - y_m
    col = int(round(10.5 + 80.0 * x_m / 9.0))
    row = int(round(10.5 + 81.0 * y_m / 18.0))
    col = min(max(col, 1), 100)
    row = min(max(row, 1), 101)
    idx = (row - 1) * 100 + col
    return f"{idx:04d}"


@dataclass
class DvwMatch:
    home: str = "Home"
    away: str = "Away"
    date: str = "01/01/2026"          # mm/dd/yyyy
    home_numbers: list[int] = field(default_factory=list)
    away_numbers: list[int] = field(default_factory=list)


def _team_code(side: int, near_is_home: bool) -> str:
    return "*" if (side == 0) == near_is_home else "a"


def _num(n: int | None) -> str:
    return UNKNOWN if n is None else f"{int(n):02d}"


def rally_rows(r: Rally, set_no: int, near_is_home: bool, score: list[int], video_offset_s: float = 0.0) -> list[str]:
    rows: list[str] = []
    rot = ";" * 11                     # columns 14..25 left empty (rotations unknown)
    for i, t in enumerate(r.touches):
        team = _team_code(t.side, near_is_home)
        ev = "+"
        last = i == len(r.touches) - 1
        if last and t.skill == "A":
            ev = "#"
        code = f"{team}{_num(t.number)}{t.skill}H{ev}"
        start = court_to_dv_coord(*t.court_xy, home_at_y0=near_is_home) if t.court_xy else ""
        clock = _dt.timedelta(seconds=int(t.t + video_offset_s))
        hh, rem = divmod(int(clock.total_seconds()), 3600)
        mm, ss = divmod(rem, 60)
        rows.append(f"{code};;;;{start};;;{hh:02d}.{mm:02d}.{ss:02d};{set_no};;;;{int(t.t + video_offset_s)};;{rot}")
    if r.winner_side in (0, 1):
        w = _team_code(r.winner_side, near_is_home)
        if w == "*":
            score[0] += 1
        else:
            score[1] += 1
        rows.append(f"{w}p{score[0]:02d}:{score[1]:02d};;;;;;;;{set_no};;;;{int(r.end_t + video_offset_s)};;{rot}")
    return rows


def infer_winner(r: Rally) -> int:
    """Crude point winner: the side that did NOT make the last touch wins, unless the last touch
    was an attack (assumed kill). Real outcomes need the ball's landing spot; see the blog."""
    if not r.touches:
        return -1
    last = r.touches[-1]
    if last.skill == "A":
        return last.side
    return 1 - last.side


def write_dvw(path: str | Path, rallies: list[Rally], match: DvwMatch, set_no: int = 1,
              near_is_home: bool = True) -> Path:
    path = Path(path)
    now = _dt.datetime.now().strftime("%m/%d/%Y %H.%M.%S")
    lines = [
        "[3DATAVOLLEYSCOUT]", "FILEFORMAT: 2.0", f"GENERATOR-DAY: {now}", "GENERATOR-IDP: ", "GENERATOR-PRG: vb",
        "GENERATOR-REL: ", "GENERATOR-VER: ", "GENERATOR-NAM: volleyball pipeline", f"LASTCHANGE-DAY: {now}",
        "LASTCHANGE-IDP: ", "LASTCHANGE-PRG: ", "LASTCHANGE-REL: ", "LASTCHANGE-VER: ", "LASTCHANGE-NAM: ",
        "[3MATCH]", f"{match.date};00.00.00;;;;;;1;;1;;Z;", ";;;;;;;;",
        "[3TEAMS]", f"1;{match.home};0;;;;", f"2;{match.away};0;;;;",
        "[3MORE]", ";;;;;;", ";;;",
        "[3COMMENTS]",
        "[3SET]", "True;;;;;;", "True;;;;;;", "True;;;;;;", "True;;;;;;", "True;;;;;;",
        "[3PLAYERS-H]",
    ]
    for k, n in enumerate(sorted(set(match.home_numbers))):
        lines.append(f"0;{n};{k + 1};;;;;;H{n:02d};Player{n};P;H{n};;;False;;;")
    lines.append("[3PLAYERS-V]")
    for k, n in enumerate(sorted(set(match.away_numbers))):
        lines.append(f"1;{n};{k + 1};;;;;;V{n:02d};Player{n};P;V{n};;;False;;;")
    lines += ["[3ATTACKCOMBINATION]", "[3SETTERCALL]", "[3WINNINGSYMBOLS]",
              "=~~~#~~~=~~~~~~~=/~~#~~~=/~~#~~~~~~~~~~~=/~~~~~~=~~~~~~~", "[3RESERVE]", "[3SCOUT]"]
    score = [0, 0]
    for r in rallies:
        if r.winner_side not in (0, 1):
            r.winner_side = infer_winner(r)
        lines += rally_rows(r, set_no, near_is_home, score)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def check_with_datavolley(path: str | Path):
    """Parse the file with py-datavolley; returns (plays, problems). Raises if the reader fails."""
    import datavolley as dv

    plays = dv.read_dv(str(path))
    problems = []
    for p in plays:
        code = p.get("code") or ""
        if len(code) >= 4 and code[3] in "SREABDF" and p.get("skill") is None:
            problems.append(f"line {p.get('file_line_number')}: skill not decoded for {code}")
    return plays, problems
