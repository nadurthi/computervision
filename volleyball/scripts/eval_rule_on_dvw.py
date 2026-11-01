"""Score the skill-from-touch-order rule on expert DataVolley files.

    python scripts/eval_rule_on_dvw.py                 # the 7 files bundled with py-datavolley
    python scripts/eval_rule_on_dvw.py my/*.dvw        # your own scouted matches
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vb.evaluate import score_rule_on_dvw_files  # noqa: E402

if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        import datavolley as dv

        files = dv.example_file_list()
    for flag in (True, False):
        r = score_rule_on_dvw_files(files, use_block_flag=flag)
        print(f"\n=== net position {'known' if flag else 'unknown'} ===")
        print(json.dumps(r, indent=1))
