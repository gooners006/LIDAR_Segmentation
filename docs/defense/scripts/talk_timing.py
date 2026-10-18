"""Estimate how long each ★ slide's script takes to say (docs/defense/talk_script.md).

Vietnamese is written one syllable per word, so the VI length is the token count. The speaking
rates are ASSUMPTIONS, not measurements of the presenter: VI 3.0 / 3.5 / 4.0 syllables per second
(slow / middle / fast careful presentation), EN 130 words per minute. Replace them with the
presenter's own pace after the first timed rehearsal (--vi-rate). Flags slides whose middle
estimate is over budget by more than 10%.

Run:  .venv\\Scripts\\python.exe docs/defense/scripts/talk_timing.py [--vi-rate 3.5]
"""
import argparse
import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "talk_script.md"


def parse(path):
    """[(number, title, budget_s, en, vi)] from '### <n> <title> (<m:ss>)' blocks."""
    out, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"### (\d+) (.+) \((\d+):(\d\d)\)$", line)
        if m:
            cur = [int(m[1]), m[2], int(m[3]) * 60 + int(m[4]), "", ""]
            out.append(cur)
        elif cur and line.startswith("EN: "):
            cur[3] = line[4:]
        elif cur and line.startswith("VI: "):
            cur[4] = line[4:]
    return out


def tokens(s):
    return [t for t in re.split(r"\s+", s.replace("⚑", "")) if re.search(r"\w", t)]


def mmss(s):
    return f"{int(s) // 60}:{int(round(s)) % 60:02d}"


ap = argparse.ArgumentParser()
ap.add_argument("--vi-rate", type=float, default=3.5, help="syllables per second (middle estimate)")
args = ap.parse_args()
rows = parse(SCRIPT)
lo, hi = 3.0, 4.0
print(f"| # | Slide | Budget | VI syll. | VI at {args.vi_rate:g}/s | VI range ({hi:g}–{lo:g}/s) | EN words | EN at 130 wpm | |")
print("|---|---|---|---|---|---|---|---|---|")
tot = {"b": 0, "vi": 0, "lo": 0, "hi": 0, "en": 0}
for n, title, b, en, vi in rows:
    sv, we = len(tokens(vi)), len(tokens(en))
    t_vi, t_en = sv / args.vi_rate, we / (130 / 60)
    flag = "over" if b and t_vi > 1.1 * b else ""
    tot["b"] += b; tot["vi"] += t_vi; tot["lo"] += sv / hi; tot["hi"] += sv / lo; tot["en"] += t_en
    print(f"| {n} | {title} | {mmss(b)} | {sv} | {mmss(t_vi)} | {mmss(sv / hi)}–{mmss(sv / lo)} | {we} | "
          f"{mmss(t_en)} | {flag} |")
print(f"| | **Total** | **{mmss(tot['b'])}** | | **{mmss(tot['vi'])}** | {mmss(tot['lo'])}–{mmss(tot['hi'])} | | "
      f"{mmss(tot['en'])} | |")
