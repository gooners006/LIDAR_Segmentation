# T9c verdict — seq-00 held-out replication

Date: 2026-08-09. Fresh judge session (executor/judge separation). Inputs read:
`docs/plans/preregistration_heldout.md` and
`docs/plans/t9b_results_heldout_seq00.md` **only**. No constants changed; no
retuning; no other project state read. Refutation criteria applied verbatim.

## Verdict: **PARTIALLY HOLDS** (coverage caveat, not metric weakness)

Both primary metrics are clean, significant wins over raw. The downgrade from
HOLDS is **purely a coverage gap** — the long band (≥4.6 m) is empty on seq 00
(0 cars), so the pre-registration's long-band predictions and long-band d2 check
are untestable. This is **not** a case of a weak or non-significant primary
metric; both primaries passed decisively.

Routed here by the taxonomy's PARTIALLY HOLDS clause: "reported with the caveat
named." The named caveat is the empty long band.

## R1 — does not generalize? NOT triggered

R1 fires if completed fails to beat raw on BEV IoU **or** on donor cov@0.1
(median, Wilcoxon p < .05). Both are wins:

- **BEV IoU (ALL, n=45):** raw 0.739 → completed 0.766, **p=1.6e-3** (< .05).
- **Donor cov@0.1 (primary τ=0.15):** raw 0.000 → completed 0.413,
  **p≈0** (< .05); ordering completed ≫ mirrored ≫ raw at every τ.

Neither refutation condition met. R1 does not fire.

## R2 — 08-specific length constants? NOT triggered (verbatim)

R2 fires if any band's completed/per-band-mirrored d2 ratio **exceeds its seq-08
level**, or a band that **passed on 08 fails on 00**.

| band (n) | seq-00 completed/mirrored | seq-00 ratio | seq-08 ref ratio | exceeds 08? | passed on 08? | pass-bit on 00 |
|----------|---------------------------|--------------|------------------|-------------|---------------|----------------|
| compact <3.6 (17) | 0.0090 / 0.0050 | 1.8× | ≈16× (0.0065/0.0004) | No | No (failed 08) | False |
| normal (28) | 0.0003 / 0.0144 | 0.02× | ≈0.12× (0.0015/0.0129) | No | Yes | True |
| long ≥4.6 (0) | — | — | 0× (0.0000/0.0292) | untestable | Yes | N/A (no cars) |

- No testable band's ratio exceeds its seq-08 level.
- The only band failing its pass-bit on 00 (compact, `False`) is a **smaller**
  violation than on 08 (1.8× vs ≈16×), and compact did **not** pass on 08 — so
  neither R2 clause applies to it.
- Normal passed on 08 and still passes on 00.

Per the verbatim R2 wording, **R2 does not trigger.** The taxonomy's "no R2 d2
regression" condition for HOLDS is therefore satisfied on the testable bands.
(The raw compact pass-bit is `False` and is surfaced for transparency, but it is
not an R2 regression as defined.)

## R3 — uncovered result? TRIGGERED by the empty long band

R3 lists "band Ns too small to test" as an explicit uncovered-result condition.
The **long band has n=0 cars** (seq-00 well-observed set is all compact/normal).
This makes the pre-registration's long-band directional predictions and its
long-band d2 hallucination check untestable on seq 00.

Escalated to the user with the tables rather than improvised past (per the
pre-registration and the delegate brief). User decision (2026-08-09): resolve via
the taxonomy's PARTIALLY HOLDS "caveat named" clause — **Option B**. Verdict
downgraded from HOLDS to PARTIALLY HOLDS with the empty long band as the named
caveat. Constants were **not** retuned; no re-run required for this verdict.

## Caveat wording (for citation in the thesis / findings)

> On the seq-00 held-out replication, completion beats the raw partial on **both**
> primary metrics with clear significance — BEV IoU 0.739 → 0.766 (Wilcoxon
> p=1.6e-3) and donor cov@0.1 0.000 → 0.413 (p≈0) — and no R2 length-constant
> regression among testable bands. The verdict is **PARTIALLY HOLDS solely
> because seq 00 contains no long (≥4.6 m) well-observed static cars**, leaving
> the long-band predictions and the long-band hallucination guard untestable on
> this sequence. The "partial" reflects a **coverage gap, not a weak metric**:
> both primaries were decisive wins.

## Tier-3 gate

T9c outcome is PARTIALLY HOLDS → Tier 3 / T13 is **permitted** to proceed
(gate requires HOLDS or PARTIALLY HOLDS).
