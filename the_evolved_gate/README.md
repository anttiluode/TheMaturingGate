# The Evolved Gate

### Don't hand-write the developmental schedule — let selection write it, with the stationary extremes available and watch what gets discarded

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.**

> Do not hype. Do not lie. Just show.

---

## The question the maturing gate left open

The maturing gate's headline was *the developmental schedule wins*. But the schedule was **hand-set** — `win_close = 0.5`, `pi_floor = 0.10`, the two `iota` values — and then the policy I designed beat the two extremes I also designed. A fair skeptic says: of course it did. The genomic-bottleneck claim (Zador 2019) is that **selection** discovers the developmental program because it is optimal — not that a designer writes it in. So the honest next step is to stop writing the schedule and let selection write it, with the stationary policies sitting in the search space as fully reachable, even **seeded into the initial population**. If selection climbs to a graded plastic→stable schedule, the thesis is shown. If it settles on always-plastic or born-mature, the thesis is wrong.

---

## The setup

The genotype is **6 scalars**: `[pi_early, pi_late, iota_early, iota_late, win_close, win_ramp]`. One schedule family,

```
close   = smoothstep((t − win_close)/win_ramp + 0.5)
pi(t)   = pi_early·(1−close)   + pi_late·close
iota(t) = iota_early·(1−close) + iota_late·close
```

**contains the stationary policies as corners** — always-plastic is `pi_early==pi_late` high with `iota` low; born-mature is `pi` low with `iota` high; developmental is `pi` high→low, `iota` low→high. Nothing is rigged: if a stationary policy were optimal, the search could reach it, and all three (always-plastic, born-mature, the v1 hand design) are placed in generation 0 by hand. Fitness is the same `C·E·A` combination the maturing gate used, in the dense overlapping world where all three axes discriminate. A tiny (μ+λ) evolution strategy optimises on one world seed; the champion is then **validated on five held-out seeds** (the honest generalisation number).

---

## What selection found

The stationary policies were in the population from the start, and selection discarded them:

| policy (validated, 5 seeds) | fitness | sd | C | E | A |
|---|---|---|---|---|---|
| always-plastic | 0.354 | 0.05 | 0.48 | 0.74 | 1.00 |
| born-mature | 0.162 | 0.33 | 0.85 | 0.88 | 0.20 |
| hand-dev (v1, my design) | 0.728 | 0.36 | 1.00 | 0.91 | 0.80 |
| **evolved** | **0.876** | **0.06** | **0.91** | **0.96** | **1.00** |

Selection rejected both stationary extremes and beat even my hand design. The schedule it found:

```
pi_early   0.98   ->   pi_late   0.57       plasticity drops, but only to a HIGH floor
iota_early 0.14   ->   iota_late 0.57       gate matures broad -> selective  (+0.43)
win_close  0.95   win_ramp 0.20             window closes late and gently
```

The gate maturing broad→selective (`iota 0.14 → 0.57`) is the maturing-gate signature, **selected rather than assumed**. The genomic-bottleneck claim holds in code: a 6-scalar genome encodes a developmental program because that program is what selection finds optimal, and the genome still contains **none of the world's concepts** — those are grown in the window it schedules.

---

## What selection *corrected* in the v1 hand design (the honest part)

This is the value of the test — it overturned a secondary assumption I had baked in:

1. **It keeps a high plasticity floor (0.57), not my 0.10 — and the two maturations decouple.** The v1 design tied plasticity-collapse and gate-maturation together into one mid-life "close." Selection pulls them apart: the **gate** matures (which buys efficiency `E` and competence against interference) while **plasticity stays high** (which buys late-life adaptability `A`). You do **not** have to spend plasticity to get efficiency — the gate does that job. That was the unexamined error in v1.
2. **The window closes late and gently** (`win_close = 0.95`, pinned near the bound): a slow continuous maturation across the whole life, not a sharp mid-life critical period.
3. **The evolved schedule is far more robust** — fitness sd **0.064** vs the v1 hand design's **0.364**. The high floor catches the brief late novelty on *every* seed; the low-floor v1 caught it only when the noise happened to align. Retaining plasticity is what buys reliability on the adaptability axis.

So the maturing-gate's core survives strongly (gate maturation beats both stationary extremes, discovered not assumed), and its *secondary* framing — that plasticity must collapse alongside the gate — is corrected: keep plasticity, mature the gate.

---

## Quickstart

```bash
python the_evolved_gate.py
```

Pure numpy, no GPU. Seeded and reproducible; prints the baselines, the per-generation best fitness, the evolved genome, and the verdict.

---

## The honest ledger

**Established (used, not claimed):** the genomic bottleneck — a compact innate program is selected, not a stored connectome (Zador 2019); critical-period plasticity gated by inhibitory maturation (Hensch 2005; Takesian & Hensch 2013); loss of plasticity under continual training unless managed (Dohare et al. 2024).

**Verified in code (reproducible, seeded):**
- given the stationary policies as seeded, reachable options, selection **discards** them (always-plastic 0.354, born-mature 0.162) and converges on a non-stationary schedule (0.876), beating the v1 hand design (0.728);
- the discovered schedule matures the gate broad→selective (`iota 0.14 → 0.57`) — the maturing-gate signature, selected rather than assumed;
- selection keeps a **high plasticity floor** and **decouples** gate-maturation from plasticity-collapse — a correction to the v1 design, and the source of a large robustness gain (sd 0.06 vs 0.36).

**Honest limits:**
- a tiny evolution strategy on a toy fitness; fitness optimised on **one** world seed then validated on five — generalisation is checked, but this is not a study of evolutionary dynamics;
- `win_close` pinned to the edge of its allowed range, so selection may want the window even later/gentler than the parameterisation permits — the bound, not the result, is the constraint there;
- relative units, parameters of the *arena* (spawn/recognition thresholds, budget, world overlap) still chosen; `gate`/`plasticity`/`salience` remain scalar abstractions of rich biology;
- the fitness is `C·E·A` with the arena's weighting — a different fitness (e.g. weighting efficiency harder) would move the optimum, and selection only ever finds what the fitness rewards.

**The bet (untouched):** that any of this is *experienced* rather than processed. Letting selection write the schedule removes the "I rigged it" objection from the maturing gate and turns the genomic-bottleneck claim into a runnable, falsifiable result. It does not touch the hard problem.

---

## Where it goes next

1. **Co-evolve the schedule with the world.** Here the world is fixed and the genome adapts to it. Let the world's statistics (overlap, novelty rate, how brief the late encounter is) vary, and ask which world features push `win_close` and the floor where — i.e. *which environments select for sharp critical periods vs lifelong plasticity.* That is the comparative-development question (altricial vs precocial) made testable.
2. **Let selection discover stacked windows.** Give the genome several gate populations each with their own `(pi, iota, win_close)` and let selection decide how many critical periods to open and when — does it rediscover the early-sensory / late-associative ordering?
3. **Evolve against the plasticity-loss wall directly.** Make the late task a *stream* of novelties rather than one, so retained-but-managed plasticity is forced to trade against interference continuously — the live continual-learning setting (Dohare et al.), with the schedule as the thing under selection.

---

## Lineage

The fourth in the Juhannus line, after **the maturing gate** (the schedule, hand-designed), **the mirror gate** (the fast net that teaches), and **the buffer** (the fast net lesioned). Here the maturing gate is put to its own test: the schedule it assumed is removed and selection is asked to find it. The framing — that the developmental program should be *selected*, not designed, and that the test must put the stationary extremes in selection's hands — is the contribution; the experiment and this document were developed collaboratively with Claude (Opus 4.8). MIT.

*The genome does not store the schedule any more than it stores the network — it is the thing selection lands on when stationary policies are cheap to reach and still lose. Given that freedom, selection matures the gate and keeps the plasticity. The window was not designed; it was earned. Do not hype. Do not lie. Just show.*
