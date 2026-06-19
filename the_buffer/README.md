# The Buffer

### Degrade the fast network, watch when the slow feed breaks — a two-network predictive video filter, and what the damage curve says about the relationship between them

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, Juhannus 2026.**

> Do not hype. Do not lie. Just show.

---

## The one question

If the fast (inhibitory, gamma) network is a learned forward model that eats the world's raw motion, and the slow (principal, theta) network's output is the smooth *feed* you actually experience — then **how much can the fast network be degraded before the feed breaks?** The shape of that curve characterizes the relationship between the two networks.

This is the companion experiment to [the mirror gate](../the-mirror-gate): there, the fast net *taught*; here, the fast net *buffers*, and we lesion it.

---

## The architecture (predictive coding is a Kalman filter)

A smooth latent video `z_t` (rotating in a few planes) with rare scene **jumps**; the observation `x_t = z_t + noise` is the raw, surprising world. Two networks:

- the **fast** network `F` is a learned forward model — it runs the Kalman *predict* step, `F·y`, eating the motion;
- the **slow** network's output is the filtered *feed*:

```
y_t = normalize( (1−K)·(F·y_{t−1})  +  K·x_t )
```

With a small `K` the feed mostly rides the fast prediction and lets only a fraction of the raw surprise in. That is the **buffer**: the fast net takes the hits; the feed lives on a smooth, predicted trajectory. Lesion `F` and the buffer fails. Grounding: predictive coding as Kalman filtering (Rao & Ballard 1999); learned predictive cancellation by inhibition — the ELL negative image (Bell; Sawtell).

---

## What the damage curve says

Healthy feed fidelity **0.987**, versus the raw world at **0.945** — the buffer genuinely denoises. Then we break `F` three ways.

**A. Fast-net noise — graceful, with a crossover.**

| noise level | feed fidelity | % of healthy |
|---|---|---|
| 0.00 | 0.987 | 100% |
| 0.50 | 0.962 | 97% |
| 1.00 | 0.885 | 90% |
| 2.00 | 0.598 | 61% |
| 3.00 | 0.321 | 33% |

The feed absorbs a lot of fast-net noise before it suffers — but note the **crossover**: it stays better than the raw world (0.945) only up to noise ≈ 0.5–0.7. Past that, a noisy fast net makes the feed *worse* than ignoring the prediction entirely.

**B. Fast-net simplification — and the confident-wrong cliff.**

| rank kept | feed fidelity | % of healthy |
|---|---|---|
| 6 (full) | 0.987 | 100% |
| 4 | 0.848 | 86% |
| 2 | 0.595 | 60% |
| 1 | 0.206 | 21% |
| 0 (offline) | 0.945 | 96% |

`F` uses only **6 of its 48 dimensions**, so 42 can be dropped with no loss. But each real plane dropped below 6 costs the feed — and a rank-1 stub (0.21) is **worse than no fast net at all** (0.95). A confidently-wrong predictor is worse than a silent one (the same lesson as the wrong-teacher in the mirror gate).

**C. Recovery lag after a surprise — the richest result.**

| fast-net state | frames to re-lock |
|---|---|
| healthy | 4.7 |
| noisy ×1.0 | 6.3 |
| noisy ×2.0 | never re-locks |
| **rank 2 (partial damage)** | **79.9** |
| offline (F = 0) | 1.0 |

Read this carefully, because the ordering is not "healthy fast, broken slow":

- the **healthy** buffer takes ~5 frames for a surprise to propagate into the feed — it *deliberately smooths the hit*. That short delay is the buffer doing its job: the smooth feed is bought by not reacting instantly.
- the **offline** net reacts in 1 frame, but only ever copies the raw, un-denoised world (stuck at 0.945, never the healthy 0.987). No buffer, no smoothing, no denoising — just the raw signal, immediately.
- the **partially damaged** net (rank 2) is the worst case by far: ~80 frames. A fast net that confidently *mispredicts* fights the correction, so the surprise reaches the feed late and smeared — much worse than cleanly removing it.

So the relationship, in three numbers: **a healthy fast net buys a denoised, anticipated feed at the cost of a short deliberate surprise-delay; removing it cleanly costs the denoising and the anticipation but keeps instant raw reactions; partially breaking it is the dangerous regime — confident misprediction makes surprises arrive late and smeared.**

---

## Quickstart

```bash
python the_buffer.py
```

Pure numpy, no GPU. Seeded and reproducible.

---

## The honest ledger

**Established (used, not claimed):** predictive coding as Kalman filtering (Rao & Ballard 1999); inhibition that learns to predict and cancel expected input (ELL negative image — Bell; Sawtell).

**Verified in code (reproducible, seeded):**
- a two-network predictive filter denoises the feed above the raw observation (0.987 vs 0.945);
- the feed degrades *gracefully* under fast-net noise, with a crossover (~0.5–0.7) past which the prediction hurts more than it helps;
- the fast model is low-rank (6 of 48); it compresses losslessly to its true rank, then degrades, and a rank-1 stub is worse than no model at all;
- recovery lag is non-monotonic in damage: healthy ≈ 5 frames (deliberate smoothing), offline = 1 frame (raw, instant, un-denoised), partial damage ≈ 80 frames (pathological) — partial lesion is worse than clean removal.

**Honest limits:**
- linear dynamics, one toy video, a single fixed gain `K`, relative units, chosen parameters;
- "fast/slow network" are linear abstractions; the recovery-lag numbers depend on the re-lock threshold (0.85) and on `K`, so treat the magnitudes as relative, not absolute;
- the rank/noise lesions are crude stand-ins for biological damage; this is a dynamical-systems statement about two coupled predictors, not a model of any specific injury.

**The bet (untouched):** the "feed" here is an estimate vector, not a felt one. That a smooth, buffered estimate is what gets *experienced* — and that a partial lesion would feel like a surprise arriving late and misrouted rather than on time — is exactly the question this does not answer. It stays in its drawer.

---

## Where it goes next

1. **Adaptive gain.** Let `K` rise when the fast prediction is unreliable (the crossover) — the system noticing its own fast net is damaged and leaning back on raw sense. Does adaptive `K` rescue the partial-damage regime?
2. **Two real clocks.** Run `F` at gamma and the feed at theta as genuinely nested rates, so the feed integrates a theta-cycle of fast predictions rather than one per frame.
3. **Localized lesions.** Damage `F` on *some* planes only (some scene features predicted, others not) and watch the feed track the intact features while smearing the lesioned ones — closer to how real damage is partial and feature-specific.

---

## Lineage

Third in the Juhannus line, after **the maturing gate** (the inhibitory gate on a developmental schedule) and **the mirror gate** (the fast net that teaches the slow one). Here the same two networks are coupled as a predictive filter and the fast one is lesioned, to read off their relationship from the damage curve. The framing — fast net as buffer that eats the world's surprise so the slow feed stays smooth — is the contribution; the experiment and this document were developed collaboratively with Claude (Opus 4.8). MIT.

*A healthy fast network lets the world arrive already predicted, and passes the slow feed only what it could not foresee. Break it a little and the surprises come late and smeared; remove it cleanly and they come raw but on time. Do not hype. Do not lie. Just show.*
