# The Mirror Gate

### A fast inhibitory network that learns the expected world and teaches the slow principal network — distillation, predictive cancellation, and why the teacher must be a mirror

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, Juhannus 2026.**

> Do not hype. Do not lie. Just show.

---

## The one idea

The fast inhibitory network — the basket and chandelier cells running at gamma, ahead of the slower theta-paced principal cells — is not just a brake. It is a **learned forward model of the expected input**: a *mirror* of the world's structure. Its job is to **teach** the slow principal network. It pulls the slow learner up faster, and once it has matured it **cancels** what it can predict, so the principal network only has to spend capacity on the *surprise*.

This repo puts that on a clock and tests it three ways — including the one place it could fail, and the place it ties back to [the maturing gate](../the-maturing-gate).

---

## Why this is more than a metaphor

- **Inhibition really does run faster.** Fast-spiking PV⁺ basket and chandelier interneurons generate gamma through local loops, while principal cells are paced slower by theta (Buzsáki; PING/ING models). A fast network holding a higher-rate version of the computation is real.
- **Inhibition that learns to predict and cancel is real.** The electric-fish ELL grows a learned **negative image** that subtracts the predicted sensory consequence, leaving only the surprise (Bell; Sawtell). Cortical predictive coding makes the same proposal (Rao & Ballard 1999; Attinger et al. 2017). Inhibition *is* an expectation, subtracted.
- **A fast net teaching a slow net is knowledge distillation** (Hinton et al. 2015) — established machine learning.
- **Inhibitory maturation closes plasticity windows** (Hensch 2005) — the schedule, from the companion repo.

**The hypothesis (the new bridge, not established):** that this fast inhibitory predictor is a learned mirror/world-model whose *role* is to teach the principal network, and that gating / "voting" (the chandelier veto; the Thousand-Brains voting that silences mismatched hypotheses) is this expectation silencing the unexpected. Tested here in a toy, and honest about being a toy.

---

## What the code shows

A slow principal network (the *student*: sparse, slow plasticity — it updates on only ~4% of steps, like slow pyramidal plasticity) learns to predict the next pattern in a mostly-predictable stream (a cyclic tour with rare surprise jumps). A fast inhibitory *mirror* (the teacher: updates every step) optionally teaches it.

**Demo 1 — the mirror teaches, and it must be a real mirror.**

| arm | early acc | final acc | steps → 0.55 |
|---|---|---|---|
| alone (no teacher) | 0.65 | 0.78 | 508 |
| **mirror teacher** | **0.71** | 0.79 | **401** |
| WRONG teacher (confident non-mirror) | 0.16 | 0.17 | 7999 (never) |

The fast mirror reaches competence sooner and scores higher early, because its expectation is *denoised* — it has averaged many samples the slow student never gets to see. But a confident teacher that is **not** a mirror of this world (trained on the wrong transitions) is catastrophic: the student inherits its errors. The teacher helps only when it mirrors the actual world.

**Demo 2 — when the mirror is born wrong, schedule the trust** (the link to the maturing gate).

| teacher born with a wrong innate prior | early | final | steps → 0.55 |
|---|---|---|---|
| trust from birth (fixed) | 0.66 | 0.79 | 582 |
| **trust on a schedule (maturing)** | 0.67 | 0.79 | **508** |

If the mirror starts with a **miscalibrated innate prior** it has to outgrow (mirror neurons are not perfect; innate priors are wrong before they are corrected), then trusting it from birth actively *slows* the learner (582 steps vs the blank-teacher's 401). The maturing schedule withholds trust until the prior is corrected, and stays clean (508). **The developmental schedule earns its keep exactly when the early world-model is confidently wrong** — which is the realistic case. The magnitude is small in a linear toy; it grows the more strongly wrong the early model is.

**Demo 3 — once matured, the mirror cancels the expected, leaves the surprise.**

| mean residual the principal must still encode | predictable step | surprise step |
|---|---|---|
| no teacher (raw input) | 1.00 | 1.00 |
| matured mirror | **0.46** | **1.26** |

On predictable steps the residual collapses (**54% cancelled**); on surprises it stays large — it even *grows*, because the mirror confidently predicted the wrong thing. This is the ELL negative-image, in code: the principal network downstream only ever sees the **surprise**, which is sparse. That is the energy argument and the predictive-coding argument in one number.

---

## Quickstart

```bash
python the_mirror_gate.py
```

Pure numpy, no GPU. Seeded and reproducible.

---

## The honest ledger

**Established (used, not claimed):** fast PV⁺ interneurons generate gamma ahead of theta-paced principal cells (Buzsáki); learned inhibition can predict and cancel expected input — the ELL negative image (Bell; Sawtell) and cortical predictive coding (Rao & Ballard 1999; Attinger 2017); knowledge distillation (Hinton 2015); inhibitory maturation closes plasticity windows (Hensch 2005).

**Verified in code (reproducible, seeded):**
- a fast mirror teacher reaches threshold sooner (401 vs 508 steps) and scores higher early (0.71 vs 0.65) than learning alone — its denoised expectation accelerates a slow, sparsely-plastic learner;
- a confident **non-mirror** teacher is catastrophic (final 0.17) — the teacher helps only as a mirror of *this* world;
- when the mirror is born with a wrong innate prior, **fixed** trust slows learning (582 steps) while **maturing** trust does not (508) — scheduling the trust protects against a miscalibrated early model;
- a matured mirror cancels 54% of the predictable residual while the surprise residual stays large — the principal encodes only surprise.

**Honest limits:**
- linear next-step predictors, one toy world, single seed per arm, relative units, chosen parameters;
- the maturing-trust effect (Demo 2) is **small** in a linear toy — a blank teacher is never confidently wrong, so the schedule only bites against an actively-wrong prior, and weakly here; the claim is directional, not a large effect;
- "teacher", "student", "mirror" are linear/scalar abstractions of rich biology; the mirror-neuron and Thousand-Brains-voting bridges are *interpretation*, not demonstrated;
- distillation here helps via *denoising* (the teacher averaged more samples), which is one real mechanism, not proof that biology uses this one.

**The bet (untouched):** that any of this is *experienced* — that the felt expectation of "how the world will be next" is anything more than a subtracted prediction. A fast mirror that teaches and cancels locates the mechanism; it does not touch the hard problem.

---

## Where it goes next

1. **Strongly-wrong priors.** Push Demo 2 where it bites: a teacher with a confidently-wrong, slow-to-correct innate prior, and watch the maturing schedule's advantage grow — the regime real development actually lives in.
2. **Wean the student off the teacher.** Add the other half of the schedule: trust the mirror while you are worse than it ("trying to become"), then *reduce* trust once you match it ("has become"), so the principal refines on truth instead of tracking a noisier fast net.
3. **Two clocks, explicitly.** Run the teacher at gamma and the student at theta as actual nested rates (the principal integrates over a theta cycle of fast mirror predictions), instead of the rate gap being only a learning-rate / update-sparsity difference.
4. **Cancellation → real sparsity downstream.** Feed the surprise residual into a principal coder and measure the active fraction directly, joining this to the energy axis of the maturing gate.

---

## Lineage

Companion to **the maturing gate** (the inhibitory gate on a developmental schedule) and to the **PredictiveHKT** teacher–student video line. The framing — that the fast inhibitory network is a learned *mirror* of the world whose job is to teach the slow principal network, and that its trust must be scheduled because the mirror is born wrong — is the contribution; the demos and this document were developed collaboratively with Claude (Opus 4.8). MIT.

*The fast network learns the world before the slow one does, and lends it the answer. But only if it is a true mirror, and only once it has earned the trust. Once matured, it says nothing about what it already expected — it passes on only the surprise. Do not hype. Do not lie. Just show.*
