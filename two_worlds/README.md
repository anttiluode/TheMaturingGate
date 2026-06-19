# Two Worlds

### A fast world-tracker and a slow decoupling student, joined by a coincidence gate that spends "spikes" only on surprise — a small numpy model, run against the obvious baselines, with the honest result

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, Juhannus 2026.**

> Do not hype. Do not lie. Just show.

---

## The one idea

Two networks watch the same stream. The **fast** one, `F`, learns the world quickly and takes the onslaught. The **slow** one, `S`, learns the same world more slowly and can do the thing `F` cannot: **decouple** — a single gain knob decides whether `S` eats the world (perception) or free-runs its own dynamics (imagination / replay). Between them sits a **coincidence gate**: per dimension, the two streams are compared, and where they *agree* the unit stays silent and passes the consensus on cheaply; where they *disagree* the gate opens, the world is let in, and the surprise is broadcast. A spike is what disagreement costs.

That gate is the buffer repo's coupling `K`, turned from a constant into a per-dimension decision:

```
y  <-  normalize( (1 - k) · ½(F·y + S·y)  +  k · x )
k_i =  clip( k0 + g·surprise_i + g·fault_i , 0, 1 )
       surprise_i = |pred_i − x_i|     # the prediction missed the world (a real spike)
       fault_i    = |f_i − s_i|        # the two models disagree (one is unreliable)
```

This is a runnable instance of a neuron that **sums and fires only if its two inputs coincide** — and it is pointed at the three questions a two-network story has to answer: does activity track competence, can one knob switch perception into imagination, and what does the *second* opinion actually buy when the first one breaks.

---

## Why this is more than metaphor (the grounding)

- **Coincidence detection.** A cortical pyramidal cell integrates feedforward drive on its basal dendrites and top-down context on its apical tuft, and fires hardest when the two *coincide* (Larkum 2013, BAC firing). "Fires if the teacher agrees" is, fairly literally, this. `F` is the apical teacher; the gate is the coincidence.
- **Predictive coding is a Kalman filter** (Rao & Ballard 1999) — the `(1−K)·predict + K·observe` feed is the standard form; here `K` is gated.
- **Inhibition that learns to predict and cancel** — the ELL negative image leaves only the surprise (Bell; Sawtell). The gate's "silent where agreed, spike where surprised" is the same shape.
- **World models** (Ha & Schmidhuber 2018) and **knowledge distillation** (Hinton 2015) — a fast model of the world, and a slow learner pulled along by it.
- **Geometry = frequency.** Each predictor is a linear operator; its action *is* the geometry (state → next state) and its eigenspectrum *is* the frequency view — the same matrix read two ways. That is the Koopman/DMD duality already built in the HKT line; "matrix ops as geometry, matrix ops as frequency, same thing" is exactly this.

The contribution is not any one of these. It is the **combination**: a two-model agreement signal used *both* as the broadcast gate *and* as the perceive/imagine knob. That combination is what is tested here.

---

## The unit, against a transformer unit

A transformer unit mixes a learned, fixed weight matrix with engineered attention over all positions; its activation does not depend on how well the input was predicted. This unit is a **coincidence-gated predictive cell**: it holds a (here linear) forward operator, compares its prediction to a second opinion and to the world, and **broadcasts only the residual** — so it goes quiet on what it has learned and loud on what it has not. Same input, very different economics: the transformer pays the same cost for the familiar and the novel; this pays for novelty.

---

## What the code shows (numbers are seeded and reproducible)

### E1 — activity follows competence

Spike fraction (mean gate `k`) as the two models learn, split predictable vs near-a-jump:

| trained | predictable `k` | near-jump `k` |
|---|---|---|
| 3% | 0.199 | 0.208 |
| 8% | 0.097 | 0.119 |
| 25% | 0.066 | 0.103 |
| 50% | 0.059 | 0.099 |
| 100% | 0.064 | 0.102 |

As the models learn to agree on the predictable world, the gate quiets there (~3× drop); on surprise it stays wide open (~1.6× the predictable rate once trained). Energy is spent on what is not predicted — a property a standard net's activation does not have.

### E2 — one knob, two modes

The **same** `S`, gate open, perceives the world at fidelity **0.968**. Gate shut (`k=0`), it free-runs its own dynamics from a real state and stays on-manifold before drifting:

| roll-out step | 1 | 10 | 30 | 60 |
|---|---|---|---|---|
| on-manifold (cos to a plausible true state) | 1.00 | 0.99 | 0.82 | 0.88 |

Perception and imagination from one weight set, switched by one scalar. (Modest claim: it is one operator, not two systems.)

### E3 — the handshake (the centerpiece, and the honest correction)

A confidently-wrong fast net (rank-lesioned `F`, the buffer's catastrophe) and what a second opinion buys. Frames to re-lock after a surprise:

| arm | recovery frames |
|---|---|
| 1. single model, fixed `K`, fast net healthy (the buffer) | 6.8 |
| 2. single model, fixed `K`, fast net **lesioned** (the buffer) | **61.0** |
| 3. single model, **adaptive** `K`, lesioned (fair baseline) | 8.0 |
| 4. two-world coincidence gate, lesioned | 1.0 |
| 5. …same, but the gate is **mistimed** by 6 frames | 9.6 |

Read 2 → 3 honestly: **the rescue is adaptive gain, not the second network.** A fixed-`K` single net smears recovery to 61 frames (the buffer result); but a *single* net that monitors its own miss and raises `K` already recovers in 8. I expected the second opinion to be necessary here, and the data says it is not — for recovery lag, self-monitoring is enough. The two-world gate is faster still (1.0) and degrades more gracefully (its consensus always has a healthy model to fall back on), but on *this* metric the margin over a self-monitoring single net is small. The second network earns its place in E2 (decoupling), not E3.

Row 5 is the handshake point: when the gate's *decision* arrives 6 frames late, recovery goes from 1 to ~10 — **the surprise arrives about as late as the gate does.** Latency in the comparison, not damage to either network, is enough to make a fast system feel slow and smeared. (This is a dynamical-systems statement about two coupled predictors and a delayed gate. It is not a model of any nervous system, any medication, or anyone's perception, and nothing here is calibrated to anything physical.)

---

## Quickstart

```bash
python two_worlds.py
```

Pure numpy, no GPU. Seeded and reproducible. Change the lesion, the gate gains, or the mistiming and watch the trade-offs move.

---

## The honest ledger

**Established (used, not claimed):** coincidence detection / BAC firing (Larkum 2013); predictive coding as Kalman filtering (Rao & Ballard 1999); predictive cancellation by inhibition — the ELL negative image (Bell; Sawtell); world models (Ha & Schmidhuber 2018); knowledge distillation (Hinton 2015); the Koopman/DMD geometry↔frequency duality (the HKT line).

**Verified in code (reproducible, seeded):**
- spike fraction on predictable input falls ~3× as the models learn, and stays high on surprise — activity tracks competence and is spent on the unpredicted;
- one operator `S` perceives (0.97) or, gate shut, free-runs a plausible continuation that holds ~10–20 steps then drifts — perception and imagination from one weight set and one knob;
- a fixed-gain single net is catastrophic under a confident-wrong lesion (61 frames); adaptive gain rescues it (8); the two-world gate is fastest (1) and degrades most gracefully;
- a 6-frame latency in the gate decision turns a 1-frame recovery into ~10 — the surprise arrives about as late as the gate.

**What did *not* pan out (the honest part):** my prior was that the second network would be what rescues the partial-lesion regime. It is not — adaptive gain on a single self-monitoring model already recovers (rows 3 vs 4). The distinctive value of the second network is the decoupling and the graceful consensus, not recovery speed. The experiment corrected the hypothesis, which is why it was worth running.

**Where standard AI still wins, plainly:** this is not a benchmark contender. Linear toy models on a toy world will not approach a transformer on raw capability, accuracy, or scale, and nothing here suggests otherwise. The axes where this design is *interesting* are different ones: energy proportional to surprise, a single knob for perceive-vs-imagine, an interpretable readout, and a **characteristic, biologically-suggestive failure mode** (latency-induced smear) rather than a silent degradation. Sell it there and only there.

**Honest limits:** linear models, one toy world, relative units, parameters chosen not measured, single seeded run per arm; "fast/slow network", "spike", "fault" are abstractions; the recovery numbers depend on the re-lock threshold and the gains, so treat magnitudes as relative.

**The bet (untouched):** that any of this is *experienced* rather than processed — that the decoupled, free-running `S` is *imagining* rather than iterating a matrix, or that a smeared late surprise is *felt* as a glitch rather than computed as one. The model locates the mechanism precisely. It does not touch the hard problem.

---

## Where it goes next

1. **Nonlinear units.** Replace the linear `F`,`S` with the per-unit collapsing-operator / Koopman cell from the HKT line, so the "neuron is a matrix that sums" is literal and the geometry↔frequency reading is live per unit.
2. **Put the gate on the clock.** The gate here is computed every frame; wire its strength to the theta phase (the HKT chandelier/medial-septum circuit) so it breathes, and ask whether a mistimed *rhythm* — not just a fixed latency — reproduces the E3 smear in a cleaner way.
3. **Run it on the HKT video stream.** Make `F` the fast video tracker and `S` the slow decoupling student literally, then test the human thing: decouple `S` mid-stream and have it attend a sub-region or an abstraction while `F` keeps eating the frame. Measure that spiking *rises* the moment attention leaves the easy prediction — the "watching a video while thinking about something else is harder" prediction, made testable.

---

## Lineage

The fifth thread in the Juhannus line — after **the maturing gate**, **the mirror gate**, **the buffer**, and **the evolved gate** — and a direct descendant of the HKT engine (the fast Koopman tracker, the chandelier/AIS gate, the theta clock). The buffer asked whether adaptive gain could rescue a partial lesion; this answers it, and finds the answer is yes but for a humbler reason than expected. The framing — two world-models, one fast and tracking, one slow and able to decouple, joined by a coincidence gate that spends energy only on surprise — is Antti Luode's, worked out across a long ride and a pier at Aurinkolahti; the experiment, the baselines, and this document were developed collaboratively with Claude (Opus 4.8). MIT.

*One network takes the world as it comes; the other learns to mime it and then, when it likes, looks away. They spend nothing on what they both expect and everything on what surprises them — and if the handshake between them runs late, the surprise arrives late too. Do not hype. Do not lie. Just show.*
