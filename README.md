# Grown Gates

EDIT: Added Two worlds. 

### The brain's fast inhibitory network, four small numpy studies: a gate grown on a developmental schedule, teaching the slow network, lesioned to read the relationship, and written by selection rather than by hand

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, Juhannus 2026.**

> Do not hype. Do not lie. Just show.

---

## The one idea

Every study here is about the same pair of networks: a **fast inhibitory network** (the basket and chandelier cells, running ahead at gamma) sitting over a **slow principal network** (the held content, paced by theta). The fast one is not just a brake. It is a learned model of the expected world that **gates** what the slow one is allowed to broadcast — a veto at the trigger zone. The four folders ask four questions about that pair, each in self-contained, pure-numpy, seeded code:

| folder | the move | the headline result (verified, relative units) |
|---|---|---|
| [`the_maturing_gate/`](the_maturing_gate/) | **grown** — put the gate on a developmental schedule (plastic+broad → stable+selective, with a floor) | the schedule beats both stationary extremes on competence × efficiency × adaptability (0.87 vs 0.46 vs 0.00 in the dense world); the matured gate fires ~3× fewer units |
| [`the_mirror_gate/`](the_mirror_gate/) | **teaches** — the fast net as a learned *mirror* of the world that distils into the slow one, then cancels what it predicts | a true mirror reaches competence sooner (401 vs 508 steps); a confident **non**-mirror is poison (0.17); a matured mirror cancels 54% of the predictable, leaving only surprise |
| [`the_buffer/`](the_buffer/) | **breaks** — couple the two as a predictive filter and lesion the fast net | the feed denoises (0.99 vs 0.95 raw); **partial** damage is far worse than clean removal — a confidently-mispredicting fast net makes a surprise arrive ~80 frames late and smeared, vs 1 frame if simply removed |
| [`the_evolved_gate/`](the_evolved_gate/) | **earned** — stop hand-writing the schedule; let selection write it, with the stationary policies seeded in as options | given the freedom to be stationary, selection discards both extremes and finds a developmental schedule (0.88), beating even the hand design (0.73) — and it *corrects* the hand design |

---

## The thread that runs through all four

Two findings recur across the studies, and together they answer the question that started the whole line — *why does an inhibitory cell get a veto over a principal cell's output?*

**Inhibition is for stability; plasticity is for adaptability; and you do not have to trade one for the other.** This is the correction the evolved gate handed back. The hand-designed maturing gate followed the standard machine-learning instinct — to protect what you have learned, lower the learning rate, freeze the weights. Selection, given the search space, refused that compromise. It **matured the gate** (inhibition `iota` 0.14 → 0.57, the chandelier turning selective) while **keeping the synapses plastic** (plasticity floor 0.57, not the hand design's 0.10), and it decoupled the two. The reading is clean: a strict, matured gate silences the interfering noise *before it reaches the synapses*, so the network does not need frozen weights to be stable — and because it stays plastic, it catches a rare late novelty the instant the gate opens for it. Inhibition for stability, synapses for adaptability. The chandelier's veto is not a brake on learning; it is what *lets* learning stay on.

**A confidently-wrong fast network is worse than no fast network at all.** This one shows up twice, independently. In the mirror gate, a confident teacher that is not a true mirror of this world drags the student below where it would have gone learning alone (0.17 vs 0.78). In the buffer, a partially-lesioned fast net that confidently mispredicts fights the correction and routes surprise to the slow feed late and smeared — far worse than a fast net cleanly switched off. A silent gate costs you the denoising and the head-start; a *wrong* gate actively corrupts. Whatever the fast network is, its competence — its being a true mirror — is load-bearing, and a broken one is a hazard, not merely a loss.

---

## How they chain

The maturing gate establishes the schedule; the mirror gate says what the fast network the schedule matures is *for* (a teacher and a predictive canceller); the buffer couples those two networks and lesions the fast one to read their relationship off the damage curve; and the evolved gate puts the maturing gate to its own test — removing the hand-set schedule and asking selection to find it, which both confirms the core claim and overturns a secondary assumption. Read in order they go: grown → teaches → breaks → earned.

---

## Why this is more than metaphor (the shared grounding)

Each piece leans on an established result, cited in its own folder, not on a guess:

- the **genomic bottleneck** — a compact innate program is *selected* and unfolds into the network; it cannot store the wiring (Zador 2019);
- **critical-period plasticity** opens and closes, and the closing is gated by the maturation of inhibitory (PV⁺ basket) interneurons and their perineuronal nets; chandelier→AIS connections are themselves developmentally plastic (Hensch 2005; Takesian & Hensch 2013; Qi et al. 2024);
- **loss of plasticity** under continual training, unless plasticity is *managed* on a schedule (Dohare et al. 2024, *Nature*);
- **predictive cancellation by inhibition** — the electric-fish ELL grows a learned "negative image" that subtracts the predicted input and leaves the surprise (Bell; Sawtell); cortical predictive coding makes the same proposal (Rao & Ballard 1999);
- **knowledge distillation** — a fast net teaching a slow one (Hinton et al. 2015).

The contribution is not any of these facts; it is wiring them into one runnable picture — a fast inhibitory mirror, grown on a developmental schedule, that teaches and gates a slow principal network — and then testing that picture hard enough to let it fail or correct itself.

---

## Quickstart

Each folder is independent, pure numpy, no GPU:

```bash
python the_maturing_gate/the_maturing_gate.py
python the_mirror_gate/the_mirror_gate.py
python the_buffer/the_buffer.py
python the_evolved_gate/the_evolved_gate.py
```

Everything is seeded and reproducible. Change the genome scalars, the world overlap, the lesion, or the fitness and watch the trade-offs move.

---

## The honest ledger (across all four)

**Verified in code (reproducible, seeded):**
- a developmental schedule resolves the plasticity–stability dilemma — it holds what it learned, stays sparse, and still catches a brief late novelty, where both stationary extremes fail one of those;
- the matured gate's energy advantage (≈3× fewer units fired) is real but **overlap-dependent** — it only appears once concepts overlap, and the sparse-world control shows no gap;
- a fast mirror accelerates a slow learner only when it is a true mirror of *this* world; a confident non-mirror is catastrophic;
- partial lesion of a coupled fast predictor is worse than clean removal;
- given the stationary policies as seeded options, selection discards them and finds a developmental schedule — and decouples gate-maturation (stability) from plasticity-retention (adaptability), a correction to the hand design with a large robustness gain.

**Honest limits — read these before believing any of it:**
- everything is in **relative units**, on **toy** worlds, with **chosen** parameters and few seeds. These show that an *architecture* produces an effect; they do not report a brain's or a model's measured numbers, and nothing here is calibrated to Joules, nats, or any benchmark;
- "the evolved schedule beats standard AI strategies" is a statement about *this toy fitness arena* with stationary policies seeded in — not a claim against any production system. Selection only ever finds what the fitness rewards;
- "gate", "plasticity", "salience", "fast/slow network" are scalar/linear abstractions of rich biology; the mirror-neuron and Thousand-Brains-voting readings are interpretation, not demonstration;
- the buffer and mirror are linear; the maturing/evolved gates are content-addressable toys — each is a dynamical-systems statement about the *shape* of the effect, not a model of any specific circuit or injury.

**The bet (untouched, as everywhere in this line):** that any of this is *experienced* rather than processed. These studies locate the developmental and inhibitory trade-offs precisely, in code that can fail. They do not touch the hard problem.

---

## Lineage

Built on the geometric-neuron / Mycelial Cortex line (PerceptionLab): the inhibitory gate as the thing that decides what may broadcast, here taken on its own and asked four questions. The spark was a single intuition — that the fast inhibitory network is a learned mirror of the world that the slow network is trying to become, which is why the chandelier cell holds a veto over the principal cell's output. The framing and the direction are Antti Luode's; the four demos and their documents were developed collaboratively with Claude (Opus 4.8). MIT.

*The genome does not write the network, or even the schedule — it writes the thing selection lands on. A window opens, the world is let in, inhibition matures to keep what was learned, and the synapses stay plastic so there is still room to learn. Grown, not given. Do not hype. Do not lie. Just show.*
