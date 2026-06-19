# The Maturing Gate

EDIT: Added mirror gate. The idea that the inhibitory network running faster is the mirror network of brain that learns the world model 
and the 'normal network' is what it is trying to become. Explaining why chandelier neurons have a veto on the signal output of normal 
neurons. Just a thought. 

### Growing a network on an innate developmental schedule — critical-period plasticity, inhibitory maturation, and the plasticity–stability compromise

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.**

> Do not hype. Do not lie. Just show.

---

## The one idea

A competent network is not specified. It is **grown** — on a small innate program that runs a *schedule*: the inhibitory gates start broad and the system plastic (an infant that learns everything and depends on an external reward to tag what matters), then inhibition matures, the gates turn selective, and the learning window closes — while a floor of plasticity is kept for the rest of life.

The genome cannot store the wiring. It stores the **schedule** and a small **prior**, and the world fills in the rest during an open window. This repo puts that on a clock and shows, in runnable code, that the schedule beats both fixed extremes on the combination of goals a lifelong system actually needs.

---

## Why this is more than a metaphor

Each piece is an established result, not a guess:

- **The genomic bottleneck** (Zador 2019). The genome is far too small to hold the connectome; it encodes a compact *program* that unfolds into the network through development. *Grown, not written down.* In the demo the innate program is **136 numbers** and it does not contain a single one of the world's concepts.
- **Critical periods** (Hensch 2005; Takesian & Hensch 2013). Plasticity windows **open and close**, and the closing is gated by the maturation of inhibitory (PV⁺ basket) interneurons and the perineuronal nets around them. *Inhibition maturing is what ends the window.* Axo-axonic (chandelier) → AIS connections are themselves developmentally plastic (Qi et al. 2024, "Specific and Plastic").
- **Loss of plasticity** (Dohare et al. 2024, *Nature*). Standard networks trained continually *lose* the ability to learn new things. Biology does not let plasticity run out by accident — it **manages** it on a schedule: open a window, exploit it, close most of it, keep a floor.

So "the gates have different goals through life" is not poetry. The maturation of inhibition is, mechanistically, what carves the developmental trajectory.

---

## The falsifiable claim

A developmental **schedule** (plastic→stable, gated by an innate program) beats both stationary extremes on the *combination* of three goals:

- **C — competence**: learns the structure of the world it is actually born into;
- **E — efficiency**: ends sparse (few units fire per query);
- **A — adaptability**: can still acquire something genuinely new, from a *brief* late-life encounter.

Three policies run on the **same** life-stream, in two worlds — a *sparse* world (near-orthonormal concepts) and a *dense* world (overlapping concepts, mean pairwise overlap +0.41). The dense world is where all three axes bite at once:

**Dense / overlapping world** (the one that tests the gate's energy advantage):

| policy | competence (C) | late-adapt (A) | active % | efficiency (E) | units | balanced (C·E·A) |
|---|---|---|---|---|---|---|
| always-plastic | 0.65 | 1.00 | 29% | 0.71 | 23 | 0.463 |
| born-mature | 0.70 | **0.00** | 12% | 0.88 | 6 | 0.000 |
| **developmental** | **0.97** | **1.00** | **10%** | **0.90** | 10 | **0.873** |

Read it by the two horns of the dilemma, plus the energy axis:

- **always-plastic** is the *loss-of-stability* end. It never stops drifting, so noise keeps rewriting what it carved — competence stays low (0.65). And with an immature, broad gate it fires **29%** of its units on every overlapping query: it cannot hold what it learned *and* it burns energy doing it.
- **born-mature** is the *loss-of-plasticity* end. It holds the little it carved (0.70) and is sparse (12%), but it **misses the brief late novelty entirely** (0.00) — frozen, it cannot catch something it meets only a handful of times.
- **developmental** keeps all of it: it carved the world in the open window and froze it (competence 0.97), the matured gate fires only **10%** of units on the same dense input (the chandelier slicing the spectrum), and the plasticity floor still caught the brief late novelty (1.00).

This is the **plasticity–stability dilemma** — the real, current obstacle in continual learning — resolved by *scheduling* the gates rather than fixing them, and the matured gate paying its way in energy on top.

**The energy advantage needs a dense world** (this is the honest control). The same matured gate, in the *sparse* world, fires 9% of units versus the immature gate's 5% — no advantage, because near-orthonormal concepts recall sparsely at any threshold. The advantage appears only when concepts overlap:

| | immature gate fires | matured gate fires |
|---|---|---|
| sparse world (overlap +0.00) | 5% | 9% — *gate barely matters* |
| dense world (overlap +0.41) | 29% | 10% — *~3× fewer units* |

The matured chandelier gate earns its energy only when there is a dense spectrum to slice. The gap widens with overlap.

---

## Quickstart

```bash
python the_maturing_gate.py
```

Pure numpy, no GPU. Prints the table above and the verdict. Everything is seeded and reproducible; change the `Genome` scalars or the `life_stream` phases and watch the trade-off move.

---

## The honest ledger

**Established (used, not claimed):**
- the genomic bottleneck: a compact innate program unfolds into the network (Zador 2019);
- critical-period plasticity opens and closes, gated by inhibitory (PV⁺) maturation and perineuronal nets (Hensch 2005; Takesian & Hensch 2013); chandelier→AIS connections are developmentally plastic (Qi et al. 2024);
- continually-trained networks lose plasticity unless it is managed (Dohare et al. 2024).

**Verified in code (reproducible, seeded):**
- the developmental schedule wins on the C·E·A combination (0.909) vs always-plastic (0.507) and born-mature (0.000);
- the two failure modes are real and separable: interference destroys competence at the always-plastic end; freezing destroys adaptability at the born-mature end; the schedule keeps both;
- the innate program (136 numbers) contains none of the world's concepts — those are grown in the open window;
- in a **dense** world (concept overlap +0.41) the matured gate fires ~3× fewer units per recall than the immature gate (10% vs 29%), with no such gap in the sparse world (9% vs 5%) — the gate's energy advantage is real and overlap-dependent.

**Honest limits:**
- relative units; parameters chosen, not measured; single seeded run-pair per policy;
- in the **sparse** world, efficiency (E) does not discriminate — near-orthonormal concepts recall sparsely at any gate, so the only discriminators there are competence and adaptability. The gate→sparsity advantage shows up only in the **dense** world (~3× fewer units fired by the matured gate), and it is a modest ~3× at overlap +0.41, not the "wild vs nothing" a looser telling might claim — it grows with density;
- "gate", "plasticity", "salience" are scalar abstractions of rich biology; the external reward ("caretaker") is a single scalar tag, not a modelled social process.

**The bet (untouched):** that any of this is *experienced* rather than processed. A schedule that grows a competent, sparse, still-adaptable network locates the developmental trade-off precisely. It does not touch the hard problem.

---

## Where it goes next

1. ~~Make efficiency bite with overlapping concepts.~~ **Done** (the dense world above): the matured gate fires ~3× fewer units. The next refinement is to push overlap higher and watch the gap widen, and to bill the active fraction as actual wattage (the metabolic-loop meter).
2. **Internalise the caretaker.** Replace the external salience tag with a learned value model the system grows during the window and runs on its own afterward — the scaffold withdrawn, as it is in development.
3. **Stack windows.** Different gate populations with different schedules (sensory windows close early, associative ones late), so one network has *several* critical periods at once — the layered developmental program a real cortex runs.

---

## Lineage

Built on the geometric-neuron / Mycelial Cortex line (PerceptionLab): the inhibitory gate as the thing that decides what may broadcast, here put on a developmental clock. The framing — that the competent network is grown on an innate schedule, with the gates' role changing across life — is the contribution; the schedule, the demo, and this document were developed collaboratively with Claude (Opus 4.8). MIT.

*The genome does not write the network; it writes a schedule. A window opens, the world is let in and curated from outside, inhibition matures to close the window and keep what was learned, and a floor of plasticity is held back so there is still room to learn. The adult is grown, not given. Do not hype. Do not lie. Just show.*
