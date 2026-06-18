"""
the_maturing_gate.py — growing a network on an innate developmental schedule
============================================================================
A runnable, numpy-only study of one idea: a competent network is not specified,
it is GROWN — on a small innate program that schedules its inhibitory gates from
broad-and-plastic (infant) to selective-and-stable (adult), while keeping a
managed residual plasticity for the rest of life.

WHY THIS IS NOT JUST A METAPHOR (the grounding):
  - GENOMIC BOTTLENECK (Zador 2019): the genome is far too small to store the
    wiring; it stores a compact *program* that unfolds into the network through
    development. The competent adult is grown, not written down.
  - CRITICAL PERIODS (Hensch 2005; Takesian & Hensch 2013): windows of high
    plasticity that OPEN and CLOSE, and the closing is gated by the maturation of
    inhibitory (PV+ basket) interneurons and perineuronal nets. Inhibition
    maturing IS what ends the learning window. Axo-axonic (chandelier) -> AIS
    connections are themselves developmentally plastic ("Specific and Plastic",
    Qi et al. 2024).
  - LOSS OF PLASTICITY (Dohare et al. 2024): standard nets, trained continually,
    LOSE the ability to learn new things. Biology manages plasticity on a
    schedule instead of letting it run out — opens a window, exploits it, closes
    most of it, keeps a floor.

THE FALSIFIABLE CLAIM:
  A developmental SCHEDULE (plastic->stable, gated by an innate program) beats
  both stationary extremes on the *combination* of three goals that matter to a
  real lifelong system:
    (C) COMPETENCE   — learns the structure of the world it is actually born into
    (E) EFFICIENCY   — ends sparse: few units fire per query (the energy/firewall win)
    (A) ADAPTABILITY — can still acquire something genuinely new, late in life
  - ALWAYS-PLASTIC  gets C and A but never E, and drifts (interference).
  - BORN-MATURE     gets E but never C or A (frozen with its innate prior).
  - DEVELOPMENTAL    is the only policy that gets all three at once.

THE INNATE PROGRAM is tiny (a handful of scalars + one value direction). It does
NOT contain the world's concepts — those are learned during the open window,
scaffolded early by an external reward (a "caretaker") that tags what is worth
keeping, then internalised. That size gap is the genomic bottleneck, made literal.

HONEST LIMITS: relative units, parameters chosen not measured; near-orthonormal
toy concepts, one corruption level, single run-pair per policy; "gate", "plasticity"
and "salience" are scalar abstractions of rich biology. This shows the ARCHITECTURE
of a developmental schedule produces the three-way win, not that any brain's or
model's numbers match. No claim about experience is made.

Run:  python the_maturing_gate.py
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np

D = 128
def norm(v): return v / (np.linalg.norm(v) + 1e-9)
def cos(a, b): return float(norm(a) @ norm(b))
def corrupt(x, c, rng):
    return norm(norm(x) + c * norm(rng.standard_normal(D)))   # unit-scaled noise, not sqrt(D)-scaled


# ----------------------------------------------------------------------
# The innate program (the "genome"): a handful of scalars + a value direction.
# It schedules plasticity pi(t) and gate-selectivity iota(t). It does NOT
# contain the world's concepts.
# ----------------------------------------------------------------------
class Genome:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.pi_max     = 0.95     # infant plasticity (window open)
        self.pi_floor   = 0.10     # adult residual plasticity (managed, not zero)
        self.pi_frozen  = 0.015    # the "born-mature" near-frozen level
        self.iota_low   = 0.20     # infant gate: broad (almost everything passes)
        self.iota_high  = 0.55     # adult gate: selective (only strong matches pass)
        self.win_close  = 0.50     # the critical period closes around mid-life
        self.win_ramp   = 0.12     # how sharply it closes
        self.spawn_thr  = 0.55     # below this best-match -> a new unit may be recruited
        self.recog_thr  = 0.55     # above this -> recall counts as recognised
        self.budget     = 60       # max units that can be grown
        self.value_dir  = norm(rng.standard_normal(D))   # innate salience bias (small)

    def size_scalars(self):
        return 8 + D   # the program: 8 scalars + one value direction


def smoothstep(x): return float(np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1)))

def schedule(t, g: Genome, policy):
    """return (plasticity pi, gate-selectivity iota) at developmental time t in [0,1]."""
    if policy == "always_plastic":
        return g.pi_max, g.iota_low
    if policy == "born_mature":
        return g.pi_frozen, g.iota_high
    # developmental: plastic+broad early -> stable+selective late, with a floor
    close = smoothstep((t - g.win_close) / g.win_ramp + 0.5)   # 0 before window, 1 after
    pi   = g.pi_max * (1 - close) + g.pi_floor * close
    iota = g.iota_low * (1 - close) + g.iota_high * close
    return pi, iota


# ----------------------------------------------------------------------
# The agent: grows units (templates), drifts them under plasticity, gates recall.
# ----------------------------------------------------------------------
class Agent:
    def __init__(self, g: Genome):
        self.g = g
        self.P = []                     # grown templates (the learned content)

    def experience(self, x, salience, pi, rng):
        """one experience: recruit a new unit, or drift the best-matching one."""
        x = norm(x)
        if not self.P:
            if salience > 0.5 and rng.random() < pi:
                self.P.append(x.copy())
            return
        sims = np.array([cos(p, x) for p in self.P])
        best = int(np.argmax(sims)); bestcos = float(sims[best])
        if bestcos < self.g.spawn_thr and salience > 0.5 and len(self.P) < self.g.budget:
            if rng.random() < pi:
                self.P.append(x.copy())          # recruit a unit for the novel+salient
        else:
            if rng.random() < pi:                # drift the best unit toward this input
                rate = 0.30 * pi                 # high plasticity -> strong drift (and interference)
                self.P[best] = norm((1 - rate) * self.P[best] + rate * x)

    def recall(self, query, iota):
        """return (recognised?, best_cos, active_fraction) under gate threshold iota."""
        if not self.P:
            return False, 0.0, 0.0
        sims = np.array([cos(p, query) for p in self.P])
        active = float(np.mean(sims > iota))     # fraction of units that broadcast (energy proxy)
        best = float(sims.max())
        recognised = (best > self.g.recog_thr) and (best > iota)
        return recognised, best, active


# ----------------------------------------------------------------------
# The world: M concepts streamed during life, plus ONE novel concept that
# appears only AFTER maturation. Early, an external "caretaker" tags the useful
# concepts as salient (the scaffold); noise is low-salience.
# ----------------------------------------------------------------------
def make_world(seed=1, M=8, rho=0.0):
    """M concepts + 1 late novel concept. rho = target pairwise overlap:
       rho<=0 -> near-orthonormal (sparse world); rho>0 -> dense, overlapping world
       (each concept shares a common component, so distinct concepts have cos~rho)."""
    rng = np.random.default_rng(seed)
    if rho <= 0:
        X = rng.standard_normal((M, D))
        Q, _ = np.linalg.qr(X.T)
        return Q.T[:M], norm(rng.standard_normal(D))
    shared = norm(rng.standard_normal(D))
    def mk():
        u = norm(rng.standard_normal(D))
        return norm(np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * u)
    env = np.stack([mk() for _ in range(M)])
    novel = mk()
    return env, novel

def life_stream(env, novel, T=2400, seed=2):
    """
    a sequence of (input, salience, t) over a life.
      phase A (t<0.45): the open window — env concepts (caretaker-salient) + noise
      phase B (0.45..0.94): mostly noise, occasional env — the interference test
      phase C (t>0.94): a BRIEF encounter with a novel concept — the late-adapt test
    """
    rng = np.random.default_rng(seed)
    M = len(env); stream = []
    for i in range(T):
        t = i / T
        if t < 0.45:
            if rng.random() < 0.7:
                x = corrupt(env[rng.integers(M)], 0.5, rng); sal = 1.0
            else:
                x = norm(rng.standard_normal(D)); sal = 0.1
        elif t < 0.94:
            if rng.random() < 0.25:
                x = corrupt(env[rng.integers(M)], 0.5, rng); sal = 1.0
            else:
                x = norm(rng.standard_normal(D)); sal = 0.1          # noise: drifts the unstable
        else:
            # phase C: a BRIEF late-life encounter with the novel concept (~14 salient
            # exposures, not hundreds). Only a net that kept plasticity can catch it.
            if rng.random() < 0.10:
                x = corrupt(novel, 0.5, rng); sal = 1.0
            else:
                x = norm(rng.standard_normal(D)); sal = 0.1
        stream.append((x, sal, t))
    return stream


def run_life(policy, g, env, novel, stream):
    rng = np.random.default_rng(7)
    a = Agent(g)
    for x, sal, t in stream:
        pi, _ = schedule(t, g, policy)
        a.experience(x, sal, pi, rng)
    _, iota_final = schedule(1.0, g, policy)
    return a, iota_final


def evaluate(a, iota, env, novel, trials=60, seed=9):
    rng = np.random.default_rng(seed); M = len(env)
    # competence on the world it was born into
    ok, act = 0, []
    for _ in range(trials):
        k = rng.integers(M)
        q = corrupt(env[k], 0.6, rng)
        rec, _, frac = a.recall(q, iota); ok += int(rec); act.append(frac)
    competence = ok / trials
    efficiency_active = float(np.mean(act))         # lower = sparser = more efficient
    # adaptability: the late novel concept
    okn = 0
    for _ in range(trials):
        q = corrupt(novel, 0.6, rng)
        rec, _, _ = a.recall(q, iota); okn += int(rec)
    adapt = okn / trials
    return competence, efficiency_active, adapt, len(a.P)


# ----------------------------------------------------------------------
def run_world(label, rho):
    g = Genome()
    env, novel = make_world(rho=rho)
    stream = life_stream(env, novel)
    # report the actual mean pairwise overlap of this world
    M = len(env)
    ov = [cos(env[i], env[j]) for i in range(M) for j in range(i + 1, M)]
    print("=" * 78)
    print(f"WORLD: {label}   (mean pairwise concept overlap = {np.mean(ov):+.2f})")
    print("=" * 78)
    print(f"  {'policy':<16}{'competence':>11}{'active%':>10}{'efficiency':>11}{'late-adapt':>12}{'units':>7}")
    rows = {}
    for policy in ["always_plastic", "born_mature", "developmental"]:
        a, iota = run_life(policy, g, env, novel, stream)
        C, act, A, n = evaluate(a, iota, env, novel)
        E = 1.0 - act
        rows[policy] = (C, E, A, act, n)
        print(f"  {policy:<16}{C:>11.2f}{act*100:>9.0f}%{E:>11.2f}{A:>12.2f}{n:>7}")
    bal = lambda r: r[0] * r[1] * r[2]
    best = max(rows, key=lambda p: bal(rows[p]))
    print(f"\n  balanced (C*E*A):  " +
          "   ".join(f"{p.split('_')[0]} {bal(rows[p]):.3f}" for p in rows))
    print(f"  best on the combination: {best}")
    return rows


if __name__ == "__main__":
    print("THE MATURING GATE — same architecture, two worlds\n")
    g0 = Genome()
    print(f"innate program size: {g0.size_scalars()} numbers (8 scalars + 1 value dir);"
          f" it contains none of the concepts.\n")

    sparse = run_world("sparse / near-orthonormal (recall is sparse at ANY gate)", rho=0.0)
    print()
    dense = run_world("dense / overlapping (the matured gate must SLICE the spectrum)", rho=0.45)

    # the efficiency story, made explicit across the two worlds
    print("\n" + "=" * 78)
    print("THE GATE'S ENERGY ADVANTAGE APPEARS ONLY IN THE DENSE WORLD")
    print("=" * 78)
    ap_s, dv_s = sparse["always_plastic"][3], sparse["developmental"][3]
    ap_d, dv_d = dense["always_plastic"][3], dense["developmental"][3]
    print(f"  active fraction per recall (lower = less energy):")
    print(f"    sparse world : immature {ap_s*100:4.0f}%   matured {dv_s*100:4.0f}%   "
          f"(gate barely matters: orthogonal concepts recall sparsely anyway)")
    print(f"    dense  world : immature {ap_d*100:4.0f}%   matured {dv_d*100:4.0f}%   "
          f"(the matured gate fires far fewer units on overlapping input)")
    print(f"  -> in a dense world the immature net fires WIDE (every partial match passes a")
    print(f"     low gate); the matured chandelier gate slices to the few necessary units.")
    print(f"     That is the energetic point that the sparse world could not show.")
    print("\n  Relative units, parameters chosen. The bet — that any of it is felt — untouched.")
    print("  Do not hype. Do not lie. Just show.")
