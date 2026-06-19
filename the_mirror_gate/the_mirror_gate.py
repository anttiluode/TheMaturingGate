"""
the_mirror_gate.py — a fast inhibitory network that learns the expected world
and teaches the slow principal network
=============================================================================
ONE HYPOTHESIS, TESTED THREE WAYS:
  The fast (gamma-rate) inhibitory network is a learned FORWARD MODEL of the
  expected input — a "mirror" of the world's temporal structure — and its job is
  to teach the slow (theta-rate) principal network:
    DEMO 1  it pulls the slow learner up faster (distillation) — but ONLY if it
            is a true mirror of THIS world; a confident non-mirror is poison.
    DEMO 2  if the mirror is born with a WRONG innate prior, trusting it from
            birth slows the learner; trusting it on a maturing schedule does not.
            (This is the link to the maturing-gate: schedule the trust.)
    DEMO 3  once matured, the mirror cancels what it can predict, so the
            principal only encodes the SURPRISE — sparse error-coding.

GROUNDING (established, used not claimed):
  - fast-spiking PV+ basket/chandelier interneurons generate gamma via local
    loops; principal cells are paced slower by theta (Buzsaki; PING/ING).
    "Inhibition runs faster" is real.
  - inhibition that LEARNS to predict and cancel expected input is real: the
    electric-fish ELL grows a "negative image" subtracting the predicted sensory
    consequence, leaving the surprise (Bell; Sawtell). Cortical predictive coding
    makes the same proposal (Rao & Ballard 1999; Attinger et al. 2017).
  - a fast net teaching a slow net is knowledge distillation (Hinton et al. 2015).
  - inhibitory maturation closes plasticity windows (Hensch 2005) — the schedule.

HYPOTHESIS (the new bridge, NOT established): that this fast inhibitory predictor
is a learned mirror/world-model whose role is to teach the principal network, and
that gating/"voting" (chandelier veto; Thousand-Brains) is this expectation
silencing the unexpected. Tested here in a toy; honest about being a toy.

HONEST LIMITS: linear next-step predictors, one toy world, single seed per arm,
relative units, parameters chosen. The maturing-trust effect is small in a linear
toy (it would grow where early models are strongly, confidently wrong).
"teacher/student/mirror" are linear abstractions of rich biology. No claim about
experience.

Run:  python the_mirror_gate.py
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, Juhannus 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np

D, K = 64, 10
def norm(v): return v / (np.linalg.norm(v) + 1e-9)
def cos(a, b): return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
def smoothstep(x): x = np.clip(x, 0, 1); return float(x * x * (3 - 2 * x))


def make_world(seed=0):
    return np.linalg.qr(np.random.default_rng(seed).standard_normal((D, K)))[0][:, :K].T


def gen_stream(P, T=8000, p_follow=0.85, seed=1):
    """mostly-cyclic tour 0->1->...->K-1->0 (predictable) with rare jumps (surprise)."""
    rng = np.random.default_rng(seed); ks = [0]
    for _ in range(T - 1):
        ks.append((ks[-1] + 1) % K if rng.random() < p_follow else int(rng.integers(K)))
    ks = np.array(ks)
    X = np.stack([norm(P[k] + 0.45 * norm(rng.standard_normal(D))) for k in ks])
    surprise = np.array([ks[t] != (ks[t - 1] + 1) % K for t in range(1, T)])
    return X, ks, surprise


def wrong_prior(P, seed=2):
    """a CONFIDENT non-mirror: a structured map onto a WRONG (shuffled) 'next'."""
    rng = np.random.default_rng(seed); perm = rng.permutation(K); W = np.zeros((D, D))
    for _ in range(4000):
        k = int(rng.integers(K)); x = norm(P[k] + 0.45 * norm(rng.standard_normal(D)))
        W += 0.3 * np.outer(P[perm[k]] - W @ x, x)
    return W


def run(arm, X, surprise, init_W, teacher_learns,
        eta_slow=0.15, eta_fast=0.03, beta_max=0.8, T_mature=0.25, p_update_s=0.04, seed=4):
    """one online life. Principal (student) has SLOW, SPARSE plasticity (updates on a
       small fraction of steps); the fast inhibitory mirror (teacher) updates EVERY
       step. arm in {alone, fixed, maturing}; init_W = teacher's innate prior."""
    rng = np.random.default_rng(seed); T = len(X)
    Ws = np.zeros((D, D)); Wt = init_W.copy(); errs = []
    for t in range(T - 1):
        x, xnext = X[t], X[t + 1]
        tpred = Wt @ x
        if   arm == "alone":    beta = 0.0
        elif arm == "maturing": beta = beta_max * smoothstep(t / (T_mature * T))
        else:                   beta = beta_max          # fixed trust from birth
        spred = Ws @ x
        errs.append(cos(spred, xnext))                   # scored against TRUTH only
        if rng.random() < p_update_s:                    # slow, sparse principal plasticity
            target = (1 - beta) * xnext + beta * tpred
            Ws += eta_slow * np.outer(target - spred, x)
        if teacher_learns:
            Wt += eta_fast * np.outer(xnext - tpred, x)  # fast mirror tracks the real world
    return np.array(errs), Wt


def early(e):  return float(np.mean(e[int(.05*len(e)):int(.20*len(e))]))
def final(e):  return float(np.mean(e[int(.80*len(e)):]))
def steps_to(e, thr=0.55):
    sm = np.convolve(e, np.ones(100)/100, "valid"); h = np.where(sm > thr)[0]
    return int(h[0]) if len(h) else len(e)


if __name__ == "__main__":
    P = make_world(); X, ks, surprise = gen_stream(P)
    Z = np.zeros((D, D)); prior = wrong_prior(P)

    print("=" * 76)
    print("THE MIRROR GATE — a fast mirror-teacher teaches the slow principal net")
    print("=" * 76)

    print("\nDEMO 1 — the mirror teaches, and it must be a real mirror")
    print(f"  {'arm':<26}{'early acc':>11}{'final acc':>11}{'steps->.55':>12}")
    for label, arm, iW, learn in [
        ("alone (no teacher)",        "alone", Z,     True),
        ("mirror teacher",            "fixed", Z,     True),
        ("WRONG teacher (non-mirror)", "fixed", prior, False)]:
        e, _ = run(arm, X, surprise, iW, learn)
        print(f"  {label:<26}{early(e):>11.2f}{final(e):>11.2f}{steps_to(e):>12}")
    print("  -> the mirror teacher reaches threshold sooner and scores higher early than")
    print("     learning alone; a confident NON-mirror is catastrophic. The fast network")
    print("     helps only when it mirrors THIS world.")

    print("\nDEMO 2 — when the mirror is born WRONG, schedule the trust (the maturing-gate link)")
    print(f"  {'arm (teacher born w/ wrong prior)':<34}{'early':>9}{'final':>9}{'steps->.55':>12}")
    for label, arm in [("trust from birth (fixed)", "fixed"), ("trust on a schedule (maturing)", "maturing")]:
        e, _ = run(arm, X, surprise, prior, True)
        print(f"  {label:<34}{early(e):>9.2f}{final(e):>9.2f}{steps_to(e):>12}")
    print("  -> trusting a miscalibrated innate prior from birth SLOWS the learner; the")
    print("     maturing schedule withholds trust until the prior is corrected. (Small")
    print("     magnitude in a linear toy; it grows where early models are confidently wrong.)")

    print("\nDEMO 3 — once matured, the mirror cancels the expected, leaves the surprise")
    _, Wt = run("maturing", X, surprise, Z, True)
    def profile(W):
        ep, es = [], []
        for t in range(1, len(X)):
            r = np.linalg.norm(X[t] - W @ X[t - 1])
            (es if surprise[t - 1] else ep).append(r)
        return float(np.mean(ep)), float(np.mean(es))
    p0, s0 = profile(Z); pM, sM = profile(Wt)
    print(f"  mean residual ||x_next - prediction|| (what the principal must still encode):")
    print(f"    {'':<20}{'predictable':>13}{'surprise':>11}")
    print(f"    no teacher (raw)   {p0:>13.2f}{s0:>11.2f}")
    print(f"    matured mirror     {pM:>13.2f}{sM:>11.2f}")
    print(f"  -> predictable residual collapses ({p0:.2f}->{pM:.2f}, {100*(1-pM/p0):.0f}% cancelled);")
    print(f"     surprise residual stays large ({sM:.2f}). The principal only encodes surprise.")

    print("\n" + "=" * 76)
    print("The fast inhibitory net learns the expected world and teaches the slow one:")
    print("it pulls the learner up faster (only as a true mirror), the trust must be")
    print("scheduled when the mirror starts wrong, and once matured it cancels the")
    print("predictable so only surprise costs energy.")
    print("Relative units, parameters chosen. The bet — that any of it is felt — untouched.")
    print("Do not hype. Do not lie. Just show.")
    print("=" * 76)
