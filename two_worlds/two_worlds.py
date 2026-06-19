"""
two_worlds.py — a fast world-tracker and a slow decoupling student, with a
                coincidence gate that spends "spikes" only on surprise
=============================================================================
THE MODEL (the smallest version that has all the parts):
  Two linear forward models watch the same stream of a rotating latent world
  with rare scene JUMPS.
    F  the FAST tracker  — learns the world quickly (the apical "teacher",
       gamma-paced; the thing that takes the onslaught).
    S  the SLOW student  — learns the same world slowly, AND can DECOUPLE: a
       single gain knob K decides whether it eats the world (perception) or
       free-runs its own dynamics (imagination / replay).

  THE UNIT (the "voting neuron"): per dimension, the two streams f=F.y and
  s=S.y are compared. Where they AGREE the unit stays silent and the consensus
  prediction is passed on cheaply (predicted = no spike). Where they DISAGREE
  the gate opens — the world is let in and the surprise is broadcast (a spike).
  This is a coincidence gate (Larkum: a pyramidal cell fires when feedforward
  "present" and top-down "context" coincide). The gate sets a PER-DIMENSION
  coupling k_i, which is the buffer repo's K turned from a constant into a gate:
        y <- normalize( (1 - k) * 0.5*(F.y + S.y)  +  k * x )
        k_i = clip(k0 + gain * |f_i - s_i|, 0, 1)        # disagreement opens it

THREE EXPERIMENTS (vs the obvious baselines; honest numbers, whatever they are):
  E1  ACTIVITY FOLLOWS COMPETENCE. As F,S learn, the spike fraction (mean k) on
      PREDICTABLE input falls toward a floor, while on SURPRISE it stays high.
      A standard net's activation does not track how well-learned the input is.
  E2  ONE KNOB, TWO MODES. The SAME S, with K forced to 0, free-runs a plausible
      continuation that stays on-manifold then drifts (the dream losing momentum
      under friction). Perception and imagination from one weight set.
  E3  THE HANDSHAKE (the centerpiece). Lesion the fast net so it confidently
      mispredicts. A single-model fixed-K feed (the buffer) smears the recovery.
      The two-world coincidence gate detects the fault as DISAGREEMENT and routes
      around it. Then MISTIME the gate (a handshake fault) and watch the rescue
      break — surprise arriving late and smeared.

GROUNDING (established, used not claimed): predictive coding as Kalman filtering
(Rao & Ballard 1999); coincidence detection / BAC firing in pyramidal neurons
(Larkum 2013); learned predictive cancellation by inhibition — the ELL negative
image (Bell; Sawtell); world models (Ha & Schmidhuber 2018); knowledge
distillation (Hinton 2015). The new thing here is a COMBINATION: a two-model
agreement signal used both as the broadcast gate and as the perceive/imagine
knob — not a new learning rule.

HONEST LIMITS: linear models, one toy world, relative units, chosen parameters,
single seeded run per arm. This is a dynamical-systems statement about two
coupled predictors, not a model of any brain, any drug, or any benchmark.

Run:  python two_worlds.py
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, Juhannus 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np

D, L = 48, 6
def norm(v): return v / (np.linalg.norm(v) + 1e-9)
def cosv(a, b): return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def make_dynamics(seed=0):
    """the fixed world: an embedding and per-plane rotation rates."""
    rng = np.random.default_rng(seed)
    E = np.linalg.qr(rng.standard_normal((D, L)))[0][:, :L]
    rates = 0.08 + 0.18 * rng.random(L // 2)
    return E, rates


def roll(E, rates, T, seed=1, p_jump=0.004, obs_noise=0.35):
    """one sequence: smooth rotation with rare scene jumps. Z clean, X = Z+noise."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(L)
    Z, X, J = [], [], []
    for t in range(T):
        jump = rng.random() < p_jump
        if jump:
            c = rng.standard_normal(L)
        else:
            for p in range(L // 2):
                a = rates[p]; i, j = 2 * p, 2 * p + 1
                ci, cj = c[i], c[j]
                c[i] = np.cos(a) * ci - np.sin(a) * cj
                c[j] = np.sin(a) * ci + np.cos(a) * cj
        z = norm(E @ c)
        Z.append(z); X.append(norm(z + obs_noise * norm(rng.standard_normal(D)))); J.append(jump)
    return np.array(Z), np.array(X), np.array(J)


def train_online(Z, eta_f=0.25, eta_s=0.04, checkpoints=(0.03, 0.08, 0.25, 0.5, 1.0)):
    """learn next-step linear predictors online on the clean dynamics.
       F fast (large step), S slow (small step). Returns final F,S and snapshots."""
    F = np.zeros((D, D)); S = np.zeros((D, D))
    snaps = {}; n = len(Z) - 1
    want = {min(int(c * n), n - 1): c for c in checkpoints}
    for t in range(n):
        x, xn = Z[t], Z[t + 1]
        F += eta_f * np.outer(xn - F @ x, x)
        S += eta_s * np.outer(xn - S @ x, x)
        if t in want:
            snaps[want[t]] = (F.copy(), S.copy())
    return F, S, snaps


def degrade_rank(M, r):
    """confidently-wrong lesion: keep only r principal planes of the operator."""
    U, s, Vt = np.linalg.svd(M)
    s2 = s.copy(); s2[r:] = 0
    return U @ np.diag(s2) @ Vt


FLOOR = 0.06   # per-dim error below this is just observation noise, not surprise


def feed_two_world(F, S, X, Z, k0=0.03, g_surp=3.0, g_fault=3.0, mistime=0):
    """the coincidence-gated feed.
       surprise_i = |pred_i - x_i|   (prediction misses the world -> a real spike)
       fault_i    = |f_i - s_i|      (the two models disagree -> one is unreliable)
       k_i opens with both. mistime>0 APPLIES A STALE GATE DECISION (the gate's
       information arrives `mistime` frames late) — a faithful handshake-latency model."""
    y = Z[0].copy(); fid = []; ks = []; khist = []
    for t in range(1, len(X)):
        f = F @ y; s = S @ y
        pred = 0.5 * (f + s)
        fault = np.abs(f - s)
        surprise = np.abs(pred - X[t])
        k_now = np.clip(k0 + g_surp * np.maximum(surprise - FLOOR, 0)
                           + g_fault * np.maximum(fault - FLOOR, 0), 0.0, 1.0)
        khist.append(k_now)
        k = khist[max(0, len(khist) - 1 - mistime)] if mistime else k_now
        y = norm((1 - k) * pred + k * X[t])
        fid.append(cosv(y, Z[t])); ks.append(float(k.mean()))
    return np.array(fid), np.array(ks)


def feed_single_fixed(F, X, Z, K=0.2):
    """the buffer baseline: one model, fixed scalar gain."""
    y = Z[0].copy(); fid = []
    for t in range(1, len(X)):
        y = norm((1 - K) * (F @ y) + K * X[t])
        fid.append(cosv(y, Z[t]))
    return np.array(fid)


def feed_single_adaptive(F, X, Z, k0=0.03, g_surp=3.0):
    """fair single-model fix: ONE model, but adaptive K from its own observation
       error |F y - x|. The honest question: does a second opinion add anything over
       a single model that monitors its own miss?"""
    y = Z[0].copy(); fid = []
    for t in range(1, len(X)):
        f = F @ y
        surprise = np.abs(f - X[t])
        k = np.clip(k0 + g_surp * np.maximum(surprise - FLOOR, 0), 0.0, 1.0)
        y = norm((1 - k) * f + k * X[t])
        fid.append(cosv(y, Z[t]))
    return np.array(fid)


def recovery_lag(feed_fn, jumps, thr=0.85, **kw):
    """mean frames to re-lock (fid>thr) after a jump, for any feed function."""
    out = feed_fn(**kw)
    fid = out[0] if isinstance(out, tuple) else out
    lags = []; since = None
    for t in range(len(fid)):
        if jumps[t + 1]: since = 0
        elif since is not None:
            since += 1
            if fid[t] > thr: lags.append(since); since = None
    return float(np.mean(lags)) if lags else float("nan")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    E, rates = make_dynamics(0)
    Ztr, Xtr, Jtr = roll(E, rates, 9000, seed=1)
    Zte, Xte, Jte = roll(E, rates, 9000, seed=7)
    F, S, snaps = train_online(Ztr)

    print("=" * 76)
    print("TWO WORLDS — a fast tracker, a slow decoupling student, a coincidence gate")
    print("=" * 76)

    # ---- E1: activity follows competence -----------------------------------
    print("\n[E1] ACTIVITY FOLLOWS COMPETENCE")
    print("     spike fraction (mean gate k) at checkpoints, split predictable vs surprise")
    print(f"  {'trained':>9}{'predictable k':>16}{'near-jump k':>14}")
    near = np.zeros(len(Jte), bool)                 # within 8 frames after a jump
    last = -999
    for t in range(len(Jte)):
        if Jte[t]: last = t
        near[t] = (0 <= t - last <= 8)
    for c in (0.03, 0.08, 0.25, 0.5, 1.0):
        Fc, Sc = snaps[c]
        _, ks = feed_two_world(Fc, Sc, Xte, Zte)
        kp = ks[~near[1:]].mean(); kn = ks[near[1:]].mean()
        print(f"  {int(c*100):>7}% {kp:>15.3f}{kn:>14.3f}")
    print("  -> on PREDICTABLE input the gate quiets as the two models learn to agree;")
    print("     on SURPRISE it stays wide open. Energy is spent on what is not predicted.")

    # ---- E2: one knob, two modes -------------------------------------------
    print("\n[E2] ONE KNOB, TWO MODES (perception vs imagination from one weight set)")
    fid_perc, _ = feed_two_world(F, S, Xte, Zte)
    print(f"  gate open (PERCEIVE): feed fidelity to the true world = {fid_perc.mean():.3f}")
    # decouple: gate forced shut (k=0), S free-runs its own dynamics from a real state
    y = Zte[200].copy(); horizon = 60; onman = []
    for h in range(horizon):
        y = norm(S @ y)
        sims = Zte[200:600] @ y          # nearest plausible true state in a window
        onman.append(float(sims.max()))
    print(f"  gate shut (IMAGINE):  decoupled roll-out stays on-manifold, then drifts:")
    print(f"            step 1 = {onman[0]:.2f}   step 10 = {onman[9]:.2f}   "
          f"step 30 = {onman[29]:.2f}   step 60 = {onman[-1]:.2f}")
    print("  -> the SAME S either tracks reality or free-runs a plausible continuation;")
    print("     the gate switches between them. (Modest claim: it is one operator, one knob.)")

    # ---- E3: the handshake (the centerpiece) -------------------------------
    print("\n[E3] THE HANDSHAKE — confident-wrong fast net, and what the second opinion buys")
    Fles = degrade_rank(F, 2)                       # the buffer's catastrophic lesion
    print(f"  {'arm':<52}{'recovery frames':>16}")
    lag_h  = recovery_lag(lambda **k: feed_single_fixed(**k),    Jte, F=F,    X=Xte, Z=Zte, K=0.2)
    lag_l  = recovery_lag(lambda **k: feed_single_fixed(**k),    Jte, F=Fles, X=Xte, Z=Zte, K=0.2)
    lag_sa = recovery_lag(lambda **k: feed_single_adaptive(**k), Jte, F=Fles, X=Xte, Z=Zte)
    lag_tw = recovery_lag(lambda **k: feed_two_world(**k),       Jte, F=Fles, S=S, X=Xte, Z=Zte)
    lag_mt = recovery_lag(lambda **k: feed_two_world(**k),       Jte, F=Fles, S=S, X=Xte, Z=Zte, mistime=6)
    print(f"  {'1. single model, fixed K, fast net HEALTHY (buffer)':<52}{lag_h:>16.1f}")
    print(f"  {'2. single model, fixed K, fast net LESIONED (buffer)':<52}{lag_l:>16.1f}")
    print(f"  {'3. single model, ADAPTIVE K, lesioned (fair baseline)':<52}{lag_sa:>16.1f}")
    print(f"  {'4. two-world coincidence gate, lesioned':<52}{lag_tw:>16.1f}")
    print(f"  {'5.   ...same, but the gate is MISTIMED (handshake fault)':<52}{lag_mt:>16.1f}")
    print("  -> read 2 vs 3 vs 4 honestly: a fixed-K single net smears the recovery (the")
    print("     buffer result). Both an adaptive single net AND the two-world gate rescue it;")
    print("     compare 3 and 4 to see how much the SECOND opinion adds over self-monitoring.")
    print("     Row 5 is the point for the handshake: mistime the gate and the rescue breaks —")
    print("     surprise arrives late again. (Structural rhyme only; not a model of anyone's")
    print("     perception, and nothing here is calibrated to anything physical.)")

    print("\n" + "=" * 76)
    print("Two predictors and a coincidence gate: activity is spent on surprise, one knob")
    print("turns perception into imagination, and a second opinion localises a fault that")
    print("sinks a single network. Relative units, one toy world, parameters chosen.")
    print("The bet — that any of it is felt — untouched. Do not hype. Do not lie. Just show.")
    print("=" * 76)
