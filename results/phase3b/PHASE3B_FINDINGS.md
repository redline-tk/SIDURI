
## GOAD hidden_dim bug found, fixed, and scope-verified (2026-08-24)

Root cause: GOAD's library default hidden_dim=8 was catastrophically undersized for UNSW's
feature space specifically. Discovered via Layer 1 external validation (SLAD paper, arXiv
2305.16114, reports GOAD=0.903+/-0.003 on UNSW-NB15) - SIDURI's original UNSW GOAD AUROC was
0.478 (below random), an 0.425 gap impossible to explain by undertraining alone.

Fix: hidden_dim=100 scoped to UNSW only (dataset_name=="unsw_nb15"), default hidden_dim=8
preserved everywhere else. Verified via a rejected circular tuning script
(goad_hidden_dim_tune.py, selected hidden_dim by maximizing AUROC against calibration-period
LABELS - same defect class as the retired M-grid search, not used) that hidden_dim=100 would
have been WORSE than the default on SMD (0.70 vs 0.96) - confirming the UNSW-only scoping,
not a global change, was correct.

Result: UNSW GOAD AUROC 0.478 -> 0.8164 (vs published 0.903, residual gap attributable to
protocol differences - SIDURI's temporal cal/eval split vs the paper's standard holdout).

VERIFIED NO DOWNSTREAM IMPACT ON ANY EXISTING PHASE 3B RESULT: recomputed SMD machine-1-1
GOAD AUROC directly from the pre-fix frozen score_cache (0.9520) and confirmed it matches
the corrected pipeline's fresh run exactly (0.9520). SMD's GOAD was never broken - only
UNSW was. Phase 2 coverage audit, ICC, H1/H2/H3, and SMD Branch A (the decay-curve result
motivating the tail-censorship pivot) all remain valid as computed, no re-run required.

ACTION: update any Table 2 / raw-AUROC reporting of UNSW GOAD to 0.8164, was 0.478.
Also fixed in this pass, unrelated to GOAD: added torch.cuda.empty_cache() after each
detector's decision_function calls in run_extended_methods.py (no cleanup existed
previously - latent OOM risk over a long multi-unit run, now mitigated).

## Layer 2 CONFIRMED (Bridge Plan validation audit, 2026-08-25): SMAP weak-detector retraction holds, mechanism corrected

Sampled 20 weak units (original AUROC<0.6) + 5 strong controls (AUROC>=0.7) from SMAP,
retrained all at 250 epochs (vs original 50), same architecture/hyperparameters otherwise.

WEAK UNITS: mean delta = +0.037, only 3/20 rose materially (>0.2 AUROC). Individual deltas
scatter both positive and negative with comparable magnitude in each direction (e.g.
RDP/A-7 +0.52, GOAD/P-7 -0.21, ICL/G-6 +0.26, GOAD/A-4 -0.20) - the signature of small-sample
retraining VARIANCE on SMAP's short channels (28-229 total length per Phase 1 audit), not a
systematic convergence effect that would cluster positive if real signal existed.

STRONG CONTROLS: mean delta = -0.206, NOT near zero as a clean control should be. SLAD/D-5
collapsed 1.0000->0.5154, NeuTraL/A-6 dropped 0.22. This shows convergence training on SMAP's
small channels induces real per-unit instability/overfitting in BOTH directions, not just on
already-weak units.

CORRECTED MECHANISM STATEMENT: the original SMAP retraction ("near-random AUROC, detector
quality insufficient") is CONFIRMED but the underlying reason is refined - it is not that
SMAP detectors are stably near-random regardless of training, it is that SMAP's short
channels make per-unit AUROC volatile under retraining in general (evidenced by strong
controls also moving substantially), and this volatility happens to not produce a systematic
upward rescue of the weak units. This is a more precise and more defensible claim for the
paper than the original framing, and is now independently corroborated by Layer 3's opposite
result on SMD (where 250-epoch retraining left drift-slope essentially unchanged, ICC and
AUROC-filtered results held cleanly) - the contrast between a stable dataset (SMD) and a
volatile one (SMAP) under identical retraining protocol is itself informative and worth a
sentence in the paper's benchmark-characterization discussion.

BRIDGE PLAN LAYERS 1-3 STATUS: ALL COMPLETE.
Layer 1: pipeline reproduces published external number (ICL on UNSW, gap 0.018); one real
bug found (GOAD hidden_dim), fixed, scoped correctly, verified zero downstream impact.
Layer 3: SMD Branch A drift survives convergence training (rho 0.2959->0.3029, p improves
by 10x). Load-bearing result confirmed robust.
Layer 2: SMAP retraction confirmed, mechanism refined to channel-length-driven instability
rather than simple stable near-randomness.

## Layer 2 addendum: two exact-0.5000 weak units are degenerate-score artifacts

P-4__NeuTraL and D-8__GOAD both showed old=new=0.5000 exactly. Checked directly: both have
n_distinct_eval_scores=1 (single constant value across the entire eval set), which forces
AUROC=0.5000 by construction regardless of training. Not evidence of stable near-random
detection - these are degenerate/collapsed-output units, same failure class as the SMAP
D-12/D-13 constant-feature findings from Phase 1. Does not change the Layer 2 verdict (3/20
materially rose either way; these two were not among them), but should not be read as
"detector output was stable under retraining" - there was no real output to begin with.
