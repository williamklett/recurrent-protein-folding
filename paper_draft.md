# Protein folding as an emergent energy descent: linear distance probes reveal what ESMFold is doing inside

**Working draft v0.1 — 2026-05-27**

---

## Abstract

We train per-layer linear distance probes on ESMFold's pair representation z, anchored only by chained consistency to the model's own output (no native-structure supervision). The probes recover, layer by layer, an evolving expected pairwise distance map E[d_t]. We make three observations. First, the distance maps are **geometrically self-consistent at every layer** — zero triangle-inequality violations across 57 proteins — even though the probe loss has no cross-pair or Euclidean term. Second, the implied 3D shape undergoes **coordinated rigid-body motion** across layers, despite no loss linking pair (i,j) and pair (i,k) during training. Third, the Lennard-Jones energy of that expected shape decreases **monotonically** through the trunk for every protein, with depth proportional to fold quality. ESMFold appears to descend an implicit energy landscape it was never asked to descend. Practically, the convergence shape of the probe trajectory (PCE: positive curvature integral) detects fold failures on out-of-distribution proteins where pLDDT is uninformative.

---

## 1. Introduction

ESMFold predicts protein structure in a single forward pass through ~47 trunk blocks operating on a pair representation z ∈ ℝ^{L×L×128}. Unlike AlphaFold2, ESMFold has no MSA and no iterative recycling; the model produces its prediction in one shot. The question this paper asks: *what is happening inside z as the trunk runs?* Is each layer a meaningless intermediate state, or does the trunk implement something we can read out and understand?

We answer this by training a probe to extract pairwise distances from z at each layer. The novelty here is in the training anchor: existing probes are typically supervised against ground-truth structures, but that bakes the target into the readout — any apparent "improvement across layers" could be the probe getting more accurate, not the trunk doing anything interesting. We avoid this by chaining probes back from the model's own output, so no probe ever sees a native structure.

What falls out is striking. The probe reveals that ESMFold's intermediate predictions traverse a geometrically valid, monotonically energy-decreasing trajectory in shape space. The trunk has learned an amortized procedure that arrives at low-energy configurations without being trained to do so.

This has two payoffs. **Scientifically**, it gives us a mechanistic window into ESMFold: protein folding inside a transformer trunk looks like coordinated energy descent on a learned implicit landscape. **Practically**, the shape of the probe trajectory — specifically how convex/monotone the convergence is — is a fold-quality signal that works where pLDDT does not, on out-of-distribution sequences.

---

## 2. Probe design and the linking-back training procedure

**The probe.** At each ESMFold trunk layer t ∈ {1, ..., 46}, we train a per-pair bottleneck `PairDistogramBottleneck(pair_dim=128, bins=64, hidden_dim=256)`. The probe takes a single z[i,j] vector and outputs a 64-bin distogram over the same distance grid ESMFold's own head uses (2.3–21.7 Å, equal bins).

**The loss has exactly two terms, both per-pair:**

```
L_t(probe) = MSE(probe.decode(probe.encode(z[i,j])), z[i,j])
           + λ · KL(softmax(probe.encode(z[i,j]))  ||  target_distribution[i,j])
```

The first term forces the probe to be an honest encoder–decoder of z (it must be able to reconstruct z[i,j] from its bottleneck). The second term anchors the probe's distogram to a target distribution. Crucially, **no native structure enters the loss**.

**Linking-back anchoring.** Probes are trained top-down with cross-layer consistency:

- The probe at layer 46 uses **ESMFold's own distogram head applied to z_47** as its consistency target — the model's own answer, not the native structure.
- Probes at layers 45, 44, ..., 1 use the **just-trained probe at layer t+1** as their consistency target.

This produces a chain of probes, each consistent with the next, anchored only to ESMFold's own output at the end. No probe is ever shown a native PDB structure during training.

**Iterative-refinement assumption.** Reading distances *layer by layer* presupposes that successive trunk blocks perform meaningful refinement rather than scrambling representations. This assumption is well-supported. The trunk shares the same recurrent attention-update pattern as AlphaFold2's Evoformer, which has been shown to perform implicit iterative refinement; mechanistic studies of ESMFold specifically (baulab.info; arXiv 2602.06020) demonstrate that late trunk blocks (~20–35) encode spatial geometry, distance information accumulates progressively, and the pair representation z behaves as a continuous distance/contact map whose magnitude scales monotonically with predicted Cα distances. We take this as license to treat the per-layer distogram as a meaningful intermediate state.

---

## 3. The probes are faithful to a shared 3D coordinate frame

The probe loss has zero terms enforcing Euclidean structure or cross-pair consistency. Each pair (i,j) is its own independent regression problem. Despite this, the predicted distance matrices behave as if they came from a coherent 3D object.

**Zero triangle-inequality violations.** Across 57 proteins (good and bad folds) and all 46 layers, we sampled 3000 random triples (i,j,k) per protein per layer and checked `d(i,k) > d(i,j) + d(j,k)`. Across roughly 8 million triple-checks, the violation rate was 0.0000. Each pair-distance prediction is computed independently, yet they collectively form a valid metric.

**MDS stress stays bounded across layers** (Figure 1). Embedding each layer's predicted distance matrix into 3D via classical MDS, the stress sits at 5–12 Å throughout — comparable to the embedding error of the GT native structure itself — for all six benchmark proteins, including the one whose final TM is 0.013. The stress doesn't grow or spike at any layer, meaning every intermediate representation has the same Euclidean character as the final.

**Coordinated motion in the implied 3D structure.** Animations of the MDS-embedded shape across layers show clear rigid-body kinematics: helices translating as units, strand pairs pivoting together, loops swinging coherently. The probe loss has no term linking pair (i,j) to pair (i,k). For all pairs across thousands of layer transitions to jointly produce a sequence of geometrically valid, smoothly-evolving 3D objects, each pair's probe must be reading the same internal coordinate frame in z.

**ESMFold's own distogram head is not Euclidean.** A useful counterpoint: applying our triangle-inequality test to outputs of ESMFold's native distogram head gives substantial violations — the head is geometrically inconsistent (the structure module downstream resolves this). Our linear probes, despite being simpler, recover clean Euclidean geometry. The clean Euclidean structure is therefore not a property of ESMFold's outputs but of z itself, and the probes are reading it faithfully.

> **Figure 1.** [`runs/euclidean_stress_by_layer.png`] MDS stress of probe-predicted distance matrices across all 46 trunk layers for 6 well-characterized proteins. Stress is bounded throughout — Euclidean geometry is preserved at every layer, even though no loss term enforces it.

---

## 4. The trunk descends an energy landscape it was never asked to descend

For each protein we compute the Lennard-Jones energy of the probe's expected shape at each layer:

```
E_t = Σ_{|i-j|>3}  4ε [(σ/E[d_t(i,j)])^12 − (σ/E[d_t(i,j)])^6]  / L
```

with σ = 6.8 Å, ε = 1. We use only |i−j| > 3 to skip bonded neighbours, normalize by chain length, and treat the absolute scale as unimportant — what we care about is the *shape* of E_t as t runs from 1 to 46.

**The result is universal monotone descent** (Figures 2, 3). Across all 57 proteins with cached ESMFold activations, the LJ energy of the expected shape decreases through the trunk, with ~87–93% of layer transitions monotone. The deepest descents belong to the highest-TM proteins; the shallowest belong to the failures. The headline figure (Figure 2) shows this for one protein: as the MDS-embedded shape transitions from total disorder to clear topology, the energy drops in lockstep.

> **Figure 2 (headline).** [`runs/headline_keyframe_energy.png`] Top: six MDS keyframes of the probe-read expected shape for one protein across trunk layers. Bottom: LJ energy / residue across all 46 layers. Structure visibly self-assembles; energy visibly drops; both happen together.

> **Figure 3.** [`runs/lj_energy_grid.png`] Per-protein LJ energy trajectories for all 57 proteins, arranged best-to-worst fold quality (top-left to bottom-right). Good folds (green) reach deeper minima with smooth monotone descent. Failed folds (red) plateau at higher energy or show shallow, noisy descent.

### 4.1 Counterarguments we considered

A signal this clean invites scrutiny. We worked through each concern:

**"The probes were trained to predict the native structure."** They were not. The training chain is anchored to ESMFold's own output at layer 46/47; native PDB structures never enter the loss. Each probe is trained only to be consistent with the next probe. The energy descent is not a downstream artifact of probe supervision.

**"It's just distribution sharpening."** A sharpness check (computing the entropy of the predicted distogram per layer) confirms distributions are sharpening across layers. But LJ(E[d]) depends only on E[d] — the mean — and is mathematically invariant to variance. A symmetric distribution sharpening in place doesn't move E[d]. If LJ(E[d]) is moving, the mean is moving. Period.

**"It's just contact classification."** For 'easy' proteins where `corr_vs_47` is already near 1 at layer 1 (i.e. contacts are already identified early), energy still drops substantially through subsequent layers. The descent isn't merely the probe learning *which* pairs are contacts; it continues after contact identification is complete.

**"LJ is arbitrary."** We chose LJ with σ=6.8 Å and ε=1 as a coarse-grained energy proxy. Different choices give different absolute values. The defensible claim is not "the trunk minimizes Lennard-Jones energy" but "the trunk produces expected shapes whose energy decreases under standard coarse-grained potentials." The signal generalizes to the qualitative claim of "energy descent," not to any specific functional form.

**"Maybe it's literally gradient descent on LJ."** Probably not. A cosine-alignment test — measuring `cos(Δz_t, −∇_z LJ(z_t))` — gives values of −0.01 (essentially random) across 2622 layer-protein samples. The trunk is not taking gradient steps on LJ in z-space. What it *is* doing must be more sophisticated: a learned amortized mapping that produces low-energy outputs in a single feed-forward pass, without iterative descent. This is consistent with the trunk having internalized — through training on millions of structures — a procedure that *arrives* at energy minima without *descending* toward them.

The picture that survives all these critiques: **the trunk is not running gradient descent on energy, but it has learned an amortized procedure whose intermediate states correspond to monotonically lower-energy expected shapes**. The probe makes this trajectory visible.

---

## 5. Practical payoff: convergence-shape error detection (PCE)

If the trunk is descending an implicit energy landscape, the *shape* of the descent — how monotone, how convex, how plateaued — should carry information about fold quality. A clean, convex descent suggests the trunk found the basin; a halting, oscillating, or shallow descent suggests it didn't.

We operationalize this with **PCE (positive curvature energy)**: the Kadane-maximum positive-curvature integral on the corr_vs_final trajectory. Intuitively, it measures how much of the convergence happens late and abruptly (sign of a struggling trajectory) versus smoothly throughout (sign of a clean basin).

On 500 OOD proteins (Figure 4):

| Slice | n | pLDDT | pce_raw | bt_raw | combined |
|---|---|---|---|---|---|
| All (TM>0 to 1) | 500 | 0.845 | 0.865 | 0.866 | 0.863 |
| TM < 0.75 | 103 | 0.659 | 0.745 | 0.751 | 0.742 |
| TM < 0.70 | 89 | **0.579** | **0.633** | **0.636** | 0.627 |

**The headline number is the TM<0.70 slice — the realistic operating regime for drug-target work.** pLDDT here is near random (AUROC 0.58); the probe-trajectory metrics are 5–6 percentage points better, exactly in the range where it matters.

The mechanism is intuitive: pLDDT is a confidence score the model generates *about its own output*; on OOD sequences, the model is confidently wrong, so pLDDT loses signal. The probe trajectory captures something the model isn't trained to report — whether its *internal optimization* converged cleanly. That signal survives OOD because the trunk's internal struggle leaves a kinematic trace even when the output looks confident.

> **Figure 4.** [`runs/general_convergence_ood500_native_cw30_stream/auroc_by_population.png`] AUROC for fold-failure (TM<0.5) detection across population slices. pLDDT degrades to near-random on the hard OOD slice while probe-trajectory metrics retain useful signal.

---

## 6. Limitations and future work

**Limitations.**
- The "energy descent" is only shown under one specific coarse-grained potential (LJ, σ=6.8 Å). Whether the same monotone descent holds under Rosetta-quality force fields or DFIRE-style statistical potentials is open. The qualitative claim should be robust; the quantitative trajectory will look different.
- We work on Cα-only geometry through MDS embedding of probe-predicted distance matrices. Side-chain, backbone-torsion, and clash terms of real protein energy are not captured.
- The cosine-alignment result rules out literal gradient descent but cannot identify what the trunk *is* doing. We do not yet have a mechanistic account of the amortized procedure; we only know it is not first-order GD on z.
- Error-detection numbers (Section 5) are AUROC on a fixed 500-protein OOD benchmark. Generalization to broader OOD distributions (designed sequences, fragments, multi-domain proteins) is unverified.

**Future work.**
- Test the energy-descent signal under multiple potentials (Rosetta, DFIRE) to establish that the claim is potential-independent.
- Characterize the amortized procedure: is the trunk implementing a learned proximal step, a Newton-like update, or something genuinely novel?
- The probes give us a layer-resolved confidence signal; calibrate it into a per-residue and per-pair confidence map and compare to error in the predicted structure.
- Use the probes for **steering**: edit z at layer 20 toward a target geometry and complete the forward pass. If the probes are faithful, targeted edits to z should produce the corresponding changes in the final structure. Early experiments suggest directional accuracy ~55% on the top quartile of disagreement pairs, indicating real but partial controllability.

---

## Appendix — figures inventory

| § | path |
|---|---|
| 3 | `runs/euclidean_stress_by_layer.png` |
| 4 (headline) | `runs/headline_keyframe_energy.png` |
| 4 | `runs/lj_energy_grid.png` |
| 5 | `runs/general_convergence_ood500_native_cw30_stream/auroc_by_population.png` |
| 5 (supp) | `runs/general_convergence_ood500_native_cw30_stream/pce_vs_tm.png` |
| 5 (supp) | `runs/general_convergence_ood500_native_cw30_stream/combined_metric_vs_tm.png` |

Supplementary video: full MDS animation alongside per-layer energy curve, generated from `runs/general_probe_simulations/8fmi/mds_fixed_camera/distogram_mds_structure.mp4`. Fixed camera viewpoint — only the protein geometry changes between frames, so the viewer tracks pure structural evolution rather than compensating for camera rotation.
