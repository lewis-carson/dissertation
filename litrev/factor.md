# Factor model and early-learned evaluation factors

This note summarizes which chess-evaluation factors are typically learned early by a supervised evaluation network, a compact multiplicative "factor model" explaining why prioritized / non-IID sampling can produce the largest effective learning-rate increase, suggested experiments to verify these claims, and practical recommendations.

## TL;DR
- Early-learned factors: material balance, immediate captures/trades (short tactics), mobility/activity, and clear king-safety failures. These features are high signal-to-noise and low representational complexity.
- The largest effective learning-rate increase comes from increasing expected squared gradient magnitude via prioritized sampling (upweighting high-gradient/high-loss samples) rather than simply raising the optimizer learning-rate globally.
- Use partial importance-sampling correction, anneal priorities, and preserve diversity to avoid bias and instability.

## 1. Which factors are learned early — intuition

Training order is governed by signal-to-noise ratio (SNR), representational complexity, and frequency. Low-complexity, high-SNR patterns that appear often are learned first.

Typical early factors:

1. Material balance
	- Extremely predictive of position value; often a near-linear relationship with target value.
	- Low representational complexity: simple counts and linear combinations of piece occupancies.

2. Immediate captures and forced trades (short tactics)
	- Produce large, abrupt changes in value; gradients from these examples are large and consistent.
	- Shallow receptive fields / local patterns suffice to detect these.

3. Mobility / piece activity
	- Correlates well with advantage; many mobility features are simple functions of piece squares and moves.

4. Obvious king-safety failures (exposed king, missing pawn cover)
	- When king safety is grossly violated the label shifts strongly; detectable with local patterns around the king.

Mid-to-late factors:

- Pawn structure subtleties (isolated/doubled pawns, long-term pawn majorities) — informative but sometimes subtle and long-term.
- Strategic plans, prophylaxis, and deep positional concepts — require deeper composition and more training to discover.

Notes on dataset bias and noise:

- If the dataset concentrates on a phase (e.g., endgames), phase-specific features can be learned earlier.
- Label noise (e.g., noisy engine scores, shallow search labels) reduces SNR and can delay learning of weaker signals.

## 2. Compact factor-model for effective learning progress

Goal: model why prioritized/non-IID sampling can amplify effective learning per optimizer step.

Definitions (scalars for intuition):
- α: base optimizer learning rate (global hyperparameter).
- p_i: sampling probability for example i.
- g_i: expected gradient vector from example i; ||g_i|| its magnitude.
- S: sampling informativeness scalar ≈ E_p[||g||^2] = Σ_i p_i * ||g_i||^2.
- κ: curvature factor (higher curvature → smaller effective step for same gradient magnitude).
- D: diversity factor, 0 < D ≤ 1, capturing reduced generalization if sampling is overly narrow.

Proxy for effective progress per optimizer step (scalarized):

	 progress ≈ α * (S / κ) * D

Where S = Σ_i p_i * ||g_i||^2. In words: the base step α is modulated by how much useful squared gradient the sampler yields, penalized by curvature and reduced by lack of diversity.

Why prioritized sampling helps:

- If a small fraction of examples have much larger ||g|| (e.g., tactical positions), increasing their p_i raises S substantially.
- The naive gain ratio ≈ S_prior / S_uniform can be several× when the gradient distribution is heavy-tailed.

Importance-sampling (IS) and bias correction:

- To keep gradient estimates unbiased you may scale per-example gradients by w_i ∝ 1 / (N * p_i). This reduces bias but also reduces the naive S gain.
- In practice, partial correction is used (e.g., priority exponentization and IS exponent β annealed to 1) to keep much of the gain while limiting bias.

Tradeoffs to watch:

- Variance: concentrating on high-||g|| samples increases gradient variance; use clipping or smaller α if instability appears.
- Curvature: if high-||g|| corresponds to high-curvature regions, effective progress may be lower than naive S suggests unless optimizer/hyperparams are adjusted.

## 3. Numeric intuition / toy example

- Suppose 5% of samples have ||g||^2 = 25 and 95% have ||g||^2 = 0.8.
- Uniform S_uniform = 0.05*25 + 0.95*0.8 = 1.25 + 0.76 = 2.01.
- If prioritized sampling boosts the 5% group to 50% sampling probability, S_prior = 0.5*25 + 0.5*0.8 = 12.9 → ≈6.4× S_uniform.

After partial IS correction / rescaling this gain is reduced but often remains multiple× for the features tied to high-gradient samples (e.g., tactical motifs).

## 4. Experiments to verify which factors are learned early and to measure gains

Instrumentation / probes:

1. Linear probes per factor
	- At checkpoints, train small linear classifiers/regressors from intermediate layer activations to predict simple factors: material diff, presence of passed pawn, existence of an immediate capture, king safety score.
	- Plot probe performance vs training steps. Early-learned factors become linearly decodable quickly.

2. Tagged gradient-norm / loss tracking
	- Tag training positions with factor labels (material imbalance, tactical flag, passed pawn, etc.). Log average loss and average ||g|| per-tag over time.
	- Early-learned tags will show rapid loss reduction and shrinking gradient norms.

3. Prioritized vs uniform ablation
	- Run matched experiments: baseline uniform sampling vs prioritized sampling (priority by recent loss or ensemble disagreement).
	- Measure: validation loss vs steps, probe curves, and per-tag loss reduction rate.
	- Report effective acceleration: steps_to_threshold_uniform / steps_to_threshold_prior.

4. IS / priority hyperparameter sweep
	- Sweep priority exponent (α_p) and IS exponent (β). Find region where speedup is maximized without harming validation/generalization.

Metrics to report:
- Validation MSE or value-RMSE over steps.
- Probe accuracy/AUROC per factor vs steps.
- Estimated S = E_p[||g||^2] under sampling distribution (can be computed from logged gradients).
- Loss reduction per-sample for each factor group.

## 5. Practical recommendations

- Start with partial priority (priority^α_p with α_p≈0.6–1.0) and β annealed from a smaller value toward 1 across training.
- Keep a background uniform sampling fraction or floor probability to preserve common low-signal features and avoid forgetting.
- Monitor probe curves for factors; if prioritized sampling accelerates some factors but slows others, introduce a mixing schedule (e.g., prioritized early, then more uniform later).
- Use gradient clipping or reduce global α if variance rises when priorities are applied.
- Prefer adaptive optimizers and track curvature diagnostics (e.g., loss curvature estimated by small finite-difference steps) when tuning.

## 6. Short next steps you can apply to the repo

1. Add lightweight probes (linear heads) for: material diff, immediate capture flag, passed-pawn flag, simple king-safety score. Log probe accuracy per checkpoint.
2. Tag a subset of training data with these simple labels (script if not already present) so you can group gradient/loss statistics.
3. Implement a prioritized buffer with priority = recent loss^α_p and optional IS weights with β scheduling.

If you want, I can also generate the exact probe-code skeleton and minimal edits to `train.py` in this repo to log these metrics.

---

References and deeper derivations can be added here later; this file is intended as a practical summary to include in the literature review or experimental plan.
