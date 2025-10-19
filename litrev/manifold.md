Manifold aware sampling for faster training of position evaluation networks

Improving training efficiency of position evaluation networks through manifold-aware sampling.
> 
> I'll explain what manifold-aware sampling means, why it helps, how to implement it (concise recipe + pseudocode), practical sampling strategies, evaluation metrics, and common risks + mitigations — with brief chess-specific notes tied to your training_fast.md draft.
> 
> ## Quick definition
> Manifold-aware sampling = generating or selecting training examples that are both (a) likely/realistic according to a learned model of the data manifold, and (b) high-utility (e.g., high information, high gradient norm, high ensemble disagreement). The goal is to concentrate training on “useful” regions of the plausible-data manifold rather than on adversarial or out-of-distribution noise.
> 
> ## Intuition
> - Real data lie on a lower-dimensional manifold in input/activation space. Blind maximization of an information score often heads off the manifold into adversarial directions.
> - Manifold-aware methods constrain selection/optimization by a plausibility density p(x) or p(z) (latent density). They find examples that are both plausible and informative: maximize InfoScore(x) − λ · (−log p(x)).
> - Practically: model p (VAE/flow/contrastive encoder), score informativeness (gradient norm, disagreement, predicted value-change), then sample/optimize in latent space with a plausibility penalty.
> 
> ## Minimal contract (2–3 bullets)
> - Inputs: dataset of real positions, base model fθ, generative plausibility model (encoder/decoder + density).
> - Outputs: prioritized buffer of plausible, high-information positions (or a sampler to draw them online).
> - Success: improves sample efficiency (lower validation loss / higher Elo per self-play step) without increasing fraction of implausible/illegal examples above a small threshold.
> 
> ## Concrete recipe (short)
> 1. Fit a generative density on real positions: encoder E, decoder D, latent density p(z) (VAE / Flow / Normalizing Flow / latent diffusion).
> 2. Define InfoScore(s) (examples: ||∇θ L(fθ(s))||, ensemble disagreement, KL-change on one-step rollout, or predicted learning progress).
> 3. Find candidates in latent space:
>    - Optimize z via gradient ascent on J(z) = InfoScore(D(z)) − λ log p(z).
>    - Or run an MCMC / tempered sampler over energy E(z) = −InfoScore(D(z)) + λ log p(z) to sample high-score but plausible z.
> 4. Decode z → s, run legality/heuristic filters (repair small illegalities, enforce material balance).
> 5. Insert s into prioritized replay buffer keyed by InfoScore(s). During training sample from this buffer with some probability and always mix with normal self-play data.
> 
> Short pseudocode:
> ```text
> Train f_θ on real data.
> Train VAE/flow: E,D,p(z) on positions.
> for i in 1..N:
>     z ~ proposal()
>     z ← gradient_ascent(z, objective = InfoScore(D(z)) - λ * log p(z))
>     s = D(z)
>     if legal(s): buffer.add(s, priority = InfoScore(s))
> Train f_θ using mixture of self-play + prioritized buffer
> ```
> 
> ## Why optimize in latent/activation space
> - Latent spaces are smoother and more semantically meaningful than raw input pixels/bitstrings; gradient moves are likely to stay on-manifold.
> - Density p(z) gives an explicit plausibility penalty; using −log p prevents degenerate, adversarial solutions.
> - Activation-space sampling (e.g., internal layer activations) can target model weaknesses while remaining plausible in representation space.
> 
> ## Sampling strategies
> - Latent gradient-ascent with λ sweep (controls tradeoff info vs plausibility).
> - MCMC/tempered sampling: control exploitation/exploration.
> - Reweight real positions by InfoScore (cheap alternative) — top-k replay.
> - Ensemble disagreement + density threshold: pick examples with high disagreement but p(z) above ε.
> - Rejection sampling: propose from p(z) and accept with probability proportional to InfoScore.
> 
> ## Metrics & evaluation
> - Sample-efficiency: validation loss and Elo as a function of training steps.
> - Distribution overlap: KL(p_generated || p_real) in latent or feature space.
> - Diagnostics: fraction illegal/implausible, gradient norms over time, per-layer sensitivity, ensemble calibration.
> - Ablations: replay-only vs synthetic-only, λ sweep, buffer size, replay probability schedule.
> 
> ## Risks and mitigations
> - Risk: generator produces adversarial or unrealistic positions.
>   - Mitigation: strong λ; repair heuristics; discriminator to detect exploitation; reject low p(z) candidates.
> - Risk: overfitting to corner cases / distribution shift.
>   - Mitigation: always mix with real self-play; anneal replay probability; keep buffer diverse; use holdout validation from true self-play.
> - Risk: optimization exploits model blind spots (high InfoScore but useless).
>   - Mitigation: use ensemble disagreement, human checks, secondary loss (e.g., discriminator), and small-step latent nudges rather than large jumps.
> - Risk: legal/semantic invalidity (especially in chess).
>   - Mitigation: legality projection and move-history heuristics; enforce material parity or simple replayability checks.
> 
> ## Chess-specific notes (practical)
> - Always run a legality pass. For small illegality (illegal pawn placements, single-square errors) apply repair heuristics; otherwise reject.
> - Use game-plausibility heuristics: material balance, common move sequences, typical ply depth for midgame vs endgame.
> - To improve endgame learning, bias sampling towards low-material, high-info endgame latent modes rather than trying to create contrived illegal endgames.
> - Useful InfoScores in chess: high policy entropy change after search, value uncertainty across an ensemble, large gradient norm on value/policy heads.
> 
> ## Edge cases to watch
> - Empty or degenerate latent regions (p(z) tiny everywhere) — choose a robust proposal.
> - Generator collapse (VAE / flow poorly fit) — monitor reconstruction quality and latent density calibration.
> - High computational cost of InfoScore (requires backprop per candidate) — approximate with cheaper proxies (ensemble disagreement, critic uncertainty) for large-scale loops.
> 
> ## Short reading map
> - Your training_fast.md already captures this idea: fit a generative density, optimize in latent for InfoScore − λ log p(z), filter for legality, and replay. Related prior work: Goodfellow et al. (adversarial examples), Schaul et al. (prioritized experience replay), Li et al. (value perspective on PER).
> 
> If you want, I can:
> - produce a small runnable prototype that fits a simple VAE on chess-lite features and runs latent gradient-ascent to generate candidates, or
> - write the prioritized-replay integration pseudocode for your training loop (PyTorch/TensorFlow), or
> - help pick practical InfoScore proxies that are cheap to compute at scale.
> 
> Which of those would you like next?
> 
> 
> 
> 
> 
> 
> 
> 
> 
> 
> 
> I'll explain what manifold-aware sampling means, why it helps, how to implement it (concise recipe + pseudocode), practical sampling strategies, evaluation metrics, and common risks + mitigations — with brief chess-specific notes tied to your training_fast.md draft.
> 
> ## Quick definition
> Manifold-aware sampling = generating or selecting training examples that are both (a) likely/realistic according to a learned model of the data manifold, and (b) high-utility (e.g., high information, high gradient norm, high ensemble disagreement). The goal is to concentrate training on “useful” regions of the plausible-data manifold rather than on adversarial or out-of-distribution noise.
> 
> ## Intuition
> - Real data lie on a lower-dimensional manifold in input/activation space. Blind maximization of an information score often heads off the manifold into adversarial directions.
> - Manifold-aware methods constrain selection/optimization by a plausibility density p(x) or p(z) (latent density). They find examples that are both plausible and informative: maximize InfoScore(x) − λ · (−log p(x)).
> - Practically: model p (VAE/flow/contrastive encoder), score informativeness (gradient norm, disagreement, predicted value-change), then sample/optimize in latent space with a plausibility penalty.
> 
> ## Minimal contract (2–3 bullets)
> - Inputs: dataset of real positions, base model fθ, generative plausibility model (encoder/decoder + density).
> - Outputs: prioritized buffer of plausible, high-information positions (or a sampler to draw them online).
> - Success: improves sample efficiency (lower validation loss / higher Elo per self-play step) without increasing fraction of implausible/illegal examples above a small threshold.
> 
> ## Concrete recipe (short)
> 1. Fit a generative density on real positions: encoder E, decoder D, latent density p(z) (VAE / Flow / Normalizing Flow / latent diffusion).
> 2. Define InfoScore(s) (examples: ||∇θ L(fθ(s))||, ensemble disagreement, KL-change on one-step rollout, or predicted learning progress).
> 3. Find candidates in latent space:
>    - Optimize z via gradient ascent on J(z) = InfoScore(D(z)) − λ log p(z).
>    - Or run an MCMC / tempered sampler over energy E(z) = −InfoScore(D(z)) + λ log p(z) to sample high-score but plausible z.
> 4. Decode z → s, run legality/heuristic filters (repair small illegalities, enforce material balance).
> 5. Insert s into prioritized replay buffer keyed by InfoScore(s). During training sample from this buffer with some probability and always mix with normal self-play data.
> 
> Short pseudocode:
> ```text
> Train f_θ on real data.
> Train VAE/flow: E,D,p(z) on positions.
> for i in 1..N:
>     z ~ proposal()
>     z ← gradient_ascent(z, objective = InfoScore(D(z)) - λ * log p(z))
>     s = D(z)
>     if legal(s): buffer.add(s, priority = InfoScore(s))
> Train f_θ using mixture of self-play + prioritized buffer
> ```
> 
> ## Why optimize in latent/activation space
> - Latent spaces are smoother and more semantically meaningful than raw input pixels/bitstrings; gradient moves are likely to stay on-manifold.
> - Density p(z) gives an explicit plausibility penalty; using −log p prevents degenerate, adversarial solutions.
> - Activation-space sampling (e.g., internal layer activations) can target model weaknesses while remaining plausible in representation space.
> 
> ## Sampling strategies
> - Latent gradient-ascent with λ sweep (controls tradeoff info vs plausibility).
> - MCMC/tempered sampling: control exploitation/exploration.
> - Reweight real positions by InfoScore (cheap alternative) — top-k replay.
> - Ensemble disagreement + density threshold: pick examples with high disagreement but p(z) above ε.
> - Rejection sampling: propose from p(z) and accept with probability proportional to InfoScore.
> 
> ## Metrics & evaluation
> - Sample-efficiency: validation loss and Elo as a function of training steps.
> - Distribution overlap: KL(p_generated || p_real) in latent or feature space.
> - Diagnostics: fraction illegal/implausible, gradient norms over time, per-layer sensitivity, ensemble calibration.
> - Ablations: replay-only vs synthetic-only, λ sweep, buffer size, replay probability schedule.
> 
> ## Risks and mitigations
> - Risk: generator produces adversarial or unrealistic positions.
>   - Mitigation: strong λ; repair heuristics; discriminator to detect exploitation; reject low p(z) candidates.
> - Risk: overfitting to corner cases / distribution shift.
>   - Mitigation: always mix with real self-play; anneal replay probability; keep buffer diverse; use holdout validation from true self-play.
> - Risk: optimization exploits model blind spots (high InfoScore but useless).
>   - Mitigation: use ensemble disagreement, human checks, secondary loss (e.g., discriminator), and small-step latent nudges rather than large jumps.
> - Risk: legal/semantic invalidity (especially in chess).
>   - Mitigation: legality projection and move-history heuristics; enforce material parity or simple replayability checks.
> 
> ## Chess-specific notes (practical)
> - Always run a legality pass. For small illegality (illegal pawn placements, single-square errors) apply repair heuristics; otherwise reject.
> - Use game-plausibility heuristics: material balance, common move sequences, typical ply depth for midgame vs endgame.
> - To improve endgame learning, bias sampling towards low-material, high-info endgame latent modes rather than trying to create contrived illegal endgames.
> - Useful InfoScores in chess: high policy entropy change after search, value uncertainty across an ensemble, large gradient norm on value/policy heads.
> 
> ## Edge cases to watch
> - Empty or degenerate latent regions (p(z) tiny everywhere) — choose a robust proposal.
> - Generator collapse (VAE / flow poorly fit) — monitor reconstruction quality and latent density calibration.
> - High computational cost of InfoScore (requires backprop per candidate) — approximate with cheaper proxies (ensemble disagreement, critic uncertainty) for large-scale loops.
> 
> ## Short reading map
> - Your training_fast.md already captures this idea: fit a generative density, optimize in latent for InfoScore − λ log p(z), filter for legality, and replay. Related prior work: Goodfellow et al. (adversarial examples), Schaul et al. (prioritized experience replay), Li et al. (value perspective on PER).
> 
> If you want, I can:
> - produce a small runnable prototype that fits a simple VAE on chess-lite features and runs latent gradient-ascent to generate candidates, or
> - write the prioritized-replay integration pseudocode for your training loop (PyTorch/TensorFlow), or
> - help pick practical InfoScore proxies that are cheap to compute at scale.
> 
> Which of those would you like next?
>


I'll give a concise, intuitive explanation, then a short formal definition and examples, and finally why the idea matters for machine learning (and your chess project).

## Intuition — the idea in plain language
A manifold is a shape that might be curved or complicated globally, but looks like ordinary flat Euclidean space if you zoom in close enough at any point. Think of the surface of the Earth: from far away it’s a sphere (curved), but any small patch looks like a flat plane. That “locally flat” property is the heart of a manifold.

Key takeaways:
- Locally: looks like $\mathbb{R}^n$ (ordinary n‑dimensional space).
- Globally: can be curved, looped, or have a complex topology.
- Dimension: the number $n$ is the manifold’s dimension (a circle is 1D, a sphere surface is 2D).

## Formal (short) definition
A topological n‑manifold M is a space such that:
- M is Hausdorff and second-countable (technical regularity conditions), and
- every point p ∈ M has a neighborhood homeomorphic (shape-preserving bijection) to an open subset of $\mathbb{R}^n$.

A smooth (differentiable) manifold adds an atlas of charts whose transition maps are smooth. This lets you take derivatives on M, define tangent spaces $T_pM$, and do calculus on the manifold.

If you add an inner product on each tangent space varying smoothly you get a Riemannian manifold, which gives notions of length, angle, volume and local distance (metric) via a Riemannian metric $g$.

## Simple examples
- Circle $S^1$: a 1‑dimensional manifold. Parametrisation: $\theta \mapsto (\cos\theta,\sin\theta)$. Locally looks like a line.
- Sphere surface $S^2$: a 2‑dimensional manifold (locally like a plane).
- Torus: a donut shape — globally different topology but locally 2D.
- Plane $\mathbb{R}^n$: the simplest manifold (both locally and globally Euclidean).

## Manifolds in machine learning (why the concept matters)
- Data manifold hypothesis: real-world high-dimensional data (images, audio, chess positions encoded as vectors) concentrate near a lower-dimensional manifold inside the ambient high-dimensional space. That is, the intrinsic dimensionality is much smaller than the raw input dimension.
- Dimensionality reduction / manifold learning methods try to recover a coordinate system on that manifold: PCA, Isomap, LLE, t‑SNE, UMAP, autoencoders, VAEs, normalizing flows, and diffusion models.
- Generative models often learn a mapping from a low‑dimensional latent space $z\in\mathbb{R}^k$ (a proxy for the manifold coordinates) to high‑dimensional observations $x$. If the model is good, samples from the latent prior map to points on (or near) the data manifold.

## Why people talk about a “chess position manifold”
- The full space of chess states is combinatorial and huge (legal positions vs. all possible bitboard patterns), but plausible human/computer-play positions are far fewer and structured.
- Intuitively, plausible positions occupy a much smaller region in the full encoding space — they have rules, piece patterns, pawn structures, typical development phases, etc. That concentration behaves like a manifold (or at least a low‑dimensional structure) which you can try to learn.
- Modeling that structure helps: realistic sampling, generating synthetic but plausible positions, and designing sampling strategies that prioritize informative regions.

## Practical implications for your project
- If positions lie (approximately) on a manifold, a generative model (VAE/flow) can estimate density or learn latent coordinates. Those can be used for:
  - estimating plausibility (low density → unlikely position),
  - sampling new plausible positions in underrepresented regions,
  - weighting training examples by manifold density or by distance in latent space (to diversify the training set).
- Be cautious: chess positions are discrete and constrained (not a perfect smooth manifold). The manifold assumption is an approximation that’s often useful but not exact.

## Edge cases and limitations
- Discreteness: purely combinatorial constraints can violate smoothness assumptions.
- Multimodality / complex topology: latent space must capture holes, loops (topology matters).
- Boundaries and rare events: positions at the edge of plausibility may be important but hard to model.
- Model misspecification: a poorly trained VAE/flow can misrepresent density, so weighting by it can mislead training.

## Quick summary
A manifold is a space that is locally like $\mathbb{R}^n$ but may be globally curved/complex. In ML we use manifold ideas to exploit a lower intrinsic dimensionality in high‑dimensional data. For chess, thinking in manifold terms helps you build generative models and sampling strategies that focus on plausible, informative positions — but remember it’s an approximation and must be validated empirically.

Would you like me to:
- add a short explanatory paragraph about manifolds to litrev.tex (tailored to chess), or
- show a short example (e.g., circle param & a tiny Python snippet illustrating local linearity or an autoencoder toy)?