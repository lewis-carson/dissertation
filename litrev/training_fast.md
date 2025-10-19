> // ...existing code...
> The most information encoded in certain positions (like as in surprise?). Or information theory - anything which shifts the gradients about in surpising ways. Think of synthetic data based on this - positions which are close to this. Think about the manifold of plausible activations - generate synthetic data which is in the most complex, highly non-linear parts of the manifold. Selectively replay these? Might make training more efficient.
> 
> Proposed method (concise)
> - Define an "information score" for a state s:
>   - gradient-based: ||∇_θ L(f_θ(s))|| (or ||∇_s L||) or Fisher information approximations;
>   - surprise-based: KL or predictive-change after one-step perturbation / rollout;
>   - activation-curvature: trace(Hessian) or layerwise activation-sensitivity.
> - Model the activation/position manifold with a generative density (VAE, flow, or contrastive encoder). Use density p(z) in latent space to estimate plausibility; rare/low-density z indicate complex/nonlinear regions.
> - Synthetic-data generation:
>   1. Train generative model on real positions.
>   2. Find latent z that maximize information score under a plausibility penalty: z* = argmax_z [InfoScore(decode(z)) - λ log p(z)].
>   3. Decode z* → position p; project to legality (small repair heuristics) and filter by game plausibility (material balance, legal move history heuristics).
>   4. Add p to a prioritized replay buffer keyed by InfoScore.
> - Selective replay / curriculum: sample more frequently from high-InfoScore buffer entries, interleave with normal self-play data to avoid distribution shift.
> - Alternative cheap heuristics: sample real positions and rank by gradient-norm / disagreement among ensemble heads / temperature of policy logits; use top-k for replay.
> 
> Mini pseudocode
> - Train base model f_θ on standard data.
> - Fit VAE/flow on positions → encoder E, decoder D, density p(z).
> - For i in 1..N:
>     z ← sample_proposal()
>     z ← gradient_ascent(z, objective = InfoScore(D(z)) - λ log p(z))
>     s ← D(z); if legal(s) add_to_buffer(s, InfoScore(s))
> - Retrain / fine-tune f_θ using prioritized replay from buffer.
> 
> Evaluation (recommended)
> - Sample-efficiency: validation loss and Elo vs number of self-play steps.
> - Ablations: replay-only, synthetic-only, λ sweep (plausibility penalty), buffer size.
> - Diagnostic: distribution overlap (KL) between generated and real positions; gradient norms; activation diversity; per-layer sensitivity.
> - Safety checks: fraction of illegal / implausible positions; move legality tests; human-inspection of a small sample.
> 
> Pitfalls & mitigations
> - Generator can produce unrealistic/illegal positions → enforce legality repair and λ strong enough to keep plausibility.
> - Overfitting to synthetic corner cases → always mix with real self-play and anneal replay probability.
> - Optimization in input or activation space may exploit model blind spots; use ensemble disagreement and a discriminator to detect exploitative examples.
> 
> Short benefits summary
> - Prioritizing high-information states should concentrate gradient updates where they cause largest representational change, improving sample efficiency.
> - Using a plausibility model keeps training on useful, realizable parts of the manifold rather than out-of-distribution adversarial activations.
> 
> 

targeted hard-example mining


> You can, but you need to understand what that does.
> 
> If you “just change” the most surprising samples—those with highest expected information gain—you’re performing *local input optimization*. That is equivalent to adversarial training or targeted hard-example mining. The risk is you destroy the joint distribution you wanted to learn from.
> 
> Reasoning from first principles:
> 
> 1. **Goal**: maximize expected parameter improvement per unit of data.
>    [
>    \text{InfoGain}(s) = \mathbb{E}*{\theta'}[L(f*{\theta'}(s)) - L(f_\theta(s))]
>    ]
>    High values identify states that would shift beliefs the most.
> 
> 2. **Naive modification**: if you change those states directly, you push the data distribution toward very low-probability or adversarial regions. Gradients then point where the model is unreliable rather than where the environment is informative. That increases gradient variance and can destabilize training.
> 
> 3. **Better approach**: *sample or replay* such states with higher frequency, not alter them arbitrarily. That preserves support overlap with the true data manifold.
> 
> 4. **When modification helps**: if the change is constrained by a plausibility prior (generator, legality heuristic, or density model). Then you can nudge samples along the manifold into nearby higher-information areas without going out-of-distribution.
> 
> So yes, the idea of targeting high-information states is sound. The failure mode is doing it *without* a manifold constraint. You want
> [
> \max_s I(s) \quad \text{s.t. } p_\text{real}(s) > \epsilon
> ]
> not unconstrained maximization.
> 
> In short:
> 
> * Reweighting high-information samples → efficient and stable.
> * Directly modifying them → likely to produce adversarial or implausible data unless you enforce a density constraint.
> 
change is constrained by a plausibility prior

expected learning progress

> Here are arXiv (or preprint) citations supporting those points:
> 
> * Goodfellow et al., *“Explaining and Harnessing Adversarial Examples”*, arXiv:1412.6572 (2014) — shows that unconstrained optimization on input space produces adversarial perturbations that exploit model linearity rather than meaningful structure. ([arXiv][1])
> * Schaul et al., *“Prioritized Experience Replay”*, arXiv:1511.05952 (2015) — proposes replaying “important” transitions more often, as a way to improve sample efficiency without corrupting distribution too aggressively. ([arXiv][2])
> * Li et al., *“Revisiting Prioritized Experience Replay: A Value Perspective”*, arXiv:2102.03261 (2021) — analyzes theoretical link between prioritization and value-of-experience, and bounds thereof. ([arXiv][3])
> 
> If you want more recent or more directly relevant references (information‐gain in latent spaces, manifold‐aware sampling), I can fetch them.
> 
> [1]: https://arxiv.org/abs/1412.6572?utm_source=chatgpt.com "Explaining and Harnessing Adversarial Examples"
> [2]: https://arxiv.org/abs/1511.05952?utm_source=chatgpt.com "Prioritized Experience Replay"
> [3]: https://arxiv.org/abs/2102.03261?utm_source=chatgpt.com "Revisiting Prioritized Experience Replay: A Value Perspective"

Talk about little risk of overfitting to adversarial examples if you mix with normal self-play data.

Already "bound by adversarial perturbations" is because the change is constrained by logical bounds, i.e. legality of moves.

Positions change a lot, but it may help learn bounds like end-games.

Essentially, the repeated augmented data is a form of curriculum learning. The repeated augmented data is a form of search through the space of plausible positions, biased toward high-information areas. 