1412.6572

Talk about "game playing" models where data is plentiful and self-play is possible. Samples are taken from a logically bounded space (legal moves).

How to find "non linearities" in the weights manifold which are "high information" and "plausible".

How to define "high information" - expected learning progress?

Use hand-crafted "expected learning progress" and then a learned model to predict it.

What are some used techniques for measuring "expected learning progress"?

Also symmetry constraints. Think about other types of constraints.

Use fisher info? Symmetry, etc

Work out ways to mix "constrained learning" with raw data which is drawn iid. Instead, the distribution is shifted toward high-information areas, but not too far.

> 
> > Fisher information measures how sensitive the model’s likelihood is to its parameters.
> Formally, for parameters (\theta) and data (x):
> [
> \mathcal{I}(\theta) = \mathbb{E}*{x \sim p(x|\theta)} \big[ (\nabla*\theta \log p(x|\theta))(\nabla_\theta \log p(x|\theta))^\top \big]
> ]
> It is the expected outer product of the score function.
> 
> ---
> 
> ### Intuition
> 
> * Large Fisher values mean small parameter changes cause large shifts in model predictions → parameters are “important” or the region is locally curved.
> * Small Fisher values mean predictions are insensitive to parameter changes → parameters lie in flat or uninformative directions.
> 
> So Fisher ≈ local curvature of the log-likelihood surface.
> In deep nets, the diagonal of the Fisher is often used as a cheap curvature proxy.
> 
> ---
> 
> ### In RL or game-playing
> 
> For a policy (\pi_\theta(a|s)):
> [
> \mathcal{I}(\theta) = \mathbb{E}*{s,a\sim\pi*\theta}!\left[ \nabla_\theta \log \pi_\theta(a|s), \nabla_\theta \log \pi_\theta(a|s)^\top \right]
> ]
> 
> * Used in **natural gradient** methods: update direction (F^{-1}\nabla_\theta J(\theta)) respects the model’s information geometry (Amari, 1998; Kakade, 2001).
> * Used in **EWC (Elastic Weight Consolidation)** to measure which weights are crucial to previous tasks (Kirkpatrick et al., arXiv:1612.00796).
> * Used as a **state importance measure**: high Fisher per-state → high expected gradient magnitude → high expected learning progress.
> 
> ---
> 
> ### Practical estimators
> 
> * **Empirical Fisher**: average over batch samples
>   (\hat{\mathcal{I}} = \frac{1}{N}\sum_i g_i g_i^\top), with (g_i=\nabla_\theta \log p(x_i|\theta)).
> * **Diagonal Fisher**: keep only diag elements for efficiency.
> * **Trace Fisher**: sum of diagonal, scalar curvature estimate.
> 
> These can be computed cheaply from backprop gradients and used as part of your “information score.”
> 


"Respects the model’s information geometry" - natural gradient descent.
Also respects the sample distribution's information geometry.

> 
> You want to balance **bias** (staying on the true data manifold) and **variance reduction** (targeting high-information regions).
> Let (p_\text{real}(s)) be the empirical training distribution and (p_\text{aug}(s)) your synthetic/high-information generator. You train under a mixture
> [
> p_\text{mix}(s) = (1-\rho)p_\text{real}(s) + \rho p_\text{aug}(s)
> ]
> where (\rho) is the *replay ratio*.
> 
> ---
> 
> ### 1. Choosing the replay ratio ( \rho )
> 
> * **Start low**: ( \rho \in [0.1, 0.3] ). Ensures gradient direction is anchored to real data.
> * **Anneal** ( \rho \downarrow 0 ) over time as model stabilizes. Synthetic data are most useful early, when gradients are under-explored.
> * **Adaptive schedule**: raise ( \rho ) when validation loss plateaus (signalling low information in real samples), lower it if KL between real and augmented activations grows too large.
> 
> ---
> 
> ### 2. Weighting within augmented data
> 
> Within (p_\text{aug}), prioritize by an **information score** (I(s)):
> [
> w(s) = \frac{I(s)^\alpha}{\sum_{s'} I(s')^\alpha}
> ]
> Typical (\alpha \in [0.5, 1.0]) (same reasoning as prioritized replay). Use importance correction if you want unbiased gradient estimates: multiply the loss term by (1 / (N w(s))).
> 
> ---
> 
> ### 3. Diagnostics
> 
> Monitor:
> 
> * KL divergence between feature distributions under (p_\text{real}) and (p_\text{aug}). Keep (D_{KL} < 0.1)–0.3 for stability.
> * Validation loss vs (\rho) schedule to detect overfitting to synthetic samples.
> * Gradient variance across minibatches: synthetic replay should reduce it, not inflate it.
> 
> ---
> 
> ### 4. Summary
> 
> | Component                    | Role                             | Typical value           |
> | ---------------------------- | -------------------------------- | ----------------------- |
> | Replay ratio ( \rho )        | Real vs synthetic mix            | 0.1–0.3 start, anneal ↓ |
> | Priority exponent ( \alpha ) | Sharpness of high-info weighting | 0.5–1.0                 |
> | Importance correction        | Optional, keeps unbiased updates | yes for off-policy RL   |
> | KL threshold                 | Prevents distribution drift      | <0.3                    |
> 
> ---
> 
> In short: treat augmented samples as an *accelerant*, not a replacement. Keep most updates on real self-play data, bias replay sampling toward high-information states, and adapt (\rho) based on divergence and validation progress.

Should I uniformly choose the tuple of moves between horizontal symmetry x vertical symmetry x color inversion x rotation x one move?

Embed all samples and use near neighbours to 