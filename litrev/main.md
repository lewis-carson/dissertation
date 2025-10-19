https://leela-interp.github.io

Here are **15 arXiv / open-access papers** (with PDF / HTML links) that I believe are quite relevant and illuminating for a dissertation on chess programming (especially with modern ML/neural / hybrid methods). Use these as “essential reading” for your literature survey and method groundwork:

---

1. *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* — Silver et al. (AlphaZero)
   PDF / arXiv: [https://arxiv.org/abs/1712.01815](https://arxiv.org/abs/1712.01815) ([arXiv][1])

2. *Neural Networks for Chess* — Dominik Klein
   PDF / arXiv: [https://arXiv.org/abs/2209.01506](https://arXiv.org/abs/2209.01506) ([arXiv][2])

3. *Monte-Carlo Graph Search for AlphaZero* — Czech, Korus, Kersting
   PDF / arXiv: [https://arXiv.org/abs/2012.11045](https://arXiv.org/abs/2012.11045) ([arXiv][3])

4. *Augmented Reality Chess Analyzer (ARChessAnalyzer)*
   PDF / arXiv: [https://arXiv.org/pdf/2009.01649](https://arXiv.org/pdf/2009.01649) ([arXiv][4])

5. *Rotated Bitboards and Reinforcement Learning in Chess*
   HTML / arXiv: [https://arXiv.org/html/2503.10822v2](https://arXiv.org/html/2503.10822v2) ([arXiv][5])

6. *Superior Computer Chess with Model Predictive Control, Reinforcement Learning, and Rollout*
   PDF / arXiv: [https://web.mit.edu/dimitrib/www/MPC-MC_ArXiv2409.06477v1.pdf](https://web.mit.edu/dimitrib/www/MPC-MC_ArXiv2409.06477v1.pdf) ([Massachusetts Institute of Technology][6])

7. *Understanding the learned look-ahead behavior of chess neural networks*
   HTML / arXiv: [https://arXiv.org/html/2505.21552v1](https://arXiv.org/html/2505.21552v1) ([arXiv][7])

8. *Evidence of Learned Look-Ahead in a Chess-Playing Neural Network*
   PDF / arXiv: [https://arXiv.org/abs/2406.00877](https://arXiv.org/abs/2406.00877) ([arXiv][8])

9. *Mastering Chess with a Transformer Model*
   HTML / arXiv: [https://arXiv.org/html/2409.12272v2](https://arXiv.org/html/2409.12272v2) ([arXiv][9])

10. *Amortized Planning with Large-Scale Transformers*
    PDF / arXiv: [https://arXiv.org/abs/2402.04494](https://arXiv.org/abs/2402.04494) ([arXiv][10])

11. *Chess AI: Competing Paradigms for Machine Intelligence*
    PDF / arXiv: [https://arXiv.org/pdf/2109.11602](https://arXiv.org/pdf/2109.11602) ([arXiv][11])

12. *ChessGPT: Bridging Policy Learning and Language Modeling*
    PDF / arXiv: [https://arXiv.org/abs/2306.09200](https://arXiv.org/abs/2306.09200) ([arXiv][12])

13. *Explore the Reasoning Capability of LLMs in the Chess Testbed*
    HTML / arXiv: [https://arXiv.org/html/2411.06655v1](https://arXiv.org/html/2411.06655v1) ([arXiv][13])

14. *Complete Chess Games Enable LLM Become A Chess Master*
    HTML / arXiv: [https://arXiv.org/html/2501.17186v2](https://arXiv.org/html/2501.17186v2) ([arXiv][14])

15. *Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM*
    HTML / arXiv: [https://arXiv.org/html/2409.11506v2](https://arXiv.org/html/2409.11506v2) ([arXiv][15])

---

### Comments on how these fit into your survey and project

* #1 (AlphaZero) is a foundational paper. It changed how the field thinks about combining search + self-play reinforcement learning.
* #3 (Monte-Carlo Graph Search) is more recent refinements of the search side in that paradigm.
* #2 (Klein's “Neural Networks for Chess”) is a good tutorial / foundational review of combining NN + search.
* #5 (Bitboards + efficient representations) is about low-level optimization, essential if you build your own engine.
* #6 (MPC + rollout) and #11 (paradigms survey) help you see hybrid or newer architectures.
* #7, #8 are studies in interpretability: how neural chess systems internally represent lookahead.
* #9, #10, #12, #13, #14 are exploring transformer / language model / planning approaches for chess. They capture the frontier of experimenting beyond classical architectures.
* #15 is more peripheral (rating estimation) but interesting if you care about auxiliary tasks or evaluation.

---

If you like, I can send you a **curated subset (5–7)** of *must-read* ones (ranked) tailored to your specific proposed project focus, so you don’t get overwhelmed. Do you want me to pick those for *your* project direction?

[1]: https://arxiv.org/abs/1712.01815?utm_source=chatgpt.com "Mastering Chess and Shogi by Self-Play with a General ..."
[2]: https://arxiv.org/abs/2209.01506?utm_source=chatgpt.com "Neural Networks for Chess"
[3]: https://arxiv.org/abs/2012.11045?utm_source=chatgpt.com "Monte-Carlo Graph Search for AlphaZero"
[4]: https://arxiv.org/abs/2009.01649?utm_source=chatgpt.com "[2009.01649] Augmented Reality Chess Analyzer (ARChessAnalyzer)"
[5]: https://arxiv.org/html/2503.10822v2?utm_source=chatgpt.com "Rotated Bitboards and Reinforcement Learning in ..."
[6]: https://web.mit.edu/dimitrib/www/MPC-MC_ArXiv2409.06477v1.pdf?utm_source=chatgpt.com "arXiv:2409.06477v1 [cs.AI] 10 Sep 2024"
[7]: https://arxiv.org/html/2505.21552v1?utm_source=chatgpt.com "Understanding the learned look-ahead behavior of chess neural ..."
[8]: https://arxiv.org/abs/2406.00877?utm_source=chatgpt.com "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network"
[9]: https://arxiv.org/html/2409.12272v2?utm_source=chatgpt.com "Mastering Chess with a Transformer Model - arXiv"
[10]: https://arxiv.org/abs/2402.04494?utm_source=chatgpt.com "Amortized Planning with Large-Scale Transformers"
[11]: https://arxiv.org/pdf/2109.11602?utm_source=chatgpt.com "[PDF] Chess AI: Competing Paradigms for Machine Intelligence - arXiv"
[12]: https://arxiv.org/abs/2306.09200?utm_source=chatgpt.com "ChessGPT: Bridging Policy Learning and Language Modeling - arXiv"
[13]: https://arxiv.org/html/2411.06655v1?utm_source=chatgpt.com "Explore the Reasoning Capability of LLMs in the Chess Testbed"
[14]: https://arxiv.org/html/2501.17186v2?utm_source=chatgpt.com "Complete Chess Games Enable LLM Become A Chess Master - arXiv"
[15]: https://arxiv.org/html/2409.11506v2?utm_source=chatgpt.com "Chess Rating Estimation from Moves and Clock Times Using a CNN ..."

Here are 10+ arXiv (or freely accessible preprint) papers that are especially relevant to **neural-network chess engines** and **interpretability / mechanistic understanding**. These are good candidates to read, compare, and cite in your dissertation. I order them in rough priority, and group by theme.

---

## Core papers on neural chess / chess with transformers / learned look-ahead

1. **Evidence of Learned Look-Ahead in a Chess-Playing Neural Network**
   Jenner, Kapur, Georgiev, Allen, Emmons, Russell (2024) — shows causal / probing evidence that Leela’s policy network internally represents future moves and uses them. ([arXiv][1])
   [PDF/HTML] [https://arxiv.org/abs/2406.00877](https://arxiv.org/abs/2406.00877)

2. **Iterative Inference in a Chess-Playing Neural Network**
   Sandmann et al. (2025) — analyzes how the policy network refines its outputs through layers, non-monotonic trajectories of logits, etc. ([arXiv][2])
   [PDF/HTML] [https://arXiv.org/abs/2508.21380](https://arXiv.org/abs/2508.21380)

3. **Mastering Chess with a Transformer Model**
   (Monroe & Chalmers, 2024) — introduces “Chessformer,” a transformer-based architecture for chess, comparing architectures, position embeddings, and examining attention maps. ([arXiv][3])
   [PDF/HTML] [https://arXiv.org/abs/2409.12272](https://arXiv.org/abs/2409.12272)

4. **Grandmaster-Level Chess Without Search**
   (2024) — trains a large transformer model purely by supervised learning (no explicit search) and achieves very high Elo and puzzle performance. Useful as a contrast to search-based or hybrid models. ([arXiv][4])
   [PDF/HTML] [https://arXiv.org/abs/2402.04494](https://arXiv.org/abs/2402.04494)

5. **Contrastive Sparse Autoencoders for Interpreting Planning in Neural Chess Agents**
   (2024) — explores interpretability of latent spaces tied to planning, building automated feature taxonomies, etc. ([arXiv][5])
   [PDF/HTML] [https://arXiv.org/abs/2406.04028](https://arXiv.org/abs/2406.04028)

6. **Neural Networks for Chess**
   (Klein, 2022) — not purely interpretability, but a well-rounded survey / introduction to neural nets in chess (AlphaZero, Leela, NNUE, etc). Good as a foundational reference. ([arXiv][6])
   [PDF/HTML] [https://arXiv.org/abs/2209.01506](https://arXiv.org/abs/2209.01506)

7. **DeepChess: End-to-End Deep Neural Network for Automatic Learning in Chess**
   (David, Netanyahu, Wolf, 2017) — an earlier attempt at end-to-end evaluation (comparing positions) without domain-specific features. Helps you see the evolution of ideas. ([arXiv][7])
   [PDF/HTML] [https://arXiv.org/abs/1711.09667](https://arXiv.org/abs/1711.09667)

---

## Supplemental / adjacent topics

8. **Superior Computer Chess with Model Predictive Control, Reinforcement Learning, and Rollout**
   Gundawar, Li, Bertsekas (2024) — integrates MPC / rollout / RL ideas, viewing engines as components in a larger architecture. ([Massachusetts Institute of Technology][8])
   [PDF] [https://web.mit.edu/dimitrib/www/MPC-MC_ArXiv2409.06477v1.pdf](https://web.mit.edu/dimitrib/www/MPC-MC_ArXiv2409.06477v1.pdf)

9. **Learning to play the Chess Variant Crazyhouse …**
   (Czech, Willig, et al., 2019) — trains a neural network for the variant “crazyhouse” using supervised and MCTS techniques, interesting as a different domain with more branching factor. ([Astrophysics Data System][9])
   [PDF/HTML] [https://arXiv.org/abs/1908.06660](https://arXiv.org/abs/1908.06660)

10. **Predicting User Perception of Move Brilliance in Chess**
    (Zaidi et al., 2024) — uses neural nets + engine outputs to classify “brilliant” moves, combining predictive modeling with features of move trees. While not about engine internals, it’s a creative use of interpretability / explainability. ([arXiv][10])
    [PDF/HTML] [https://arXiv.org/abs/2406.11895](https://arXiv.org/abs/2406.11895)

11. **Maia-2: A Unified Model for Human-AI Alignment in Chess**
    (Tang, Jiao, McIlroy-Young, Kleinberg, Sen, Anderson, 2024) — focuses more on human move prediction and alignment across skill levels, but includes network architectures and interpretability of skills embedding. ([arXiv][11])
    [PDF/HTML] [https://arXiv.org/abs/2409.20553](https://arXiv.org/abs/2409.20553)

---

If you like, I can also gather the **PDFs directly** (downloadable links) and assemble a BibTeX file for you. Do you want me to fetch that?

[1]: https://arxiv.org/abs/2406.00877?utm_source=chatgpt.com "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network"
[2]: https://arxiv.org/abs/2508.21380?utm_source=chatgpt.com "Iterative Inference in a Chess-Playing Neural Network"
[3]: https://arxiv.org/html/2409.12272v2?utm_source=chatgpt.com "Mastering Chess with a Transformer Model"
[4]: https://arxiv.org/html/2402.04494v1?utm_source=chatgpt.com "Grandmaster-Level Chess Without Search"
[5]: https://arxiv.org/html/2406.04028v1?utm_source=chatgpt.com "Contrastive Sparse Autoencoders for Interpreting Planning ..."
[6]: https://arxiv.org/abs/2209.01506?utm_source=chatgpt.com "Neural Networks for Chess"
[7]: https://arxiv.org/abs/1711.09667?utm_source=chatgpt.com "DeepChess: End-to-End Deep Neural Network for Automatic Learning in Chess"
[8]: https://web.mit.edu/dimitrib/www/MPC-MC_ArXiv2409.06477v1.pdf?utm_source=chatgpt.com "arXiv:2409.06477v1 [cs.AI] 10 Sep 2024"
[9]: https://ui.adsabs.harvard.edu/abs/arXiv%3A1908.06660?utm_source=chatgpt.com "Learning to play the Chess Variant Crazyhouse above ..."
[10]: https://arxiv.org/abs/2406.11895?utm_source=chatgpt.com "Predicting User Perception of Move Brilliance in Chess"
[11]: https://arxiv.org/html/2409.20553v1?utm_source=chatgpt.com "Maia-2: A Unified Model for Human-AI Alignment in Chess"

https://www.lesswrong.com/posts/FF8i6SLfKb4g7C4EL/inside-the-mind-of-a-superhuman-go-model-how-does-leela-zero-2

 path patching allows us to precisely isolate the effect of every causal pathway in the model, and causal scrubbing

 Mean ablating also takes the model out of distribution because the mean is not necessarily on the manifold of plausible activations

 In hindsight, storing global information about the game state along the edges of the board makes sense, as the edges are less frequently used than other parts of the board in most games of Go. But this is by no means an obvious choice a priori. In fact, the four corners of the board are used even less frequently. Why did the model not learn to use the corners instead, which would be more efficient and interfere with the board state even less? The model appears to use the other edges of the board as well, as it is fairly common to observe activations that highlight the other edges of the board. We think some of these are related to the value prediction (see below), but we don’t have concrete hypotheses for what they represent. There is of course the questions of how: How does the model compute the goodness of passing, and how does that information end up at the top edge? We leave these as future research directions.

The most information encoded in certain positions (like as in surprise?). Or information theory - anything which shifts the gradients about in surpising ways. Think of synthetic data based on this - positions which are close to this. Think about the manifold of plausible activations - generate synthetic data which is in the most complex, highly non-linear parts of the manifold. Selectively replay these? Might make training more efficient.