Here are the arXiv / preprint links I found:

* *Training Data Influence Analysis and Estimation: A Survey* — arXiv:2212.04612 ([arXiv][1])
* *Prioritized Experience Replay* (the original PER) — arXiv:1511.05952 ([arXiv][2])
* *Investigating the Interplay of Prioritized Replay and Generalization* — arXiv:2407.09702 ([arXiv][3])
* *Uncertainty Prioritized Experience Replay (UPER)* — arXiv:2506.09270 ([arXiv][4])
* *Distributed Prioritized Experience Replay* — arXiv:1803.00933 ([arXiv][5])

If you like, I can also fetch BibTeX entries for all of them.

[1]: https://arxiv.org/abs/2212.04612?utm_source=chatgpt.com "Training Data Influence Analysis and Estimation: A Survey"
[2]: https://arxiv.org/abs/1511.05952?utm_source=chatgpt.com "[1511.05952] Prioritized Experience Replay - arXiv"
[3]: https://arxiv.org/abs/2407.09702?utm_source=chatgpt.com "Investigating the Interplay of Prioritized Replay and Generalization"
[4]: https://arxiv.org/abs/2506.09270?utm_source=chatgpt.com "[2506.09270] Uncertainty Prioritized Experience Replay - arXiv"
[5]: https://arxiv.org/abs/1803.00933?utm_source=chatgpt.com "Distributed Prioritized Experience Replay"

Zero noise

| Title / Authors                                                                       | Core idea / relevance                                                                                                               | Notes / caveats                                                      |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| *Training data influence analysis and estimation: a survey* (Hammoudeh & Lowd, 2024)  | Broad survey of influence / data-attribution / importance methods for training data. ([SpringerLink][1])                            | Good to map the landscape and tradeoffs                              |
| *Optimal Subsampling with Influence Functions* (Ting et al.)                          | Using influence functions to derive optimal sampling weights for regression / estimator tasks. ([papers.neurips.cc][2])             | More theoretical; more suited for small/convex models                |
| *Influence Functions in Deep Learning Are Fragile* (Basu, Pope, Feizi, 2020)          | Empirical study showing limitations of influence estimators in nonconvex models (deep nets). ([arXiv][3])                           | Warns you: influence is approximate and sometimes misleading         |
| *Revisiting the fragility of influence functions* (Epifano et al., 2023)              | Extends critique and explores when influence methods fail. ([NSF Public Access Repository][4])                                      | Use as caution filter                                                |
| *Distributed Prioritized Experience Replay* (Horgan et al., 2018)                     | A main reference for prioritized replay in RL. ([arXiv][5])                                                                         | The classic architecture for prioritized replay                      |
| *Investigating the Interplay of Prioritized Replay and Neural Network Generalization* | Empirical work exploring how prioritization affects generalization. ([arXiv][6])                                                    | Helps you see pitfalls in combining prioritized sampling + deep nets |
| *Uncertainty Prioritized Experience Replay (UPER)*                                    | Proposes using epistemic uncertainty / info gain rather than raw TD error for replay priority. ([rlj.cs.umass.edu][7])              | Closer to “information score” notion (Bayesian)                      |
| *Leveraging Influence Functions for Dataset Exploration and Cleaning*                 | Practical use of influence functions to detect mislabeled or complex points in datasets. ([UTwente Research Information System][8]) | Good for applying influence in practice                              |

[1]: https://link.springer.com/article/10.1007/s10994-023-06495-7?utm_source=chatgpt.com "Training data influence analysis and estimation: a survey"
[2]: https://papers.neurips.cc/paper/7623-optimal-subsampling-with-influence-functions.pdf?utm_source=chatgpt.com "Optimal Subsampling with Influence Functions"
[3]: https://arxiv.org/abs/2006.14651?utm_source=chatgpt.com "Influence Functions in Deep Learning Are Fragile"
[4]: https://par.nsf.gov/servlets/purl/10409977?utm_source=chatgpt.com "Revisiting the fragility of influence functions"
[5]: https://arxiv.org/abs/1803.00933?utm_source=chatgpt.com "Distributed Prioritized Experience Replay"
[6]: https://arxiv.org/html/2407.09702v2?utm_source=chatgpt.com "Investigating the Interplay of Prioritized Replay and ..."
[7]: https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_45.pdf?utm_source=chatgpt.com "Uncertainty Prioritized Experience Replay"
[8]: https://ris.utwente.nl/ws/portalfiles/portal/316906318/Leveraging_Influence_Functions_for_Dataset_Exploration_and_Cleaning_finalV2.pdf?utm_source=chatgpt.com "Leveraging Influence Functions for Dataset Exploration and ..."


> In deep nets, the diagonal of the Fisher is often used as a cheap curvature proxy.


The environment is zero noise, but chess is not smooth and evaluations can change sharply with small changes in position.