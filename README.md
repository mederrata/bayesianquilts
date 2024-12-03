# Bayesianquilts
This library provides tools for building truly interpretable input -> output maps based on the principle of piecewise linearity. Rather than doing so
indirectly by building a ReLU-activated neural network, this library uses a 
combination of representation learning, clustering, and multilevel linear regression modeling.

Some examples of this class of models are in the literature:

* Chang TL, Xia H, Mahajan S, Mahajan R, Maisog J, et al. (2024) Interpretable (not just posthoc-explainable) medical claims modeling for discharge placement to reduce preventable all-cause readmissions or death. PLOS ONE 19(5): e0302871. https://doi.org/10.1371/journal.pone.0302871

* Hongjing Xia, Joshua C. Chang, Sarah Nowak, Sonya Mahajan, Rohit Mahajan, Ted L. Chang, Carson C. Chow Proceedings of the 8th Machine Learning for Healthcare Conference, PMLR 219:884-905, 2023.

The core of the method is in developng an additive decomposition of each paramter where the effecitve local value for a parameter arises as a sum over contributions at different length skills.

See here for usage example for the parameter decomposition method: https://github.com/mederrata/bayesianquilts/blob/main/notebooks/decomposition.ipynb

Additionally, this repository contains code for Gradient-flow adaptive importance sampling as described by this manuscript:

* Chang JC, Li X, Xu S, Yao HR, Porcino J, Chow CC. Gradient-flow adaptive importance sampling for Bayesian leave one out cross-validation with application to sigmoidal classification models. ArXiv [Preprint]. 2024 Oct 20:arXiv:2402.08151v2. PMID: 38711425; PMCID: PMC11071546.



 