# Boer Mulders Functions

## Abstract

At low energies, the structure of the proton is relatively simple. However, at high energies, the structure of the proton becomes much more dynamic and fluid. High-energy collisions provide an opportunity to study the internal structure of the proton by creating conditions that allow the observation of its constituent particles. Understanding the total spin of protons plays a crucial role in comprehending the proton's structure.

The $\cos2\phi$ asymmetry in the Drell-Yan process, where $\phi$ represents the azimuthal angle of the $\mu^{+}\mu^{-}$ pair in the Collins-Soper frame, can be described by the Boer-Mulders (BM) function. This function characterizes the transverse-polarization asymmetry of quarks within an unpolarized hadron and arises from the coupling between the quark's transverse momentum and transverse spin inside the hadron.

SeaQuest is a fixed-target Drell-Yan experiment conducted at Fermilab, involving an unpolarized proton beam colliding with unpolarized LH2 and LD2 targets. The $\cos2\phi$ asymmetry is determined by detecting $\mu^{+}\mu^{-}$ pairs. Accurately extracting the $\cos2\phi$ asymmetry is crucial for determining the BM function.

Measurements obtained from experiments typically require correction for detector inefficiencies, smearing, and acceptance. Traditionally, these corrections involve "unfolding" the detector-level measurements through matrix operations. However, in higher dimensions in parameter space, these conventional methods fail to scale effectively.

To overcome these limitations, we propose a novel approach that utilizes Deep Neural Networks for directly extracting the angular coefficients using high-dimensional information from the detector level. Neural networks excel in approximating nonlinear functions, making them suitable for representing the full phase space for parameter optimization.

In this repository, our objective is to investigate machine learning algorithms to accurately extract the $\cos2\phi$ asymmetry.

![Complicated structure of the proton](https://bigthink.com/wp-content/uploads/2022/12/960x0.jpg?resize=768,644)

![Drell-Yan process](https://spinquest.fnal.gov/wp-content/uploads/2019/05/DY_web-768x292.jpg)


## Method

- An example of neural positive reweighing can be found in this [notebook](put_url).
- An example of a Gaussian example can be found in this [notebook](https://github.com/dinupa1/BMF/blob/GaussianExample/GaussianExample.ipynb).
- An example of simple neural reweighing can be found in this [notebook](put_url).

TODO ....

## Talks and Presentations

- New Perpectives Meeting, June 2023: [slides](https://github.com/dinupa1/BMF/blob/New-Perspectives-June-2023/slides/NP-June-27-2023.pdf)

TODO ....

## References

- Measurement of Angular Distributions of Drell-Yan Dimuons in p + d Interaction at 800 GeV/c, [arXiv:hep-ex/0609005](https://arxiv.org/abs/hep-ex/0609005)
- The Asymmetry of Antimatter in the Proton, [arXiv:2103.04024](https://arxiv.org/abs/2103.04024)
- Time-reversal odd distribution functions in leptoproduction, [arXiv:hep-ph/9711485](https://arxiv.org/abs/hep-ph/9711485)
- Extracting Boer-Mulders functions from $p+D$ Drell-Yan processes, [arXiv:0803.1692](https://arxiv.org/abs/0803.1692)
- A Neural Resampler for Monte Carlo Reweighting with Preserved Uncertainties [arXiv:2007.11586](https://arxiv.org/abs/2007.11586)
- Neural Networks for Full Phase-space Reweighting and Parameter Tuning, [arXiv:1907.08209](https://arxiv.org/abs/1907.08209)
- Parameter Estimation using Neural Networks in the Presence of Detector Effects, [arXiv:2010.03569](https://arxiv.org/abs/2010.03569)
- OmniFold: A Method to Simultaneously Unfold All Observables, [arXiv:1911.09107](https://arxiv.org/abs/1911.09107)

TODO ....