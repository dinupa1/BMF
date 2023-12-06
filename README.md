# Extraction of Drell-Yan Angular Coefficients using U-Nets

## Abstract

Understanding the total spin of protons plays a major role in comprehending the proton's structure. The $\cos2\phi$ asymmetry in the Drell-Yan (DY) process, where $\phi$ denotes the azimuthal angle of the $\mu^{+}\mu^{-}$ pair in the Collins-Soper frame, can be described by the Boer-Mulders (BM) function. This function characterizes the transverse-polarization asymmetry of quarks within an unpolarized hadron and arises from the coupling between the quark's transverse momentum and transverse spin inside the hadron. SeaQuest is a fixed-target Drell-Yan experiment conducted at Fermilab, which involves an unpolarized proton beam colliding with an unpolarized LH2 and LD2 targets. The $\cos2\phi$ asymmetry is determined by detecting $\mu^{+}\mu^{-}$ pairs. Accurately extracting the $\cos2\phi$ asymmetry is important for determining the BM function. Measurements obtained from experiments typically require correction for detector inefficiencies, smearing, and acceptance. Traditionally, these corrections involve `unfolding` the detector-level measurements through matrix operations. However, in higher dimensions in parameter space, these conventional methods fail to scale effectively. To overcome these limitations, we propose a novel approach that utilizes Deep Neural Networks for directly extracting the angular coefficients using high-dimensional information from the detector level. Neural networks excel in approximating nonlinear functions, making them suitable for representing the full phase space for parameter optimization. In this repository, we will explain the design of the neural network architecture, training strategies, and outline our plans to achieve conclusive results.

## U-Nets

We use U-Nets as the model architecture for the unfolding procedure. U-Nets made a major breakthrough in `Image Segmentation` in 2015. This is a `Fully Convolutional Network`. Since U-Nets excel at segmentation, we use them to extract DY angular coefficients as a function of $mass$, $x_{F}$, and $p_{T}$ at the particle level using detector-level information as a segmentation task.

## Data structure

Consider the DY angular cross-section:

\[ \frac{d\sigma}{d\Omega} \propto 1  + \lambda \cos^{2}\theta + \mu \sin 2 \theta \cos \phi + \frac{1}{2}\nu \sin^{2}\theta \cos 2 \phi \]

The input to the model consists of three channels: $\phi$ vs $\cos\theta$, $\cos\phi$ vs. $\cos\theta$, and $\cos2\phi$ vs. $\cos\theta$. The target (segmented data) is a 3D histogram of $mass$, $p_{T}$, and $x_{F}$ which contains DY angular coefficients and their errors. DY angular coefficients have been injected into the input histograms as Gaussian distributions with the mean as the DY angular coefficient and the width as its error.

Follow the below steps to extract DY coefficients.

- Create `E906 messy data`: [here](https://github.com/abinashpun/seaquest-projects)
- Create data sets using: [directory](https://github.com/dinupa1/bm-function/tree/dev-11-26-2023/data-sets)
- Train U-Net models and make predictions: [directory](https://github.com/dinupa1/bm-function/tree/dev-11-26-2023/models)
- Plot the predictions: [directory](https://github.com/dinupa1/bm-function/tree/dev-11-26-2023/plots)

## References

- Measurement of Angular Distributions of Drell-Yan Dimuons in p + d Interaction at 800 GeV/c, [arXiv:hep-ex/0609005](https://arxiv.org/abs/hep-ex/0609005)
- The Asymmetry of Antimatter in the Proton, [arXiv:2103.04024](https://arxiv.org/abs/2103.04024)
- Time-reversal odd distribution functions in leptoproduction, [arXiv:hep-ph/9711485](https://arxiv.org/abs/hep-ph/9711485)
- Extracting Boer-Mulders functions from $p+D$ Drell-Yan processes, [arXiv:0803.1692](https://arxiv.org/abs/0803.1692)
- A Neural Resampler for Monte Carlo Reweighting with Preserved Uncertainties [arXiv:2007.11586](https://arxiv.org/abs/2007.11586)
- Neural Networks for Full Phase-space Reweighting and Parameter Tuning, [arXiv:1907.08209](https://arxiv.org/abs/1907.08209)
- Parameter Estimation using Neural Networks in the Presence of Detector Effects, [arXiv:2010.03569](https://arxiv.org/abs/2010.03569)
- OmniFold: A Method to Simultaneously Unfold All Observables, [arXiv:1911.09107](https://arxiv.org/abs/1911.09107)
- How to GAN Event Unweighting, [arXiv:2012.07873](https://arxiv.org/abs/2012.07873)
- Autoencoders [arXiv:2003.05991](https://arxiv.org/abs/2003.05991)
- U-Net: Convolutional Networks for Biomedical Image Segmentation [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Estimation of Combinatoric Background in SeaQuest using an Event-Mixing Method [arXiv:2302.04152](https://arxiv.org/abs/2302.04152)