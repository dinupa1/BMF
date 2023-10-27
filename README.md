# README

## Abstract

At low energies, the structure of the proton is relatively simple. However, at high energies, the structure of the proton becomes much more dynamic and fluid. High-energy collisions provide an opportunity to study the internal structure of the proton by creating conditions that allow the observation of its constituent particles. Understanding the total spin of protons plays a crucial role in comprehending the proton's structure.

The $\cos2\phi$ asymmetry in the Drell-Yan process, where $\phi$ represents the azimuthal angle of the $\mu^{+}\mu^{-}$ pair in the Collins-Soper frame, can be described by the Boer-Mulders (BM) function. This function characterizes the transverse-polarization asymmetry of quarks within an unpolarized hadron and arises from the coupling between the quark's transverse momentum and transverse spin inside the hadron.

SeaQuest is a fixed-target Drell-Yan experiment conducted at Fermilab, involving an unpolarized proton beam colliding with unpolarized LH2 and LD2 targets. The $\cos2\phi$ asymmetry is determined by detecting $\mu^{+}\mu^{-}$ pairs. Accurately extracting the $\cos2\phi$ asymmetry is crucial for determining the BM function.

Measurements obtained from experiments typically require correction for detector inefficiencies, smearing, and acceptance. Traditionally, these corrections involve "unfolding" the detector-level measurements through matrix operations. However, in higher dimensions in parameter space, these conventional methods fail to scale effectively.

To overcome these limitations, we propose a novel approach that utilizes Deep Neural Networks for directly extracting the angular coefficients using high-dimensional information from the detector level. Neural networks excel in approximating nonlinear functions, making them suitable for representing the full phase space for parameter optimization.

In this repository, our objective is to investigate machine learning algorithms to accurately extract the $\cos2\phi$ asymmetry.

## U-Net based acceptance correction

