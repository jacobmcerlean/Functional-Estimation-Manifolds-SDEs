Functional Estimation of Manifold Valued Diffusion Processes
Jacob McErlean and Hau Tieng Wu
https://arxiv.org/pdf/2603.20945

This repository accompanies the above paper and provides a complete pipeline for

- simulating diffusion processes on manifolds
- estimating drift and diffusion from observed data
- reproducing the numerical results in the manuscript



------------------------------------------------------------
Repository Structure
------------------------------------------------------------


SRC

This folder contains the core implementation

SDE_sample_KB.py
Simulation of stochastic differential equations on the Klein bottle

SDE_sample_Sphere.py
Simulation of stochastic differential equations on the sphere

kernel_estimators.py
Kernel based estimators for drift and diffusion

observed_ellipsoid.py
Helper functions for ellipsoid embeddings and geometry


------------------------------------------------------------


EXPERIMENTS

This folder contains scripts for generating data and running experiments

KB_gen_traj.py
Generate trajectories on the Klein bottle and perform down sampling

ellipsoid_gen_traj.py
Generate trajectories on ellipsoids via pushforward from the sphere

KB_obs_est.py
Main global estimation experiment on the Klein bottle

ellipsoid_obs_est.py
Main global estimation experiment on ellipsoids

KB_invariant_density.py
Visualization of invariant density on the Klein bottle

ellipsoid_invariant_density.py
Visualization of invariant density on ellipsoids

ellipsoid_normality_simulations.py
Experiment evaluating normality of the estimators


------------------------------------------------------------


FIGURES AND TABLES

This folder contains notebooks for reproducing all figures and tables

KB_Figures_and_Table.ipynb
Ellipsoid_Figures_and_Tables.ipynb

Saved outputs required for the figures are also included in this folder


------------------------------------------------------------


SUMMARY

This codebase supports

simulation of manifold valued diffusion processes
estimation of drift and diffusion from observed trajectories
evaluation of estimators through controlled experiments
reproduction of all figures and tables from the manuscript
