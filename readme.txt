FUNCTIONAL ESTIMATION OF MANIFOLD-VALUED DIFFUSION PROCESSES
Jacob McErlean and Hau-Tieng Wu
https://arxiv.org/pdf/2603.20945

This repository accompanies the above paper and contains code for simulating and estimating diffusion processes on manifolds.

The repository is organized into three main folders.

The folder “src” contains the core implementation. The files SDE_sample_KB.py and SDE_sample_Sphere.py simulate SDEs on the Klein bottle and sphere, respectively. The file kernel_estimators.py implements the kernel functions and estimation procedures, and observed_ellipsoid.py provides helper routines for working with ellipsoid embeddings.

The folder “experiments” contains scripts for generating data and running estimation experiments. The files KB_gen_traj.py and ellipsoid_gen_traj.py generate SDE trajectories on the Klein bottle and ellipsoids and perform down-sampling. The files KB_obs_est.py and ellipsoid_obs_est.py implement the main global estimation experiments on the Klein bottle and ellipsoids. The scripts observed_KB_experiment_3.py and observed_ellipsoid_experiment_3.py visualize invariant densities, while observed_ellipsoid_experiment_1.py provides an additional experiment evaluating normality of the estimators.

The folder “figures and tables” contains the notebooks KB_Figures_and_Table.ipynb and Ellipsoid_Figures_and_Tables.ipynb, which generate all figures and tables in the manuscript. This folder also includes the necessary saved outputs used by the notebooks.

This codebase provides a complete pipeline for simulating manifold-valued diffusion processes, estimating drift and diffusion functions from observed data, and reproducing the numerical results presented in the paper.
