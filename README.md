# Sparse NNLS Tomographic Inversion — Vector Output

`invertTomographySparseNNLSVector` estimates a nonnegative water vapor density field from slant integrated water vapor (SIWV) observations. It solves a sparse nonnegative least-squares (NNLS) problem using a FISTA-based projected-gradient method with column scaling and adaptive restart.

## Problem formulation

Given a geometry matrix $A$ and observations $b$, the function solves

$$
\min_{\rho_v \geq 0} \frac{1}{2}\|A\rho_v-b\|_2^2.
$$

Each row of $A$ represents a ray, and each column represents a voxel. Entries are ray lengths within voxels in meters. With $b$ in kg m$^{-2}$, the recovered density is in kg m$^{-3}$.

Here, **sparse** refers to the geometry matrix and its storage. The objective does not include a sparsity penalty, spatial smoothing, or other regularization. Nonnegativity alone does not guarantee a unique solution for an underdetermined system.

## Main features

- Sparse matrix operations; full input matrices are converted to sparse storage.
- Column normalization to reduce numerical imbalances associated with uneven ray coverage.
- Power iteration to estimate the gradient Lipschitz constant without explicitly forming the normal matrix.
- FISTA momentum, projection onto the nonnegative orthant, and adaptive restart.
- A projected-gradient stopping test based on the Karush–Kuhn–Tucker (KKT) conditions.
- Optional initial density estimate in physical units.
- Geometry-independent vector output in the column order of the input matrix.
- Fit statistics, convergence diagnostics, and a mask of unobserved voxels.

## Installation and dependencies

Save the function as `invertTomographySparseNNLSVector.m` and place it on the MATLAB path or in the current folder.

The solver uses base MATLAB functions and does not call `lsqnonneg`, `lsqlin`, or an Optimization Toolbox solver. 

The caller supplies the geometry matrix and observations. This function does not perform ray tracing, grid generation, observation preprocessing, or spatial assembly of the output. A minimum supported MATLAB release has not been established.

## API

```matlab
[rhov, exitflag, info] = invertTomographySparseNNLSVector(A, b, opts);
```

### Inputs

| Input | Description |
| --- | --- |
| `A` | Real geometry matrix of size `nRays × nVoxels`, preferably sparse and double precision. Physical path lengths are nonnegative and expressed in meters. |
| `b` | Real SIWV observations with `nRays` elements, in kg m^-2. Converted internally to a column vector. |
| `opts` | Optional solver-options structure. Omit it or use `[]` to use the defaults. |

`A` must have at least one row and one column. The function checks dimensions and rejects nonfinite entries in `A` and `b`. Supply real-valued data; the implementation is intended for physical tomography inputs.

The output is a column vector; entry `j` corresponds exactly to column `j` of `A`.

### Options and defaults

Missing or empty numerical options use the defaults below.

| Field | Default | Description |
| --- | --- | --- |
| `maxIterations` | `5000` | Maximum number of iterations; positive integer. |
| `tolerance` | `1e-6` | Positive multiplier for the scaled projected-gradient threshold. |
| `checkEvery` | `10` | Check convergence at this interval, also at the first and final iterations; positive integer. |
| `powerIterations` | `30` | Number of power iterations used to estimate the Lipschitz constant; positive integer. |
| `lipschitzSafety` | `1.05` | Positive factor multiplying the Lipschitz estimate. |
| `display` | `true` | Print solver progress and final diagnostics. Set to `false` for silent operation. |
| `x0` | `[]` | Initial density vector with `nVoxels` finite entries, in kg m^-3 and in the column order of `A`. Empty means zero initialization. Negative values are clipped to zero. |

### Outputs

| Output | Description |
| --- | --- |
| `rhov` | Nonnegative density column vector of size `nVoxels × 1`, in kg m^-3. |
| `exitflag` | `1`: projected-gradient criterion satisfied. `0`: iteration limit reached without satisfying the criterion. Invalid inputs raise errors. |
| `info` | Structure containing the solution vector, convergence diagnostics, and fit statistics. |

`info.rhovVector` is identical to `rhov`. Both follow the column order of `A`.

#### Diagnostic fields

| Field(s) | Meaning |
| --- | --- |
| `algorithm` | Description of the implemented solver. |
| `iterations`, `converged` | Iteration count and logical convergence status. |
| `objective` | Final value of `0.5 * norm(A*rhovVector - b)^2`. |
| `projectedGradientNorm` | Infinity norm of the final projected-gradient mapping in scaled coordinates. |
| `gradientScale`, `convergenceThreshold` | Scale and threshold used by the stopping test. |
| `projectedGradientRatio` | Projected-gradient norm divided by the threshold; convergence requires a value at most `1`. |
| `Lestimate`, `L` | Estimated Lipschitz constant and value used for the step size, respectively. |
| `nRestarts` | Number of momentum restarts. |
| `columnScale` | Column norms, with zero norms replaced by `1`. |
| `relativeChange` | `norm(z-zOld)/max(1,norm(zOld))` at the last convergence check; a diagnostic, not a stopping condition. |
| `bFit` | Modeled observations, `A*rhovVector`, in kg m^-2. |
| `residual` | Observed minus modeled values: `b - bFit`. |
| `residualNorm`, `residualNormSquared` | Euclidean residual norm and its square. |
| `rmse`, `bias` | Root-mean-square residual and mean residual, in kg m^-2. Positive bias means the fitted observations are lower on average. |
| `nnzA`, `matrixDensity` | Nonzero count and nonzero fraction of `A`; `matrixDensity` is a fraction, not a percentage. |
| `nObservations`, `nVoxels` | Number of observations and voxels. |
| `nUnobservedVoxels`, `unobservedVoxels` | Number and logical mask of zero columns in `A`. |
| `rhovVector` | Column vector of recovered physical densities. |

## Example

This synthetic example contains six observations and four unknown densities. It illustrates inversion without defining a spatial grid.

```matlab
% Rows: rays. Columns: voxels. Path lengths in meters.
A = sparse([100  20   0   0;
             0 120  30   0;
             0   0  90  10;
            10   0   0 110;
            50  50  50  50;
            30   0  70  20]);

rhoTrue = [0.010; 0.012; 0.006; 0.008]; % kg m^-3
b = A * rhoTrue;                        % kg m^-2

opts = struct;
opts.maxIterations = 5000;
opts.tolerance = 1e-6;
opts.checkEvery = 10;
opts.display = false;

[rhov, exitflag, info] = invertTomographySparseNNLSVector(A, b, opts);

fprintf('Converged: %d; iterations: %d\n', exitflag == 1, info.iterations);
fprintf('Residual RMSE: %.3e kg m^-2\n', info.rmse);
disp(rhov);

% Density associated with the second column of A:
rhoVoxel2 = rhov(2);
```

For measured data, replace `A` and `b` with the ray-tracing matrix and SIWV observations. To warm-start a later solve on the same voxel ordering, set `opts.x0 = rhov`.

For a call using all defaults, including console output:

```matlab
[rhov, exitflag, info] = invertTomographySparseNNLSVector(A, b);
```

The example is illustrative; automated MATLAB execution was not completed during preparation of this documentation.

## Geometry and output ordering

The function operates on `A` and `b` and does not inspect grid metadata. It can therefore invert systems assembled from rectangular/sigma grids, Voronoi/sigma grids, or other geometries represented by the same linear model.

The caller is responsible for constructing `A` consistently and retaining the mapping between columns and voxels. The solver does not reshape, interpolate, or reorder the solution. Any conversion to a spatial array must be performed separately using the original voxel numbering. 

## Algorithm and convergence

For each column, the solver computes $s_j=\|A_{:,j}\|_2$, replacing zero norms with one. It then defines

$$
A_s=A\,\operatorname{diag}(1/s),\qquad z=\operatorname{diag}(s)\rho_v.
$$

This change of variables preserves the least-squares problem and nonnegativity constraint. The iteration combines projected-gradient steps with FISTA momentum. Momentum is reset when `(y-zNew)'*(zNew-z) > 0`.

Starting from the scaled initial estimate, with $y=z$ and $t=1$, each iteration computes

$$
z_{\mathrm{new}}=\max\left(0,y-\frac{1}{L}A_s^T(A_s y-b)\right),
$$

followed by the momentum update

$$
t_{\mathrm{new}}=\frac{1+\sqrt{1+4t^2}}{2},\qquad
 y_{\mathrm{new}}=z_{\mathrm{new}}+\frac{t-1}{t_{\mathrm{new}}}(z_{\mathrm{new}}-z).
$$

The elementwise projection enforces nonnegativity. If the restart criterion is met, the solver sets $t_{\mathrm{new}}=1$ and $y_{\mathrm{new}}=z_{\mathrm{new}}$, removing the accumulated momentum. After termination, physical densities are recovered as $\rho_{v,j}=z_j/s_j$.

The step size is `1/L`, where `L = lipschitzSafety * max(Lestimate, eps)` and `Lestimate` approximates the largest eigenvalue of $A_s^T A_s$. The implementation uses a fixed estimate and does not perform backtracking. A finite power iteration and safety factor do not guarantee an upper bound for every matrix; convergence behavior should be checked for new datasets.

At a convergence check, the function computes

$$
g=A_s^T(A_s z-b),\qquad p=z-\max(0,z-g),
$$

where the maximum is element-wise. Iteration stops successfully when

$$
\|p\|_\infty\leq \texttt{tolerance}\,\max(1,\|A_s^T b\|_\infty).
$$

This is a scaled projected-gradient test in the normalized variables, with an absolute floor in the scale. It is not a relative residual test. By default, it is checked at iteration 1, every 10 iterations, and at the final permitted iteration, with a maximum of 5000 iterations.

In all experiments reported by the author to date, convergence was reached in fewer than 2000 iterations using the stated settings. This is an empirical observation for those experiments, not a universal iteration bound.

Voxels not crossed by any ray are fixed at zero, including when an initial guess is supplied. These zeros represent unconstrained voxels, not evidence of zero atmospheric water vapor; use `info.unobservedVoxels` when interpreting or plotting the result.

## References

- **Beck, A., & Teboulle, M. (2009).** *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*. **SIAM Journal on Imaging Sciences, 2**(1), 183–202. DOI: **[10.1137/080716542](https://doi.org/10.1137/080716542)**. Foundation for FISTA acceleration; here the proximal operation is projection onto the nonnegative orthant.
- O'Donoghue, B., & Candès, E. *Adaptive Restart for Accelerated Gradient Schemes*. [Author preprint, arXiv:1204.3982](https://arxiv.org/abs/1204.3982). Background on adaptive momentum restart.

## Citation and license

If this software contributes to a publication, please cite the associated research article, when available, and the software version or commit used. Cite the algorithm references above when describing the numerical method. Repository-specific citation metadata can be added in a `CITATION.cff` file once authorship and publication details are finalized.

License terms should be provided in a separate `LICENSE` file chosen by the copyright holder. This README does not assign a license or grant usage rights.
