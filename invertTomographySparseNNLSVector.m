function [rhov,exitflag,info] = invertTomographySparseNNLSVector(A,b,opts)
%INVERTTOMOGRAPHYSPARSENNLSVECTOR Solve sparse NNLS and return a vector.
%
%   [RHOV,EXITFLAG,INFO] = INVERTTOMOGRAPHYSPARSENNLSVECTOR(A,B,OPTS)
%   solves min 0.5*norm(A*RHOV-B)^2 subject to RHOV >= 0 using
%   FISTA projected gradient, column scaling, and adaptive restart.
%
% INPUTS
%   A     Real geometry matrix [nRays x nVoxels], with path lengths [m].
%         Full matrices are converted to sparse storage.
%   B     Real SIWV observations [kg m^-2], with nRays elements.
%   OPTS  Optional structure. Missing/empty numerical options use defaults:
%         maxIterations   5000  Maximum iteration count.
%         tolerance       1e-6  Scaled projected-gradient tolerance.
%         checkEvery        10  Check interval; also first/final iteration.
%         powerIterations   30  Lipschitz estimation iterations.
%         lipschitzSafety  1.05  Factor applied to the Lipschitz estimate.
%         display         true  Print progress and final diagnostics.
%         x0                []  Initial density vector in physical units;
%                               empty means zeros. Negatives are clipped.
%
% OUTPUTS
%   RHOV      [nVoxels x 1] density vector [kg m^-3]. Entry j corresponds
%             exactly to column j of A. No grid definition is required.
%   EXITFLAG  1: convergence criterion met; 0: iteration limit reached.
%   INFO      Solver diagnostics and fit statistics:
%             algorithm, iterations, converged, objective,
%             projectedGradientNorm, gradientScale, convergenceThreshold,
%             projectedGradientRatio, Lestimate, L, nRestarts, columnScale,
%             bFit, residual, residualNorm, residualNormSquared, rmse, bias,
%             nnzA, matrixDensity, nObservations, nVoxels,
%             nUnobservedVoxels, unobservedVoxels, relativeChange, rhovVector.
%             INFO.rhovVector equals RHOV. No gridType field is returned.
%
% CONVERGENCE
%   With As = A*diag(1./s), z = s.*RHOV, and g = As'*(As*z-B):
%       norm(z-max(0,z-g),inf) <= tolerance*max(1,norm(As'*B,inf)).
%   The step uses a power-iteration Lipschitz estimate, without backtracking.
%   relativeChange is diagnostic only, not a stopping condition.
%
% NOTES
%   Zero columns denote unobserved voxels, which are fixed at zero.
%   These zeros do not establish that the physical density is zero.
%   residual = B-A*RHOV; positive bias means underprediction on average.
%   Sparse refers to matrix storage; no sparsity penalty is applied.
%   No grid assembly or assembleRhov dependency is included.
%
% EXAMPLE
%   A = sparse([100 20; 0 120; 50 50]);
%   b = A*[0.010; 0.012];
%   opts = struct('display',false);
%   [rhov,exitflag,info] = invertTomographySparseNNLSVector(A,b,opts);
%
% REFERENCES
%   Beck & Teboulle (2009), doi:10.1137/080716542.
%   O'Donoghue & Candes, Adaptive Restart for Accelerated Gradient Schemes,
%   https://arxiv.org/abs/1204.3982.

% =========================================================================
% INPUT
% =========================================================================

if nargin < 3 || isempty(opts)
    opts = struct;
end

if ~isnumeric(A) || ndims(A) ~= 2

    error( ...
        'invertTomographySparseNNLSVector:InvalidMatrix', ...
        'A must be a two-dimensional numeric matrix.');

end

if ~issparse(A)
    A = sparse(A);
end

b = b(:);

[nObs,nVoxels] = size(A);

if nObs == 0

    error( ...
        'invertTomographySparseNNLSVector:EmptyObservations', ...
        'A must contain at least one observation.');

end

if nVoxels == 0

    error( ...
        'invertTomographySparseNNLSVector:EmptyVoxels', ...
        'A must contain at least one voxel.');

end

if numel(b) ~= nObs

    error( ...
        'invertTomographySparseNNLSVector:DimensionMismatch', ...
        ['The number of elements in b (%d) must equal the number ' ...
         'of rows in A (%d).'], ...
        numel(b),nObs);

end

if any(~isfinite(b))

    error( ...
        'invertTomographySparseNNLSVector:InvalidObservations', ...
        'b contains NaN or Inf values.');

end

if any(~isfinite(nonzeros(A)))

    error( ...
        'invertTomographySparseNNLSVector:InvalidMatrix', ...
        'A contains NaN or Inf values.');

end

% =========================================================================
% DEFAULT OPTIONS
% =========================================================================

if ~isfield(opts,'maxIterations') || isempty(opts.maxIterations)
    opts.maxIterations = 5000;
end

if ~isfield(opts,'tolerance') || isempty(opts.tolerance)
    opts.tolerance = 1e-6;
end

if ~isfield(opts,'checkEvery') || isempty(opts.checkEvery)
    opts.checkEvery = 10;
end

if ~isfield(opts,'powerIterations') || isempty(opts.powerIterations)
    opts.powerIterations = 30;
end

if ~isfield(opts,'lipschitzSafety') || isempty(opts.lipschitzSafety)
    opts.lipschitzSafety = 1.05;
end

if ~isfield(opts,'display') || isempty(opts.display)
    opts.display = true;
end

if ~isfield(opts,'x0')
    opts.x0 = [];
end

% =========================================================================
% VALIDATE OPTIONS
% =========================================================================

validateattributes( ...
    opts.maxIterations, ...
    {'numeric'}, ...
    {'scalar','real','finite','integer','positive'}, ...
    mfilename, ...
    'opts.maxIterations');

validateattributes( ...
    opts.tolerance, ...
    {'numeric'}, ...
    {'scalar','real','finite','positive'}, ...
    mfilename, ...
    'opts.tolerance');

validateattributes( ...
    opts.checkEvery, ...
    {'numeric'}, ...
    {'scalar','real','finite','integer','positive'}, ...
    mfilename, ...
    'opts.checkEvery');

validateattributes( ...
    opts.powerIterations, ...
    {'numeric'}, ...
    {'scalar','real','finite','integer','positive'}, ...
    mfilename, ...
    'opts.powerIterations');

validateattributes( ...
    opts.lipschitzSafety, ...
    {'numeric'}, ...
    {'scalar','real','finite','positive'}, ...
    mfilename, ...
    'opts.lipschitzSafety');

% =========================================================================
% COLUMN SCALING
%
% Each column of A may have substantially different ray coverage.
%
% Define:
%
%       s_j = ||A(:,j)||_2
%
% and:
%
%       As = A * diag(1./s)
%
% If:
%
%       rhov = diag(1./s) * z
%
% then:
%
%       A*rhov = As*z
%
% and:
%
%       z >= 0  <=>  rhov >= 0
%
% Voxels not crossed by any ray have zero column norm. Their scale is set
% to one to avoid division by zero.
% =========================================================================

sRaw = sqrt(full(sum(A.^2,1)))';

unobservedVoxels = ...
    sRaw == 0;

nUnobservedVoxels = ...
    nnz(unobservedVoxels);

s = sRaw;

s(unobservedVoxels) = 1;

invS = ...
    1./s;

% -------------------------------------------------------------------------
% Scaled geometry matrix
% -------------------------------------------------------------------------

As = ...
    A*spdiags(invS,0,nVoxels,nVoxels);

% =========================================================================
% INITIAL GUESS
% =========================================================================

if isempty(opts.x0)

    z = ...
        zeros(nVoxels,1);

else

    x0 = ...
        opts.x0(:);

    if numel(x0) ~= nVoxels

        error( ...
            'invertTomographySparseNNLSVector:InvalidInitialGuess', ...
            ['opts.x0 must contain exactly %d elements, matching ' ...
             'the number of columns in A.'], ...
            nVoxels);

    end

    if any(~isfinite(x0))

        error( ...
            'invertTomographySparseNNLSVector:InvalidInitialGuess', ...
            'opts.x0 contains NaN or Inf values.');

    end

    % Enforce non-negativity
    x0 = ...
        max(0,x0);

    % Since:
    %
    %       rhov = z ./ s
    %
    % then:
    %
    %       z = rhov .* s

    z = ...
        x0.*s;

end

% Voxels with no ray coverage cannot be determined from the observations.
z(unobservedVoxels) = 0;

% FISTA extrapolated point
y = z;

% FISTA momentum parameter
t = 1;

% =========================================================================
% ESTIMATE THE LARGEST EIGENVALUE OF As'*As
%
%       L = ||As||_2^2
%
% Power iteration is used without explicitly forming As'*As.
% =========================================================================

v = ...
    ones(nVoxels,1);

v(unobservedVoxels) = 0;

nv = ...
    norm(v);

if nv > 0
    v = v./nv;
end

for p = 1:opts.powerIterations

    w = ...
        As'*(As*v);

    nw = ...
        norm(w);

    if nw == 0
        break
    end

    v = ...
        w./nw;

end

Av = ...
    As*v;

Lestimate = ...
    Av'*Av;

L = ...
    opts.lipschitzSafety*max(Lestimate,eps);

% =========================================================================
% CONVERGENCE SCALE
%
% The convergence condition is:
%
%       projectedGradientNorm
%
%       <=
%
%       tolerance * gradientScale
% =========================================================================

gradientScale = ...
    max(1,norm(As'*b,inf));

convergenceThreshold = ...
    opts.tolerance*gradientScale;

% =========================================================================
% DISPLAY
% =========================================================================

if opts.display

    fprintf('\n');
    fprintf('=============================================\n');
    fprintf(' Sparse NNLS tomography inversion\n');
    fprintf('=============================================\n\n');

    fprintf('Observations:        %d\n',nObs);
    fprintf('Voxels:              %d\n',nVoxels);
    fprintf('Unobserved voxels:   %d\n',nUnobservedVoxels);
    fprintf('nnz(A):              %d\n',nnz(A));

    fprintf( ...
        'Matrix density:      %.6f %%\n', ...
        100*nnz(A)/(nObs*nVoxels));

    fprintf( ...
        'L estimate:          %.6e\n', ...
        Lestimate);

    fprintf( ...
        'L used:              %.6e\n', ...
        L);

    fprintf( ...
        'Gradient scale:      %.6e\n', ...
        gradientScale);

    fprintf( ...
        'PG threshold:        %.6e\n\n', ...
        convergenceThreshold);

end

% =========================================================================
% INITIALIZATION
% =========================================================================

exitflag = 0;

converged = false;

nRestarts = 0;

projectedGradientNorm = inf;

objective = inf;

relativeChange = inf;

iteration = 0;

% =========================================================================
% FISTA LOOP
% =========================================================================

for iteration = 1:opts.maxIterations

    % ---------------------------------------------------------------------
    % Store previous solution
    % ---------------------------------------------------------------------

    zOld = z;

    % ---------------------------------------------------------------------
    % Gradient evaluated at extrapolated point y
    %
    %       f(y) = 0.5 ||As*y-b||^2
    %
    %       grad f(y) = As'*(As*y-b)
    % ---------------------------------------------------------------------

    residualY = ...
        As*y-b;

    gradient = ...
        As'*residualY;

    % ---------------------------------------------------------------------
    % Projected gradient step
    % ---------------------------------------------------------------------

    zNew = ...
        max( ...
        0, ...
        y-gradient/L);

    zNew(unobservedVoxels) = 0;

    % ---------------------------------------------------------------------
    % FISTA momentum
    % ---------------------------------------------------------------------

    tNew = ...
        0.5*(1+sqrt(1+4*t^2));

    beta = ...
        (t-1)/tNew;

    yNew = ...
        zNew + beta*(zNew-z);

    yNew(unobservedVoxels) = 0;

    % ---------------------------------------------------------------------
    % Adaptive restart
    %
    % Restart criterion:
    %
    %       (y-zNew)'*(zNew-z) > 0
    % ---------------------------------------------------------------------

    if dot( ...
            y-zNew, ...
            zNew-z) > 0

        tNew = 1;

        yNew = zNew;

        nRestarts = ...
            nRestarts+1;

    end

    % ---------------------------------------------------------------------
    % Update
    % ---------------------------------------------------------------------

    z = zNew;
    y = yNew;
    t = tNew;

    % ---------------------------------------------------------------------
    % Convergence check
    % ---------------------------------------------------------------------

    doCheck = ...
        iteration == 1 || ...
        mod(iteration,opts.checkEvery) == 0 || ...
        iteration == opts.maxIterations;

    if doCheck

        % -----------------------------------------------------------------
        % Residual
        % -----------------------------------------------------------------

        residualScaled = ...
            As*z-b;

        % -----------------------------------------------------------------
        % Gradient at current solution
        % -----------------------------------------------------------------

        gradientZ = ...
            As'*residualScaled;

        % -----------------------------------------------------------------
        % Projected-gradient mapping
        %
        % For z >= 0, an equivalent KKT condition is:
        %
        %       z = P_+(z-gradient)
        %
        % Therefore:
        %
        %       PG = z-P_+(z-gradient)
        % -----------------------------------------------------------------

        projectedGradient = ...
            z-max(0,z-gradientZ);

        projectedGradientNorm = ...
            norm(projectedGradient,inf);

        % -----------------------------------------------------------------
        % Objective
        % -----------------------------------------------------------------

        objective = ...
            0.5*(residualScaled'*residualScaled);

        % -----------------------------------------------------------------
        % Relative solution change
        % -----------------------------------------------------------------

        relativeChange = ...
            norm(z-zOld) / ...
            max(1,norm(zOld));

        % -----------------------------------------------------------------
        % Display
        % -----------------------------------------------------------------

        if opts.display && ...
                ( ...
                iteration == 1 || ...
                mod(iteration,100) == 0 || ...
                iteration == opts.maxIterations ...
                )

            fprintf( ...
                ['Iter %6d | obj = %.6e | PG = %.6e | ' ...
                 'target = %.6e | dX = %.3e | restarts = %d\n'], ...
                iteration, ...
                objective, ...
                projectedGradientNorm, ...
                convergenceThreshold, ...
                relativeChange, ...
                nRestarts);

        end

        % -----------------------------------------------------------------
        % KKT convergence
        % -----------------------------------------------------------------

        if projectedGradientNorm <= convergenceThreshold

            converged = true;

            exitflag = 1;

            break

        end

    end

end

% =========================================================================
% TRANSFORM FROM z TO RHOV
%
%       rhov = z ./ s
% =========================================================================

rhov = ...
    z.*invS;

% Numerical safety
rhov(rhov < 0) = 0;

% Keep unobserved voxels explicitly at zero
rhov(unobservedVoxels) = 0;

rhov = rhov(:);

if numel(rhov) ~= nVoxels

    error( ...
        'invertTomographySparseNNLSVector:InternalDimensionError', ...
        ['Internal error: rho_v contains %d elements but A contains ' ...
         '%d columns.'], ...
        numel(rhov),nVoxels);

end

% =========================================================================
% FINAL FIT
% =========================================================================

bFit = ...
    A*rhov;

% Residual convention:
%
%       residual = observation - fitted
%
% A positive bias therefore indicates that the fitted observations tend
% to be smaller than the measured observations.

residual = ...
    b-bFit;

residualNormSquared = ...
    residual'*residual;

residualNorm = ...
    sqrt(residualNormSquared);

rmse = ...
    sqrt(mean(residual.^2));

bias = ...
    mean(residual);

% =========================================================================
% FINAL PROJECTED GRADIENT
% =========================================================================

residualScaled = ...
    As*z-b;

gradientZ = ...
    As'*residualScaled;

projectedGradient = ...
    z-max(0,z-gradientZ);

projectedGradientNorm = ...
    norm(projectedGradient,inf);

objective = ...
    0.5*(residualScaled'*residualScaled);

% =========================================================================
% INFO
% =========================================================================

info = struct();

info.algorithm = ...
    'FISTA projected gradient NNLS with column scaling and adaptive restart';

info.iterations = ...
    iteration;

info.converged = ...
    converged;

info.objective = ...
    objective;

info.projectedGradientNorm = ...
    projectedGradientNorm;

info.gradientScale = ...
    gradientScale;

info.convergenceThreshold = ...
    convergenceThreshold;

info.projectedGradientRatio = ...
    projectedGradientNorm/convergenceThreshold;

info.Lestimate = ...
    Lestimate;

info.L = ...
    L;

info.nRestarts = ...
    nRestarts;

info.columnScale = ...
    s;

info.bFit = ...
    bFit;

info.residual = ...
    residual;

info.residualNorm = ...
    residualNorm;

info.residualNormSquared = ...
    residualNormSquared;

info.rmse = ...
    rmse;

info.bias = ...
    bias;

info.nnzA = ...
    nnz(A);

info.matrixDensity = ...
    nnz(A)/(nObs*nVoxels);

info.nObservations = ...
    nObs;

info.nVoxels = ...
    nVoxels;

info.nUnobservedVoxels = ...
    nUnobservedVoxels;

info.unobservedVoxels = ...
    unobservedVoxels;

info.relativeChange = ...
    relativeChange;

% Keep the raw solution vector because it follows exactly the columns of A.
info.rhovVector = ...
    rhov;

% =========================================================================
% FINAL DISPLAY
% =========================================================================

if opts.display

    fprintf('\n');
    fprintf('---------------------------------------------\n');

    if converged

        fprintf( ...
            'CONVERGED after %d iterations\n', ...
            iteration);

    else

        fprintf( ...
            'MAXIMUM ITERATIONS reached: %d\n', ...
            iteration);

    end

    fprintf( ...
        'Objective:           %.6e\n', ...
        objective);

    fprintf( ...
        'Projected gradient:  %.6e\n', ...
        projectedGradientNorm);

    fprintf( ...
        'PG threshold:        %.6e\n', ...
        convergenceThreshold);

    fprintf( ...
        'PG / threshold:      %.3f\n', ...
        info.projectedGradientRatio);

    fprintf( ...
        'Restarts:            %d\n', ...
        nRestarts);

    fprintf( ...
        'Residual norm:       %.6e\n', ...
        residualNorm);

    fprintf( ...
        'Residual RMSE:       %.6e kg m^-2\n', ...
        rmse);

    fprintf( ...
        'Residual bias:       %.6e kg m^-2\n', ...
        bias);

    fprintf( ...
        'Unobserved voxels:   %d / %d\n', ...
        nUnobservedVoxels,nVoxels);

    fprintf('---------------------------------------------\n\n');

end

end