% ista_lasso_demo.m
% Companion code for the lecture notes on numerical optimization and
% l1 regularization. Three experiments:
%   1. sparse recovery via ISTA
%   2. convergence of the objective gap
%   3. the lasso regularization path
%
% Solves   min_x  0.5*||A x - b||^2 + lambda*||x||_1
% by proximal gradient descent (ISTA). The proximal operator of the
% l1 norm is the soft-thresholding map.

rng(0)

% problem setup: underdetermined sparse recovery
n = 200;            % number of unknowns
m = 80;             % number of measurements, m < n
s = 8;              % number of nonzero entries in the ground truth
sigma = 0.02;       % measurement noise level

A = randn(m, n) / sqrt(m);
xtrue = zeros(n, 1);
support = randperm(n, s);
xtrue(support) = randn(s, 1);
b = A * xtrue + sigma * randn(m, 1);

% step size from the Lipschitz constant of the smooth part
L = norm(A)^2;      % spectral norm squared = largest eigenvalue of A'A
t = 1 / L;

% smallest lambda for which the all-zero vector is optimal
lambda_max = max(abs(A' * b));
lambda = 0.1 * lambda_max;

% soft-thresholding operator, the prox of the l1 norm
soft = @(z, th) sign(z) .* max(abs(z) - th, 0);

% experiment 1: recover the sparse signal
maxit = 2000;
[xhat, Fhist] = ista(A, b, lambda, t, maxit, soft);

figure
stem(xtrue, 'filled')
hold on
stem(xhat, 'o')
hold off
axis tight
grid on
xlabel('$\mathrm{index}\ i$', 'Interpreter', 'latex')
ylabel('$x_i$', 'Interpreter', 'latex')
title('$\mathrm{Sparse\ recovery:\ truth\ vs\ ISTA}$', 'Interpreter', 'latex')
legend({'$x_{\mathrm{true}}$', '$\hat{x}$'}, 'Interpreter', 'latex')

% experiment 2: convergence of the objective gap F(x_k) - F(x_star)
% obtain a reference optimum with a long run
[~, Flong] = ista(A, b, lambda, t, 20000, soft);
Fstar = Flong(end);
gap = Fhist - Fstar;
gap(gap <= 0) = NaN;    % drop nonpositive tail for the log axis

figure
loglog(1:maxit, gap)
hold on
loglog(1:maxit, gap(1) ./ (1:maxit), '--')
hold off
axis tight
grid on
xlabel('$\mathrm{iteration}\ k$', 'Interpreter', 'latex')
ylabel('$F(x_k) - F(x_\star)$', 'Interpreter', 'latex')
title('$\mathrm{ISTA\ convergence}$', 'Interpreter', 'latex')
legend({'$\mathrm{objective\ gap}$', '$O(1/k)\ \mathrm{reference}$'}, ...
    'Interpreter', 'latex')

% experiment 3: the regularization path
nl = 60;
lambdas = logspace(log10(lambda_max), log10(0.01 * lambda_max), nl);
Xpath = zeros(n, nl);
xwarm = zeros(n, 1);    % warm start across the sweep
for k = 1:nl
    xwarm = ista_warm(A, b, lambdas(k), t, 500, soft, xwarm);
    Xpath(:, k) = xwarm;
end

figure
semilogx(lambdas, Xpath')
axis tight
grid on
xlabel('$\lambda$', 'Interpreter', 'latex')
ylabel('$x_\lambda$', 'Interpreter', 'latex')
title('$\mathrm{Lasso\ regularization\ path}$', 'Interpreter', 'latex')

% report recovered support
fprintf('true support:      %s\n', mat2str(sort(support)))
fprintf('recovered support: %s\n', mat2str(find(abs(xhat) > 1e-6)'))


% local functions

function [x, Fhist] = ista(A, b, lambda, t, maxit, soft)
    % proximal gradient descent on the lasso objective from a zero start
    x = zeros(size(A, 2), 1);
    Fhist = zeros(maxit, 1);
    for k = 1:maxit
        grad = A' * (A * x - b);
        x = soft(x - t * grad, t * lambda);
        Fhist(k) = 0.5 * norm(A * x - b)^2 + lambda * sum(abs(x));
    end
end

function x = ista_warm(A, b, lambda, t, maxit, soft, x0)
    % same iteration with a supplied warm start, used for the path sweep
    x = x0;
    for k = 1:maxit
        grad = A' * (A * x - b);
        x = soft(x - t * grad, t * lambda);
    end
end
