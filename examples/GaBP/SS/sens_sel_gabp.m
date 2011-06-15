% Source code for the paper:
% Distributed Sensor Selection via Gaussian Belief Propagation
% by Danny Bickson and Danny Dolev
% submitted to Allerton 2009.
% http://arxiv.org/abs/0907.0931

% Original simulation generates the numerical example in the paper
% Sensor Selection via Convex Optimization
% www.stanford.edu/~boyd/papers/sensor_selection.html
% May 2008 Siddharth Joshi & Stephen Boyd
%
% Modified and created a distributed version: July 2009 by Danny Bickson


%% Solves the problem
%	maximize log det (sum_{i=1}^m z_i a_i a_i^T) + kappa sum_{i=1}^m(log(z_i)+ log(1-z_i))
%	subject to sum(z) = k
%			   0 <= z_i <= 1, i=1,..., m
% variable z in R^m
% problem parameters kappa (>0), a_1, ..., a_m in R^n
%
% see paper Sensor Selection via Convex Optimization
% www.stanford.edu/~boyd/papers/sensor_selection.html
%
% Nov 2007 Siddharth Joshi & Stephen Boyd
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [zhat L ztilde Utilde iter] = sens_sel_gabp(A, k)


% Newton's method parameters
MAXITER  = 30;
NT_TOL = 1e-3;
GAP = 1.005;
% Backtracking line search parameters
alpha = 0.01;
beta = 0.5;

[m n] = size(A);
z = ones(m,1)*(k/m); % initialize
g = zeros(m,1);
ones_m = ones(m,1);
kappa = log(GAP)*n/m; 
% guarantees GM of lengths of semi-axes of ellipsoid corresponding to 
% ztilde <= 1.01 from optimal
addpath('..');

fprintf('\nIter.  Step_size  Newton_decr.  Objective  log_det\n');

fz = -log(det(A'*diag(z)*A)) - kappa*sum(log(z) + log(1-z));

fprintf('   0\t  -- \t     --   %10.3f  %10.3f\n', -fz, log(det(A'*diag(z)*A)));

for iter=1:MAXITER

%     W = inv(A'*diag(z)*A);
%     V = A*W*A';
%     g_orig = -diag(V)- kappa*(1./z - 1./(1-z));
    
    AA = sparse(n,n);
    BB = A';
    CC = A;
    DD = sparse(-diag(1./z));
    
    W01 =  [speye(n,n) BB; CC DD ];
    [w1,w2] = sparse_gabp(W01,ones(m+n,1),10,0.00001);

    w2 = w2(n+1:end);
    w02 = (diag(DD).^2).*(w2' - diag(inv(DD)));
    g = - w02 - kappa*(1./z - 1./(1-z));
    H=sparse(diag(g.*g)) + kappa*diag(1./(z.^2) + 1./((1-z).^2));
    
    Hinc = H;%+eye(length(H))*.3;
    Hinvg = sparse_gabp(Hinc,g,100,0.00001)';
    Hinv1 = sparse_gabp(Hinc,ones_m, 100,0.00001)';
    dz = -Hinvg + ((ones_m'*Hinvg) / (ones_m'*Hinv1))*Hinv1;
    
    deczi = find(dz < 0);
    inczi = find(dz > 0);
    s = min([1; 0.99*[-z(deczi)./dz(deczi) ; (1-z(inczi))./dz(inczi)]]);

    while (1)
        zp = z + s*dz;
        fzp = -log(det(A'*diag(zp)*A)) - kappa*sum(log(zp) + log(1-zp));

        if (fzp <= fz + alpha*s*g'*dz)
            break;
        end
        s = beta*s;
    end
    z = zp; fz = fzp;
    
    fprintf('%4d %10.3f %10.3f %10.3f %10.3f\n', iter, s, -g'*dz/2, -fz, log(det(A'*diag(z)*A)));

    if(-g'*dz/2 <= NT_TOL)
        break;
    end
    if(abs(s) < 1e-4)%DB!
        break;
    end
end

zsort=sort(z); thres=zsort(m-k); zhat=(z>thres);
L = log(det(A'*diag(zhat)*A));
ztilde = z; 
Utilde = log(det(A'*diag(z)*A)) + 2*m*kappa;
%iter = iter*10
