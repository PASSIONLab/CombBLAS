%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Supplamentary material of the paper:
% "Distributed large scale network utility maximization",
% by D. Bickson, Y. Tock, A. Zymnis, S. Boyd and D. Dolev.
% Submitted to ISIT 2009.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation writen by A. Zymnis, Stanford
% The GaBP implemented was writen by D. Bickson, IBM Haifa Lab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% large test scenario


%% Simple test of ls_num solver
clear
rand('state',sum(100*clock));
addpath('..');

n = 1e5;   %number of flows
m = 2*1e5; %number of links
links_per_flow = 1.5;

%% Create random routing matrix
R = (sprand(m,n,links_per_flow/m)>0);

% Remove empty rows and columns of R
ind1 = find(sum(R,2)==0)'; ind_rows = setdiff(1:m,ind1);
ind2 = find(sum(R,1)==0); ind_cols = setdiff(1:n,ind2);
R = R(ind_rows,ind_cols);
[m,n] = size(R);
fprintf(1,'Created routing matrix with m = %e flows and n = %e links\n',m,n);

%% Random link capacities uniform on [0.1,1]
c = 0.1 + 0.9*rand(m,1);

%% Interior-point method params
params.max_iters = 100;
params.alpha = 0.0001;
params.beta = 0.5;
params.kappa = 10;
params.tol = .0001;      % relative duality gap (U-L)/L tolerance
params.status_print_flag = 1;

%% PCG parameters
params.PCG_MAX_ITER = 100;
params.PCG_tol = 0.1;
params.PCG_WARM_START = 1;    %warm start flag, set 0 to disable
params.PCG_PRECOND = 1;       %preconditioning flag

%% Utility function and first and second derivatives and duality gap
funcs.u = @(f) log(f);
funcs.g = @(f) 1./f;
funcs.h = @(f) -1./f.^2;
L_const_fct = n; 
funcs.dgap = @(R,c,f,lambda,mu) -sum(1.*log(f))-(n-lambda'*c + sum(1.*log(R'*lambda)));

%% Solve problem
% uncomment the following to run GaBP (may be slow!)
% [f, lambda, mu, failure_flag, res] = dense_pdip_solver_sparse_gabp(funcs, R, c, params);
% if ~failure_flag
%     fprintf(1,'PDIP finished in %d steps\n',sum(res.pcg_iters));
% else
%     fprintf(1,'PDIP failed\n');
% end

[f2, lambda2, mu2, failure_flag2, res2] = trunc_pdip_solver(funcs, R, c, params);
if ~failure_flag2
    fprintf(1,'PDIP2 finished in %d steps\n',sum(res2.pcg_iters));
else
    fprintf(1,'PDIP2 failed\n');
end


figure
%semilogy(cumsum(res.pcg_iters),res.rel_gap,'b-','LineWidth',1)
%hold on;
semilogy(cumsum(res2.pcg_iters),res2.rel_gap,'g-','LineWidth',1)
set(gca,'FontSize',12);
xlabel('Number of iterations')
ylabel('Duality dgap')
title('GaBP Newton method vs. PCG Truncated Newton method');
%legend('GaBP', 'PCG');
legend('PCG');
%print -depsc myfigure;
%saveas(gcf,'myfigure','fig');

