%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Supplamentary material of the paper:
% "Distributed large scale network utility maximization",
% by D. Bickson, Y. Tock, A. Zymnis, S. Boyd and D. Dolev.
% Submitted to ISIT 2009.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation writen by A. Zymnis, Stanford
% The GaBP implemented was writen by D. Bickson, IBM Haifa Lab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Small test scenario

addpath('..');
clear
rand('state',sum(100*clock));

% Routing matrix parameters
m = 2*round(10^2); %links
n = round(10^2);   %flows
links_per_flow = 3;
c_min = .1;
c_max = 1;

% Routing matrix
R = rand(m,n)<(links_per_flow/m);
ind_rows = setdiff(1:m,find(sum(R,2)==0));
ind_cols = setdiff(1:n,find(sum(R,1)==0));
R = sparse(R(ind_rows,ind_cols));
[m,n] = size(R);
fprintf(1,'Creating routing matrix with m = %d flows and n = %d links\n\n',m,n);

% Link capacities
c = c_min + (c_max-c_min)*rand(m,1);

% pdIP params
params.max_iters = 100;
params.alpha = 0.0001;
params.beta = 0.5;
params.kappa = 10;
params.tol = .00001;      % relative duality gap (U-L)/L toleration

% PCG parameters
params.PCG_MAX_ITER = 100;
params.PCG_tol = 0.1;
params.PCG_WARM_START = 1;    %warm start flag, set 0 to disable
params.PCG_PRECOND = 1;       %preconditioning flag
params.status_print_flag = 1;

w = ones(n,1);
funcs.u = @(f) log(f);
funcs.g = @(f) 1./f;
funcs.h = @(f) -1./f.^2;
L_const_fct = n; 
funcs.dgap = @(R,c,f,lambda,mu) -sum(w.*log(f))-(n - lambda'*c + sum(w.*log(R'*lambda)));

% GaBP solver
[f21, lambda21, mu21, failure_flag21, res21] = dense_pdip_solver_gabp2(funcs, R, c, params);
if ~failure_flag21
    fprintf(1,'GaBP finished\n\n');
else
    fprintf(1,'GaBP failed\n\n');
end

% Truncated Newton solver
[f2, lambda2, mu2, failure_flag2, res2] = trunc_pdip_solver(funcs, R, c, params);
if ~failure_flag2
    fprintf(1,'Trunc PDIP finished in %d steps\n\n',sum(res2.pcg_iters));
else
    fprintf(1,'Trunc PDIP failed\n\n');
end

% DD params
dparams.max_iters = 2000;
dparams.tol = .001;
dparams.status_print_flag = 1;
dparams.vary_alpha = 0;
dparams.alpha = 2.4;

[df, dfailure_flag, dres] = dd_solver(R, c, w, dparams);
if ~dfailure_flag
    fprintf(1,'DD finished in %d steps\n',length(dres.U));
else
    fprintf(1,'DD failed\n');
end

figure;
semilogy(cumsum(res21.pcg_iters),res21.rel_gap,'g-','LineWidth',2)
hold on
semilogy(cumsum(res2.pcg_iters),res2.rel_gap,'b-','LineWidth',2)
hold on
semilogy((dres.U-dres.L)/n,'r--','LineWidth',2)
legend('Customized Newton via GaBP', 'Trancated Newton via PCG', 'Dual decomposition');
set(gca,'FontSize',12);
xlabel('Iteration number')
ylabel('Duality Gap')

