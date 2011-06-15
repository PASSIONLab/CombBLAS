function [f,lambda, mu, failure_flag, results] = dense_pdip_solver_sparse_gabp(funcs,R, c, params)
% DENSE_PDIP_SOLVER.M
% 
% solves:
%           minimize    -sum(U_i(f_i))
%           subject to  R*f <= c
%                         f >= 0
%
% using a direct netwon primal-dual interior point method
% funcs:  u, g, h
% params: max_iters, alpha, beta, nu, tol (for pdip method)
%         PCG_MAX_ITER, PCG_tol, PCG_WARM_START, PCG_DIAG_PRECOND
% results: 
% [iter_number, U, eta, stepsize, time]


%% ----------- problem data -----------
[m, n] = size(R);

%% ----------- function handles ------------
u = funcs.u;
g = funcs.g;
h = funcs.h;
dgap = funcs.dgap;

%% ----------- primal dual IP parameters -----------
MAX_ITERS = params.max_iters;
alpha = params.alpha;
beta = params.beta;
kappa = params.kappa;
tol = params.tol;    % surrogate gap tolerance

%% ----------- initialization -----------
% primal and dual variables
f = ones(n,1);    
tau = (R*f)./c; % Scale flows to feasibility
f = 0.99*f./max(R'*spdiags(tau,0,m,m),[],2);
%lambda = 1./(c-R*f);
%mu = 1./f;
lambda = ones(m,1);
mu = ones(n,1);
df = zeros(n,1); Rdf = R*df;
%eta = [c-R*f; f]'*[lambda; mu];
eta = dgap(R,c,f,lambda,mu);

%% ----------- results logging -----------
% keep a record of the results
results.iter_number(1) = 0;
results.U(1) = sum(u(f));
results.rel_gap(1) = eta/n;
results.stepsize(1) = 0;
results.iter_times(1) = 0;

%% -------------------


 retn = cell(length(2*n+m),1);
%% ----------- main iteration -----------
for iter = 1:MAX_ITERS
    tic
    % Determine t
    t = kappa*m/eta;

    % Compute primal dual search direction
    s = c-R*f;    
    %df = (spdiags(-h(f),0,n,n)+spdiags(mu./f,0,n,n)+R'*spdiags(lambda./s,0,m,m)*R)\...
    %    (g(f)-(1/t)*R'*(1./s)+(1/t)./f); %%dannybi
    
    bigMAT = [spdiags(-h(f),0,n,n) R' -speye(n,n); ...
              R -spdiags(s./lambda,0,m,m) sparse(m,n); ...
              -speye(n,n) sparse(n,m) -spdiags(f./mu,0,n,n)];
    bigvec = [ -g(f) + R'*lambda - mu; ...
               -s + (1/t)./lambda; ...
               -f + (1/t)./mu];
   
           
    ret = bigMAT\-bigvec;

    %learn matrix topology only once
    if (iter == 1)
         for i=1:length(bigMAT)
            neighbors = find (bigMAT(i,:) ~= 0);
            neighbors(neighbors == i) = [];
            retn{i} = neighbors;
         end
    end
     
   % [bigvec,J,r ] = asynch_GBP(bigMAT, -bigvec, 100, 1e-3);
    [bigvec,J,r ] = sparse_gabp_opt(bigMAT, -bigvec, 100, 1e-3,retn,iter > 1);
    disp(['GBP relative error to direct method is: ', num2str(norm(ret - bigvec'))]);
    bigvec = bigvec';
    df = bigvec(1:n);
    dlambda = bigvec(n+1:n+m);
    dmu = bigvec(n+m+1:end);
    
    
%     df2 = (spdiags(-h(f),0,n,n)+spdiags(mu./f,0,n,n)+R'*spdiags(lambda./s,0,m,m)*R)\...
%         (g(f)-(1/t)*R'*(1./s)+(1/t)./f); %%dannybi
%     Rdf = R*df2; %caching
%     dmu2 = -mu-(mu./f).*df+(1/t)./f;
%     dlambda2 = -lambda+(lambda./s).*Rdf+(1/t)./s;

    
    
    
    %mat1 = spdiags(-h(f),0,n,n)+spdiags(mu./f,0,n,n)+R'*spdiags(lambda./s,0,m,m)*R;
%     grad = (g(f)-(1/t)*R'*(1./s)+(1/t)./f);
%     diagonal = max(abs(diag(mat1) - sum(mat1')'));
%     %[df,Pf] = gabp(mat1 + 0.2*diagonal*eye(length(mat1)), grad, 100, 0.0001);
%     xj = zeros(length(grad),10);
%     xj(:,1) = grad;
%     norms = zeros(1,10);
%     for l=2:10
%           	Minc = mat1 + eye(length(mat1)) *  0.5*diagonal;
%             [b_est,J] = asynch_GBP(Minc, grad - mat1*xj(:,l-1), 100, 1e-6);
%             %b_est = inv(Minc)*(y - M*xj(:,l-1));
%             xj(:,l) = xj(:,l-1) + b_est';
%             df = xj(:,l);
%             disp(['error norm for round ', num2str(l), ' is ', num2str(norm(grad - mat1*xj(:,l)))]);
%             norms(l) = norm(grad - mat1*xj(:,l));
%     end
%         
%     %plot(norms);
    
    
    %df = df';
    Rdf = R*df; %caching
%     dmu = -mu-(mu./f).*df+(1/t)./f;
%     dlambda = -lambda+(lambda./s).*Rdf+(1/t)./s;
        
    % Find the maximum gamma for which s-gamma*R*df>0
    % Select gamma to be .99 this value    
    ind_f = find(Rdf>0);
    gamma = 1;
    if size(ind_f) > 0
        gamma = min(1,0.99*min(s(ind_f)./Rdf(ind_f)));
    end

    % Force f+gamma*df > 0
    if min(f+gamma*df)<=0
        % Find the maximum s for which f+gamma*df > 0
        % Select gamma to be .99 this value
        ind_l = find(df<0);
        gamma = min(gamma,0.99*min(-f(ind_l)./df(ind_l)));
    end
    
    % Force lambda+gamma*dlambda > 0
    if min(lambda+gamma*dlambda)<0
        % Find the maximum s for which lambda+gamma*dlambda > 0
        % Select gamma to be .99 this value
        ind_l = find(dlambda<0);
        gamma = min(gamma,0.99*min(-lambda(ind_l)./dlambda(ind_l)));
    end
    
    % Force mu+gamma*dmu > 0
    if min(mu+gamma*dmu)<0
        % Find the maximum s for which mu+gamma*dmu > 0
        % Select gamma to be .99 this value
        ind_l = find(dmu<0);
        gamma = min(gamma,0.99*min(-mu(ind_l)./dmu(ind_l)));
    end
    
    % Line search
    Rtlambda = R'*lambda; Rtdlambda = R'*dlambda; %caching
    r_t =  [-g(f)+Rtlambda-mu; 
                     lambda.*s-(1/t)*ones(m,1); 
                     mu.*f-(1/t)*ones(n,1)];
    norm_res = norm(r_t);
    ddgamma_res = (1/norm_res)*r_t'*[df.*(-h(f))+Rtdlambda-dmu;-lambda.*Rdf+s.*dlambda; mu.*df+f.*dmu];
    line_search_failed = 0;
    while 1
        r_dual = -g(f+gamma*df)+(Rtlambda+gamma*Rtdlambda)-mu-gamma*dmu;
        r_central = [(lambda+gamma*dlambda).*(s-gamma*Rdf)-(1/t)*ones(m,1);
                     (mu+gamma*dmu).*(f+gamma*df)-(1/t)*ones(n,1)];
        new_norm_res = norm([r_dual; r_central]);
        if new_norm_res <= norm_res+alpha*gamma*ddgamma_res
            break
        end
        
        if(gamma < 1e-11)
            fprintf(1,'Ran into numerical problems...\n\n')
            line_search_failed = 1; gamma = 0;
            break;
        end

        gamma = gamma*beta;
    end

    % Update
    f = f+gamma*df; 
    lambda = lambda+gamma*dlambda;
    mu = mu+gamma*dmu;
    %eta = [c-R*f; f]'*[lambda; mu];
    eta = dgap(R,c,f,lambda,mu);
    
    % results
    % [iter_number, U, eta, stepsize, time]
    iter_time = toc;
    status = [iter sum(u(f)) eta/n gamma];
    fprintf(1,'Iteration: %2d   Utility: %3.3e  Surr. Dgap: %3.3e  Stepsize: %1.3f GaBP iter %2d time%f\n'...
        ,status(1),status(2),status(3),status(4),r,iter_time);
    
    % keep a record of the results
    results.iter_number(iter+1) = status(1);
    results.U(iter+1) = status(2);
    results.rel_gap(iter+1) = status(3);
    results.stepsize(iter+1) = status(4);
    results.iter_times(iter+1) = iter_time;
    results.pcg_iters(iter+1) = r;
    % Break condition
    if (eta/n <= tol)||(line_search_failed)
        break
    end
end
failure_flag = (eta/n > tol);
fprintf(1,'Done!\n')

