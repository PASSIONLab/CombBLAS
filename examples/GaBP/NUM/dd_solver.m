function [f, failure_flag, results] = dd_solver(R, c, w, params)
% DD_SOLVER.M
% 
% solves:
%           minimize    -sum(w.*log(f))
%           subject to  R*f <= c
%
% using dual decomposition
% params: max_iter, tol, status_print_flag
% results: U, L, time

if params.status_print_flag
    fprintf(1,'DD_SOLVER status ... \n');
end

%% ----------- dual decomposition -----------
MAX_ITER_DD = params.max_iters;
[m, n] = size(R);
%% ----------- initialization -----------
% primal and dual variables
f = ones(n,1);    
eta = (R*f)./c; % Scale flows to feasibility
f = 0.99*f./max(R'*spdiags(eta,0,m,m),[],2);
U(1) = -sum(w.*log(f));
%lambda_dd = 1./(c-R*f); 
lambda_dd = 1*ones(m,1);
Rtlambda_dd = R'*lambda_dd;
L_const_fct = sum(w.*(1-log(w)));   %caching
L(1) = sum(w.*log(Rtlambda_dd))-lambda_dd'*c+L_const_fct;
tol = params.tol;
alpha_dd = params.alpha;
failure_flag = 0;
for i = 1:MAX_ITER_DD
    tic
    % Flow updates
    f = w./(Rtlambda_dd);
    
    % Scale flows to feasibility
    Rf = R*f;                %caching
    eta = (Rf)./c;
    f_feas = f./max(R'*spdiags(eta,0,m,m),[],2);
    U(i) = -sum(w.*log(f_feas));
   
    % Dual subgradient
    g = c-Rf;
    
    % Price update
    orig_lambda_dd = lambda_dd;
    lambda_dd = max(orig_lambda_dd - alpha_dd*1*g,0);
    Rtlambda_dd = R'*lambda_dd; %caching
    while params.vary_alpha
        %if all(Rtlambda_dd>0)&&(sum(log(Rtlambda_dd))-lambda_dd'*c+n>0)
        if all(Rtlambda_dd>0)
            break
        end
        alpha_dd = alpha_dd/2;
        if alpha_dd < 1e-6
            failure_flag = 1;
            break;
        end
        lambda_dd = max(orig_lambda_dd - alpha_dd*1*g,0);
        Rtlambda_dd = R'*lambda_dd; %caching
    end
    
    if all(Rtlambda_dd>0)
        L(i) = sum(w.*log(Rtlambda_dd))-lambda_dd'*c+L_const_fct;
    else
        failure_flag = 1;
        break;
    end
    time = toc;
    results.time(i) = time;
    
    if params.status_print_flag && (mod(i,50)==0)
        fprintf(1,'Iteration: %3d, Duality gap per flow: %3.3e\n',i,(U(i)-L(i))/n)
    end    
    
    % Breaking condition
    if ((U(i)-L(i))/n<tol)||(failure_flag)
        break;
    end
end
f = f_feas;
results.U = U;
results.L = L;

failure_flag = ((U(end)-L(end))/n > tol);
if params.status_print_flag
    if failure_flag
        fprintf(1,'DD_SOLVER failed ... \n');
    else
        fprintf(1,'DD_SOLVER successfull ... \n');
    end
end