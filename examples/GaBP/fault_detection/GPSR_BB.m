function [x,it,convergence,x_debias,objective,times,debias_start,mses,taus]= ...
    GPSR_BB(y,A,tau,show,varargin)
%
% GPSR_BB version 5.0, December 4, 2007
%
% This function solves the convex problem 
% arg min_x = 0.5*|| y - A x ||_2^2 + tau || x ||_1
% using the algorithm GPSR-BB, described in the following paper
%
% "Gradient Projection for Sparse Reconstruction: Application
% to Compressed Sensing and Other Inverse Problems"
% by Mario A. T. Figueiredo, Robert D. Nowak, Stephen J. Wright,
% Journal of Selected Topics on Signal Processing, December 2007
% (to appear).
%
% 
% -----------------------------------------------------------------------
% Copyright (2007): Mario Figueiredo, Robert Nowak, Stephen Wright
% 
% GPSR is distributed under the terms
% of the GNU General Public License 2.0.
% 
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
% 
% 
% Please check for the latest version of the code and paper at
% www.lx.it.pt/~mtf/GPSR
%
%  ===== Required inputs =============
%
%  y: 1D vector or 2D array (image) of observations
%     
%  A: if y and x are both 1D vectors, A can be a 
%     k*n (where k is the size of y and n the size of x)
%     matrix or a handle to a function that computes
%     products of the form A*v, for some vector v.
%     In any other case (if y and/or x are 2D arrays), 
%     A has to be passed as a handle to a function which computes 
%     products of the form A*x; another handle to a function 
%     AT which computes products of the form A'*x is also required 
%     in this case. The size of x is determined as the size
%     of the result of applying AT.
%
%  tau: usually, a non-negative real parameter of the objective 
%       function (see above). It can also be an array, the same 
%       size as x, with non-negative entries; in this  case,
%       the objective function weights differently each element 
%       of x, that is, it becomes
%       0.5*|| y - A x ||_2^2 + tau^T * abs(x)
%
%  ===== Optional inputs =============
%
%  
%  'AT'    = function handle for the function that implements
%            the multiplication by the conjugate of A, when A
%            is a function handle. If A is an array, AT is ignored.
%
%  'StopCriterion' = type of stopping criterion to use
%                    0 = algorithm stops when the relative 
%                        change in the number of non-zero 
%                        components of the estimate falls 
%                        below 'ToleranceA'
%                    1 = stop when the relative 
%                        change in the objective function 
%                        falls below 'ToleranceA'
%                    2 = stop when the norm of the difference between 
%                        two consecutive estimates, divided by the norm
%                        of one of them falls below toleranceA
%                    3 = stop when LCP estimate of relative
%                        distance to solution
%                        falls below 'ToleranceA'
%                    4 = stop when the objective function 
%                        becomes equal or less than toleranceA.
%                    5 = stop when the norm of the difference between 
%                        two consecutive estimates, divided by the norm
%                        of one of them falls below toleranceA
%                    Default = 3.
%
%  'ToleranceA' = stopping threshold; Default = 0.01
% 
%  'Debias'     = debiasing option: 1 = yes, 0 = no.
%                 Default = 0.
%
%  'ToleranceD' = stopping threshold for the debiasing phase:
%                 Default = 0.0001.
%                 If no debiasing takes place, this parameter,
%                 if present, is ignored.
%
%  'MaxiterA' = maximum number of iterations allowed in the
%               main phase of the algorithm.
%               Default = 10000
%
%  'MiniterA' = minimum number of iterations performed in the
%               main phase of the algorithm.
%               Default = 5
%
%  'MaxiterD' = maximum number of iterations allowed in the
%               debising phase of the algorithm.
%               Default = 200
%
%  'MiniterD' = minimum number of iterations to perform in the
%               debiasing phase of the algorithm.
%               Default = 5
%
%  'Initialization' must be one of {0,1,2,array}
%               0 -> Initialization at zero. 
%               1 -> Random initialization.
%               2 -> initialization with A'*y.
%           array -> initialization provided by the user.
%               Default = 0;
%
%  'Monotone' =  enforce monotonic decrease in f, or not? 
%               any nonzero -> enforce monotonicity
%               0 -> don't enforce monotonicity.
%               Default = 1;
%
%  'Continuation' = Continuation or not (1 or 0) 
%                   Specifies the choice for a continuation scheme,
%                   in which we start with a large value of tau, and
%                   then decrease tau until the desired value is 
%                   reached. At each value, the solution obtained
%                   with the previous values is used as initialization.
%                   Default = 0
%
% 'ContinuationSteps' = Number of steps in the continuation procedure;
%                       ignored if 'Continuation' equals zero.
%                       If -1, an adaptive continuation procedure is used.
%                       Default = -1.
% 
% 'FirstTauFactor'  = Initial tau value, if using continuation, is
%                     obtained by multiplying the given tau by 
%                     this factor. This parameter is ignored if 
%                     'Continuation' equals zero.
%                     Default = such that the first tau is equal to
%                               0.5*max(abs(AT(y))).
% 
%  'True_x' = if the true underlying x is passed in 
%                this argument, MSE evolution is computed
%
%  'AlphaMin' = the alphamin parameter of the BB method.
%               (See the paper for details)
%               Default = 1e-30;
%
%  'AlphaMax' = the alphamax parameter of the BB method.
%               (See the paper for details)
%               Default = 1e30;
%
%  'Verbose'  = work silently (0) or verbosely (1)
%
% ===================================================  
% ============ Outputs ==============================
%   x = solution of the main algorithm
%
%   x_debias = solution after the debiasing phase;
%                  if no debiasing phase took place, this
%                  variable is empty, x_debias = [].
%
%   objective = sequence of values of the objective function
%
%   times = CPU time after each iteration
%
%   debias_start = iteration number at which the debiasing 
%                  phase started. If no debiasing took place,
%                  this variable is returned as zero.
%
%   mses = sequence of MSE values, with respect to True_x,
%          if it was given; if it was not given, mses is empty,
%          mses = [].
% ========================================================

% test for number of required parametres
if (nargin-length(varargin)) ~= 4
  error('Wrong number of required parameters');
end

% flag for initial x (can take any values except 0,1,2)
Initial_X_supplied = 3333;

% Set the defaults for the optional parameters
stopCriterion = 2;%3;
tolA = 0.01;
tolD = 0.0001;
debias = 0;
maxiter = 10000;
maxiter_debias = 500;
miniter = 5;
miniter_debias = 5;
init = 0;
enforceMonotone = 1;
alphamin = 1e-30;
alphamax = 1e30;
compute_mse = 0;
AT = 0;
verbose = 0;%1;
continuation = 0;
cont_steps = -1;
firstTauFactorGiven = 0;
conv = zeros(size(A,2),100);

% Set the defaults for outputs that may not be computed
debias_start = 0;
x_debias = [];
mses = [];

% Read the optional parameters
if (rem(length(varargin),2)==1)
  error('Optional parameters should always go by pairs');
else
  for i=1:2:(length(varargin)-1)
    switch upper(varargin{i})
     case 'STOPCRITERION'
       stopCriterion = varargin{i+1};
     case 'TOLERANCEA'       
       tolA = varargin{i+1};
     case 'TOLERANCED'
       tolD = varargin{i+1};
     case 'DEBIAS'
       debias = varargin{i+1};
     case 'MAXITERA'
       maxiter = varargin{i+1};
     case 'MAXITERD'
       maxiter_debias = varargin{i+1};
     case 'MINITERA'
       miniter = varargin{i+1};
     case 'MINITERD'
       miniter_debias = varargin{i+1};
     case 'INITIALIZATION'
       if prod(size(varargin{i+1})) > 1   % initial x supplied as array
	 init = Initial_X_supplied;    % flag to be used below
	 x = varargin{i+1};
       else 
	 init = varargin{i+1};
       end
     case 'MONOTONE'
       enforceMonotone = varargin{i+1};
     case 'CONTINUATION'
       continuation = varargin{i+1};  
     case 'CONTINUATIONSTEPS' 
       cont_steps = varargin{i+1};
     case 'FIRSTTAUFACTOR'
       firstTauFactor = varargin{i+1};
       firstTauFactorGiven = 1;
     case 'TRUE_X'
       compute_mse = 1;
       true = varargin{i+1};
     case 'ALPHAMIN'
       alphamin = varargin{i+1};
     case 'ALPHAMAX'
       alphamax = varargin{i+1};
     case 'AT'
       AT = varargin{i+1};
     case 'VERBOSE'
       verbose = varargin{i+1};
     otherwise
      % Hmmm, something wrong with the parameter string
      error(['Unrecognized option: ''' varargin{i} '''']);
    end;
  end;
end
%%%%%%%%%%%%%%

if (sum(stopCriterion == [0 1 2 3 4 5])==0)
  error(['Unknown stopping criterion']);
end

% if A is a function handle, we have to check presence of AT,
if isa(A, 'function_handle') & ~isa(AT,'function_handle')
  error(['The function handle for transpose of A is missing']);
end 

% if A is a matrix, we find out dimensions of y and x,
% and create function handles for multiplication by A and A',
% so that the code below doesn't have to distinguish between
% the handle/not-handle cases
if ~isa(A, 'function_handle')
  AT = @(x) A'*x;
  A = @(x) A*x;
end
% from this point down, A and AT are always function handles.

% Precompute A'*y since it'll be used a lot
Aty = AT(y);

% Initialization
switch init
    case 0   % initialize at zero, using AT to find the size of x
       x = AT(zeros(size(y)));
    case 1   % initialize randomly, using AT to find the size of x
       x = randn(size(AT(zeros(size(y)))));
    case 2   % initialize x0 = A'*y
       x = Aty; 
    case Initial_X_supplied
       % initial x was given as a function argument; just check size
       if size(A(x)) ~= size(y)
          error(['Size of initial x is not compatible with A']); 
       end
    otherwise
       error(['Unknown ''Initialization'' option']);
end

% now check if tau is an array; if it is, it has to 
% have the same size as x
if prod(size(tau)) > 1
   try,
      dummy = x.*tau;
   catch,
      error(['Parameter tau has wrong dimensions; it should be scalar or size(x)']),
   end
end
     

% if the true x was given, check its size
if compute_mse & (size(true) ~= size(x))  
   error(['Initial x has incompatible size']); 
end

% if tau is scalar, we check its value; if it's large enough,
% the optimal solution is the zero vector
if prod(size(tau)) == 1
   aux = AT(y);
   max_tau = max(abs(aux(:)));
   if tau >= max_tau
      x = zeros(size(aux));
      if debias
         x_debias = x;
      end
      objective(1) = 0.5*(y(:)'*y(:));
      times(1) = 0;
      if compute_mse
          mses(1) = sum(true(:).^2);
      end
      return
   end 
end

% initialize u and v
u =  x.*(x >= 0);
v = -x.*(x <  0);

% define the indicator vector or matrix of nonzeros in x
nz_x = (x ~= 0.0);
num_nz_x = sum(nz_x(:));

% start the clock
t0 = cputime;

% store given tau, because we're going to change it in the
% continuation procedure
final_tau = tau;

% store given stopping criterion and threshold, because we're going 
% to change them in the continuation procedure
final_stopCriterion = stopCriterion;
final_tolA = tolA;

% set continuation factors
if continuation&&(cont_steps > 1)
   % If tau is scalar, first check top see if the first factor is 
   % too large (i.e., large enough to make the first 
   % solution all zeros). If so, make it a little smaller than that.
   % Also set to that value as default
   if prod(size(tau)) == 1
      if (firstTauFactorGiven == 0)|(firstTauFactor*tau >= max_tau)
         firstTauFactor = 0.5*max_tau / tau;
         if verbose
             fprintf(1,'\n setting parameter FirstTauFactor\n')
         end
      end
   end
   cont_factors = 10.^[log10(firstTauFactor):...
                    log10(1/firstTauFactor)/(cont_steps-1):0];
end

if ~continuation
  cont_factors = 1;
  cont_steps = 1;
end
  
iter = 1;
if compute_mse
       mses(iter) = sum((x(:)-true(:)).^2);
end

keep_continuation = 1;
cont_loop = 1;
iter = 1;
taus = [];

% loop for continuation
while keep_continuation
    

    % Compute and store initial value of the objective function
    resid =  y - A(x);
    
    
    if cont_steps == -1
        gradq = AT(resid);
        tau = max(final_tau,0.2*max(abs(gradq)));
        if tau == final_tau
            stopCriterion = final_stopCriterion;
            tolA = final_tolA;
            keep_continuation = 0;
        else
            stopCriterion = 1;
            tolA = 1e-5;
        end
    else
        tau = final_tau * cont_factors(cont_loop);
        if cont_loop == cont_steps
            stopCriterion = final_stopCriterion;
            tolA = final_tolA;
            keep_continuation = 0;
        else
            stopCriterion = 1;
            tolA = 1e-5;
        end
    end
    
    taus = [taus tau];
    
    if verbose
        fprintf(1,'\nSetting tau = %0.5g\n',tau)
    end
    
    % if in first continuation iteration, compute and store
    % initial value of the objective function
    if cont_loop == 1
        alpha = 1.0;
        f = 0.5*(resid(:)'*resid(:)) + ...
             sum(tau(:).*u(:)) + sum(tau(:).*v(:));
        objective(1) = f;
        if compute_mse
            mses(1) = (x(:)-true(:))'*(x(:)-true(:));
        end
        if verbose
            fprintf(1,'Initial obj=%10.6e, alpha=%6.2e, nonzeros=%7d\n',...
                f,alpha,num_nz_x);
        end
    end

    % Compute the initial gradient and the useful 
    % quantity resid_base
    resid_base = y - resid;

    % control variable for the outer loop and iteration counter
    keep_going = 1;

    if verbose
      fprintf(1,'\nInitial obj=%10.6e, nonzeros=%7d\n',f,num_nz_x);
    end

    while keep_going

      % compute gradient
      temp = AT(resid_base);

      term  =  temp - Aty;
      gradu =  term + tau;
      gradv = -term + tau;

      % projection and computation of search direction vector
      du = max(u - alpha*gradu, 0.0) - u;
      dv = max(v - alpha*gradv, 0.0) - v;
      dx = du-dv;
      old_u = u; 
      old_v = v;

      % calculate useful matrix-vector product involving dx
      auv = A(dx);
      dGd = auv(:)'*auv(:);

      if (enforceMonotone==1)
        % monotone variant: calculate minimizer along the direction (du,dv)
        lambda0 = - (gradu(:)'*du(:) + gradv(:)'*dv(:))/(realmin+dGd);
        if lambda0 < 0
          fprintf(' ERROR: lambda0 = %10.3e negative. Quit\n', lambda0);
          return;
        end
        lambda = min(lambda0,1);
      else
        %nonmonotone variant: choose lambda=1
        lambda = 1;
      end

      u = old_u + lambda * du;
      v = old_v + lambda * dv;
      uvmin = min(u,v);
      u = u - uvmin; 
      v = v - uvmin; 
      x = u - v;
      
      % calculate nonzero pattern and number of nonzeros (do this *always*)
      nz_x_prev = nz_x;
      nz_x = (x~=0.0);
      num_nz_x = sum(nz_x(:));
      
      % update residual and function
      resid = y - resid_base - lambda*auv;
      prev_f = f;
      f = 0.5*(resid(:)'*resid(:)) +  sum(tau(:).*u(:)) + ...
          sum(tau(:).*v(:));

      % compute new alpha
      dd  = du(:)'*du(:) + dv(:)'*dv(:);  
      if dGd <= 0
        % something wrong if we get to here
        fprintf(1,' dGd=%12.4e, nonpositive curvature detected\n', dGd);
        alpha = alphamax;
      else
        alpha = min(alphamax,max(alphamin,dd/dGd));
      end
      resid_base = resid_base + lambda*auv; 
      
      % print out stuff
      if verbose
         fprintf(1,'It=%4d, obj=%9.5e, alpha=%6.2e, nz=%8d  ',...
             iter, f, alpha, num_nz_x);
      end
        prn = 2*(x-0.5);
        conv(:,iter) = prn;
        s = sprintf('[ %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f]', prn([ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]));
        if show
            disp(s);
        end

      % update iteration counts, store results and times
      iter = iter + 1;
      objective(iter) = f;
      times(iter) = cputime-t0;

      if compute_mse
        err = true - x;
        mses(iter) = (err(:)'*err(:));
      end

      switch stopCriterion
          case 0,
              % compute the stopping criterion based on the change
              % of the number of non-zero components of the estimate
              num_changes_active = (sum(nz_x(:)~=nz_x_prev(:)));
              if num_nz_x >= 1
                  criterionActiveSet = num_changes_active;
              else
                  criterionActiveSet = tolA / 2;
              end
              keep_going = (criterionActiveSet > tolA);
              if verbose
                  fprintf(1,'Delta n-zeros = %d (target = %e)\n',...
                      criterionActiveSet , tolA)
              end
          case 1,
              % compute the stopping criterion based on the relative
              % variation of the objective function.
              criterionObjective = abs(f-prev_f)/(prev_f);
              keep_going =  (criterionObjective > tolA);
              if verbose
                  fprintf(1,'Delta obj. = %e (target = %e)\n',...
                      criterionObjective , tolA)
              end
          case 2,
              % stopping criterion based on relative norm of step taken
              delta_x_criterion = norm(dx(:))/norm(x(:));
              keep_going = (delta_x_criterion > tolA);
              if verbose
                  fprintf(1,'Norm(delta x)/norm(x) = %e (target = %e)\n',...
                      delta_x_criterion,tolA)
              end
          case 3,
              % compute the "LCP" stopping criterion - again based on the previous
              % iterate. Make it "relative" to the norm of x.
              w = [ min(gradu(:), old_u(:)); min(gradv(:), old_v(:)) ];
              criterionLCP = norm(w(:), inf);
              criterionLCP = criterionLCP / ...
                  max([1.0e-6, norm(old_u(:),inf), norm(old_v(:),inf)]);
              keep_going = (criterionLCP > tolA);
              if verbose
                  fprintf(1,'LCP = %e (target = %e)\n',criterionLCP,tolA)
              end
          case 4,
              % continue if not yeat reached target value tolA
              keep_going = (f > tolA);
              if verbose
                  fprintf(1,'Objective = %e (target = %e)\n',f,tolA)
              end
          case 5,
            % stopping criterion based on relative norm of step taken
            delta_x_criterion = sqrt(dd)/sqrt(x(:)'*x(:));
            keep_going = (delta_x_criterion > tolA);
            if verbose
                fprintf(1,'Norm(delta x)/norm(x) = %e (target = %e)\n',...
                    delta_x_criterion,tolA)
            end
          otherwise,
              error(['Unknown stopping criterion']);
      end % end of the stopping criteria switch
      
      % take no less than miniter... 
      if iter<=miniter
	      keep_going = 1;
      elseif iter > maxiter %and no more than maxiter iterations  
	        keep_going = 0;
      end
      
    end % end of the main loop of the BB-QP algorithm

    % increment continuation loop counter
    cont_loop = cont_loop+1;
    
end % end of the continuation loop
% Print results

if verbose
   fprintf(1,'\nFinished the main algorithm!\nResults:\n')
   fprintf(1,'||A x - y ||_2^2 = %10.3e\n',resid(:)'*resid(:))
   fprintf(1,'||x||_1 = %10.3e\n',sum(abs(x(:))))
   fprintf(1,'Objective function = %10.3e\n',f);
   nz_x = (x~=0.0); num_nz_x = sum(nz_x(:));
   fprintf(1,'Number of non-zero components = %d\n',num_nz_x);
   fprintf(1,'CPU time so far = %10.3e\n', times(iter));
   fprintf(1,'\n');
end

% If the 'Debias' option is set to 1, we try to remove the bias from the l1
% penalty, by applying CG to the least-squares problem obtained by omitting
% the l1 term and fixing the zero coefficients at zero.

% do this only if the reduced linear least-squares problem is
% overdetermined, otherwise we are certainly applying CG to a problem with a
% singular Hessian

if (debias & (sum(x(:)~=0)~=0))
  
  if (num_nz_x > length(y(:)))
    if verbose
      fprintf(1,'\n')
      fprintf(1,'Debiasing requested, but not performed\n');
      fprintf(1,'There are too many nonzeros in x\n\n');
      fprintf(1,'nonzeros in x: %8d, length of y: %8d\n',...
	  num_nz_x, length(y(:)));
    end
  elseif (num_nz_x==0)
    if verbose
      fprintf(1,'\n')
      fprintf(1,'Debiasing requested, but not performed\n');
      fprintf(1,'x has no nonzeros\n\n');
    end
  else
    if verbose
      fprintf(1,'\n')
      fprintf(1,'Starting the debiasing phase...\n\n')
    end
    
    x_debias = x;
    zeroind = (x_debias~=0); 
    cont_debias_cg = 1;
    debias_start = iter;
    
    % calculate initial residual
    resid = A(x_debias);
    resid = resid-y;
    resid_prev = eps*ones(size(resid));
    
    rvec = AT(resid);
    
    % mask out the zeros
    rvec = rvec .* zeroind;
    rTr_cg = rvec(:)'*rvec(:);
    
    % set convergence threshold for the residual || RW x_debias - y ||_2
    tol_debias = tolD * (rvec(:)'*rvec(:));
    
    % initialize pvec
    pvec = -rvec;
    
    % main loop
    while cont_debias_cg
      
      % calculate A*p = Wt * Rt * R * W * pvec
      RWpvec = A(pvec);      
      Apvec = AT(RWpvec);
      
      % mask out the zero terms
      Apvec = Apvec .* zeroind;
      
      % calculate alpha for CG
      alpha_cg = rTr_cg / (pvec(:)'* Apvec(:));
      
      % take the step
      x_debias = x_debias + alpha_cg * pvec;
      resid = resid + alpha_cg * RWpvec;
      rvec  = rvec  + alpha_cg * Apvec;
      
      rTr_cg_plus = rvec(:)'*rvec(:);
      beta_cg = rTr_cg_plus / rTr_cg;
      pvec = -rvec + beta_cg * pvec;
      
      rTr_cg = rTr_cg_plus;
      
      iter = iter+1;
      
      objective(iter) = 0.5*(resid(:)'*resid(:)) + ...
	  sum(tau(:).*abs(x_debias(:)));
      times(iter) = cputime - t0;
      
      if compute_mse
	err = true - x_debias;
	mses(iter) = (err(:)'*err(:));
      end
      
      % in the debiasing CG phase, always use convergence criterion
      % based on the residual (this is standard for CG)
      if verbose
	fprintf(1,' Iter = %5d, debias resid = %13.8e, convergence = %8.3e\n', ...
	    iter, resid(:)'*resid(:), rTr_cg / tol_debias);
      end
      cont_debias_cg = ...
	  (iter-debias_start <= miniter_debias )| ...
	  ((rTr_cg > tol_debias) & ...
	  (iter-debias_start <= maxiter_debias));
      
    end
    if verbose
      fprintf(1,'\nFinished the debiasing phase!\nResults:\n')
      fprintf(1,'||A x - y ||_2^2 = %10.3e\n',resid(:)'*resid(:))
      fprintf(1,'||x||_1 = %10.3e\n',sum(abs(x(:))))
      fprintf(1,'Objective function = %10.3e\n',f);
      nz = (x_debias~=0.0);
      fprintf(1,'Number of non-zero components = %d\n',sum(nz(:)));
      fprintf(1,'CPU time so far = %10.3e\n', times(iter));
      fprintf(1,'\n');
    end
  end
  
  if compute_mse
    mses = mses/length(true(:));
  end
  
end
it = iter-1;
convergence = conv(:,1:it);
  


