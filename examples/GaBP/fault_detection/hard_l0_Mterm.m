function [s, it, convergence, err_mse, iter_time]=hard_l0_Mterm(x,A,m,M,show,varargin)
% hard_l0_Mterm: Hard thresholding algorithm that keeps exactly M elements 
% in each iteration. 
%
% This algorithm has certain performance guarantees as described in [1],
% [2] and [3].
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Usage
%
%   [s, err_mse, iter_time]=hard_l0_Mterm(x,P,m,M,'option_name','option_value')
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Input
%
%   Mandatory:
%               x   Observation vector to be decomposed
%               P   Either:
%                       1) An nxm matrix (n must be dimension of x)
%                       2) A function handle (type "help function_format" 
%                          for more information)
%                          Also requires specification of P_trans option.
%                       3) An object handle (type "help object_format" for 
%                          more information)
%               m   length of s 
%               M   non-zero elements to keep in each iteration
%
%   Possible additional options:
%   (specify as many as you want using 'option_name','option_value' pairs)
%   See below for explanation of options:
%__________________________________________________________________________
%   option_name    |     available option_values                | default
%--------------------------------------------------------------------------
%   stopTol        | number (see below)                         | 1e-16
%   P_trans        | function_handle (see below)                | 
%   maxIter        | positive integer (see below)               | n^2
%   verbose        | true, false                                | false
%   start_val      | vector of length m                         | zeros
%   step_size      | number                                     | 0 (auto)
%
%   stopping criteria used : (OldRMS-NewRMS)/RMS(x) < stopTol
%
%   stopTol: Value for stopping criterion.
%
%   P_trans: If P is a function handle, then P_trans has to be specified and 
%            must be a function handle. 
%
%   maxIter: Maximum number of allowed iterations.
%
%   verbose: Logical value to allow algorithm progress to be displayed.
%
%   start_val: Allows algorithms to start from partial solution.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Outputs
%
%    s              Solution vector 
%    err_mse        Vector containing mse of approximation error for each 
%                   iteration
%    iter_time      Vector containing computation times for each iteration
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Description
%
%   Implements the M-sparse algorithm described in [1], [2] and [3].
%   This algorithm takes a gradient step and then thresholds to only retain
%   M non-zero elements. It allows the step-size to be calculated
%   automatically as described in [3] and is therefore now independent from 
%   a rescaling of P.
%   
%   
% References
%   [1]  T. Blumensath and M.E. Davies, "Iterative Thresholding for Sparse 
%        Approximations", submitted, 2007
%   [2]  T. Blumensath and M. Davies; "Iterative Hard Thresholding for 
%        Compressed Sensing" to appear Applied and Computational Harmonic 
%        Analysis 
%   [3] T. Blumensath and M. Davies; "A modified Iterative Hard 
%        Thresholding algorithm with guaranteed performance and stability" 
%        in preparation (title may change) 
% See Also
%   hard_l0_reg
%
% Copyright (c) 2007 Thomas Blumensath
%
% The University of Edinburgh
% Email: thomas.blumensath@ed.ac.uk
% Comments and bug reports welcome
%
% This file is part of sparsity Version 0.4
% Created: April 2007
% Modified January 2009
%
% Part of this toolbox was developed with the support of EPSRC Grant
% D000246/1
%
% Please read COPYRIGHT.m for terms and conditions.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    Default values and initialisation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



[n1 n2]=size(x);
if n2 == 1
    n=n1;
elseif n1 == 1
    x=x';
    n=n2;
else
   error('x must be a vector.');
end
    
sigsize     = x'*x/n;
oldERR      = sigsize;
err_mse     = [];
iter_time   = [];
STOPTOL     = 0.01;%1e-16;
MAXITER     = n^2;
verbose     = false;
initial_given=0;
s_initial   = zeros(m,1);
MU          = 0;

if verbose
   display('Initialising...') 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Output variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch nargout 
    case 5
        comp_err=true;
        comp_time=true;
    case 4 
        comp_err=true;
        comp_time=false;
    case 3
        comp_err=false;
        comp_time=false;      
    otherwise
        error('Wrong number of output arguments specified')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Look through options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Put option into nice format
Options={};
OS=nargin-5;
c=1;
for i=1:OS
    if isa(varargin{i},'cell')
        CellSize=length(varargin{i});
        ThisCell=varargin{i};
        for j=1:CellSize
            Options{c}=ThisCell{j};
            c=c+1;
        end
    else
        Options{c}=varargin{i};
        c=c+1;
    end
end
OS=length(Options);
if rem(OS,2)
   error('Something is wrong with argument name and argument value pairs.') 
end
for i=1:2:OS
   switch Options{i}
        case {'stopTol'}
            if isa(Options{i+1},'numeric') ; STOPTOL     = Options{i+1};   
            else error('stopTol must be number. Exiting.'); end
        case {'P_trans'} 
            if isa(Options{i+1},'function_handle'); Pt = Options{i+1};   
            else error('P_trans must be function _handle. Exiting.'); end
        case {'maxIter'}
            if isa(Options{i+1},'numeric'); MAXITER     = Options{i+1};             
            else error('maxIter must be a number. Exiting.'); end
        case {'verbose'}
            if isa(Options{i+1},'logical'); verbose     = Options{i+1};   
            else error('verbose must be a logical. Exiting.'); end 
        case {'start_val'}
            if isa(Options{i+1},'numeric') && length(Options{i+1}) == m ;
                s_initial     = Options{i+1};  
                initial_given=1;
            else error('start_val must be a vector of length m. Exiting.'); end
        case {'step_size'}
            if isa(Options{i+1},'numeric') && (Options{i+1}) > 0 ;
                MU     = Options{i+1};   
            else error('Stepsize must be between a positive number. Exiting.'); end
        otherwise
            error('Unrecognised option. Exiting.') 
   end
end

if nargout >=2
    err_mse = zeros(MAXITER,1);
end
if nargout ==3
    iter_time = zeros(MAXITER,1);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Make P and Pt functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if          isa(A,'float')      P =@(z) A*z;  Pt =@(z) A'*z;
elseif      isobject(A)         P =@(z) A*z;  Pt =@(z) A'*z;
elseif      isa(A,'function_handle') 
    try
        if          isa(Pt,'function_handle'); P=A;
        else        error('If P is a function handle, Pt also needs to be a function handle. Exiting.'); end
    catch error('If P is a function handle, Pt needs to be specified. Exiting.'); end
else        error('P is of unsupported type. Use matrix, function_handle or object. Exiting.'); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Do we start from zero or not?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



if initial_given ==1;
    
    if length(find(s_initial)) > M
        display('Initial vector has more than M non-zero elements. Keeping only M largest.')
    
    end
    s                   =   s_initial;
    [ssort sortind]     =   sort(abs(s),'descend');
    s(sortind(M+1:end)) =   0;
    Ps                  =   P(s);
    Residual            =   x-Ps;
    oldERR      = Residual'*Residual/n;
else
    s_initial   = zeros(m,1);
    Residual    = x;
    s           = s_initial;
    Ps          = zeros(n,1);
    oldERR      = sigsize;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 Random Check to see if dictionary norm is below 1 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        x_test=randn(m,1);
        x_test=x_test/norm(x_test);
        nP=norm(P(x_test));
        if abs(MU*nP)>1;
            display('WARNING! Algorithm likely to become unstable.')
            display('Use smaller step-size or || P ||_2 < 1.')
        end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Main algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if verbose
   display('Main iterations...') 
end
tic
t=0;
done = 0;
%%%%my additions
iter=1;
conv = zeros(m,50);

while ~done
    
    if MU == 0

        %Calculate optimal step size and do line search
        olds                =   s;
        oldPs               =   Ps;
        IND                 =   s~=0;
        d                   =   Pt(Residual);
        % If the current vector is zero, we take the largest elements in d
        if sum(IND)==0
            [dsort sortdind]    =   sort(abs(d),'descend');
            IND(sortdind(1:M))  =   1;    
         end  

        id                  =   (IND.*d);
        Pd                  =   P(id);
        mu                  =   id'*id/(Pd'*Pd);
        s                   =   olds + mu * d;
        [ssort sortind]     =   sort(abs(s),'descend');
        s(sortind(M+1:end)) =   0;
        Ps                  =   P(s);
        
        % Calculate step-size requirement 
        omega               =   (norm(s-olds)/norm(Ps-oldPs))^2;

        % As long as the support changes and mu > omega, we decrease mu
        while mu > (0.99)*omega && sum(xor(IND,s~=0))~=0 && sum(IND)~=0
%             display(['decreasing mu'])
                    
                    % We use a simple line search, halving mu in each step
                    mu                  =   mu/2;
                    s                   =   olds + mu * d;
                    [ssort sortind]     =   sort(abs(s),'descend');
                    s(sortind(M+1:end)) =   0;
                    Ps                  =   P(s);
                    % Calculate step-size requirement 
                    omega               =   (norm(s-olds)/norm(Ps-oldPs))^2;
        end
        
    else
        % Use fixed step size
        s                   =   s + MU * Pt(Residual);
        [ssort sortind]     =   sort(abs(s),'descend');
        s(sortind(M+1:end)) =   0;
        Ps                  =   P(s);
        
    end
        Residual            =   x-Ps;
    prn = 2*(s-0.5);
    conv(:,iter) = prn;
    str = sprintf('[ %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f]', prn([ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]));
    if show
        disp(str);
    end

        
     ERR=Residual'*Residual/n;
     if comp_err
         err_mse(iter)=ERR;
     end
     
     if comp_time
         iter_time(iter)=toc;
     end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Are we done yet?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
         if comp_err && iter >=2
             if ((err_mse(iter-1)-err_mse(iter))/sigsize<STOPTOL);
                 if verbose
                    display(['Stopping. Approximation error changed less than ' num2str(STOPTOL)])
                 end
                done = 1; 
             elseif verbose && toc-t>10
                display(sprintf('Iteration %i. --- %i mse change',iter ,(err_mse(iter-1)-err_mse(iter))/sigsize)) 
                t=toc;
             end
         else
             if ((oldERR - ERR)/sigsize < STOPTOL) && iter >=2;
                 if verbose
                    display(['Stopping. Approximation error changed less than ' num2str(STOPTOL)])
                 end
                done = 1; 
             elseif verbose && toc-t>10
                display(sprintf('Iteration %i. --- %i mse change',iter ,(oldERR - ERR)/sigsize)) 
                t=toc;
             end
         end
         
    % Also stop if residual gets too small or maxIter reached
     if comp_err
         if err_mse(iter)<1e-16
             display('Stopping. Exact signal representation found!')
             done=1;
         end
     elseif iter>1 
         if ERR<1e-16
             display('Stopping. Exact signal representation found!')
             done=1;
         end
     end

     if iter >= MAXITER
         display('Stopping. Maximum number of iterations reached!')
         done = 1; 
     end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    If not done, take another round
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
     if ~done
        iter=iter+1; 
        oldERR=ERR;        
     end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  Only return as many elements as iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
it = iter;
convergence = conv(:,1:it);
if nargout >=2
    err_mse = err_mse(1:iter);
end
if nargout ==3
    iter_time = iter_time(1:iter);
end
if verbose
   display('Done') 
end

