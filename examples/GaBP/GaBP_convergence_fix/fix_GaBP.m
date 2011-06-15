

% Function that demonstrates how to fix the convergence of the GaBP
% algorithm. Given a system of linear equation Ax = b, the algorithm
% solves x = inv(A)*b iteratively, even when the sufficient convergence
% conditions of gaussian BP (or other iterative algorithms like Jacobi, Gauss-Seidel etc. ) 
% do not hold.
% Input: A - symmetric matrix of size nxn
%        b - vector of size nx1
%        max_iter - maximal number of iterations 
%        epsilon - convergence threshold
% Output:
%        x - vector of size nx1, x = inv(A)*b
% Algorithm is described in the paper:
% "Fixing convergence of GaBP algorithm" by
% J. K. Johnson, D. Bickson and D. Dolev
% In ISIT 2009
% http://arxiv.org/abs/0901.4192
% Code written by Danny Bickson
% December 2008.
% Code updated January 2010.
function [x, total_rounds] = fix_GaBP(A,b,max_iter,epsilon,gamma)

%assert(sum(sum(A~=A')) == 0);
if ~exist('gamma','var')
    gamma = 1;
end

dloading = max(sum(abs(A)) - max(diag(A)));
Minc = A + eye(length(A)) *  gamma*dloading;

disp(['diagonal loading is ' num2str(dloading*gamma)]);
if (dloading == 0)
   warning('No need to use double loop construction, since GaBP converges anyway. aborting');
   return;
end

total_rounds=0;
xj = b;
old_xj = zeros(length(A),1);

for l=2:max_iter
    % This is a single Newton step
    % Algorithm is described in
    % Linear Detection via Belief Propagation. Danny Bickson, Danny Dolev, Ori
    % Shental, Paul H. Siegel and Jack K. Wolf. In the 45th Annual Allerton Conference on Communication, Control, 
    % and Computing, Allerton House, Illinois, Sept. 07.
   
   
   [direc,J,r1] = asynch_GBP(Minc, b - A*xj, max_iter, epsilon);
   total_rounds = total_rounds+r1;
   xj=xj+direc';

   if (norm(old_xj - xj)<epsilon)
      disp(['Newton iteration converged to accuracy ', num2str(norm(old_xj - xj))]); 
      break;
   end
   
   old_xj = xj;
end

       
x = xj;


end
