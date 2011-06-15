%This program is free software: you can redistribute it and/or modify
%it under the terms of the GNU General Public License as published by
%the Free Software Foundation, either version 3 of the License, or
%(at your option) any later version.

%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.

%You should have received a copy of the GNU General Public License
%along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Implementation of the Gaussian BP algorithm, as given in: 
% Linear Detection via Belief Propagation
% By Danny Bickson, Danny Dolev, Ori Shental, Paul H. Siegel and Jack K. Wolf.
% In the 45th Annual Allerton Conference on Communication, Control and Computing, Allerton House, Illinois, 
% Sept. 07'
%
%
% Written by Danny Bickson.
% updated: 24-Aug-2008
%
% input: A - square matrix nxn
% b - vector nx1
% max_iter - maximal number of iterations
% epsilon - convergence threshold
% output: x - vector of size nx1, which is the solution to linear systems of equations A x = b
%         Pf - vector of size nx1, which is an approximation to the main
%         diagonal of inv(A)
%         r - number of rounds until convergence
function [x,Pf,l] = gabp(A, b, max_iter, epsilon)

% Stage 1 - initilize
P = diag(diag(A));
U = diag(b./diag(A));
n = length(A);

% Stage 2 - iterate
for l=1:max_iter
  % record last round messages for convergence detection
     old_U = U; 

     for i=1:n
       for j=1:n
         % Compute P i\j - line 2
         if (i~=j && A(i,j) ~= 0)
             p_i_minus_j = sum(P(:,i)) - P(j,i); % 
             assert(p_i_minus_j ~= 0);
             %iterate - line 3
             P(i,j) = -A(i,j) * A(j,i) / p_i_minus_j;
             % Compute U i\j - line 2
             h_i_minus_j = (sum(P(:,i).*U(:,i)) - P(j,i)*U(j,i)) / p_i_minus_j;
             %iterate - line 3
             U(i,j) = - A(i,j) * h_i_minus_j / P(i,j);
         end
       end
 end

 

 % Stage 3 - convergence detection
 if (sum(sum((U - old_U).^2)) < epsilon)
     disp(['GABP converged in round ', num2str(l)]);
     break;
 end

end % iterate

% Stage 4 - infer
Pf = zeros(1,n);
x = zeros(1,n);
for i = 1:n
   Pf(i) = sum(P(:,i)); 
   x(i) = sum(U(:,i).*P(:,i))./Pf(i);
end


end
