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
                                                                     
                                                                     
                                             
% Implementation of the Min-sum ussian BP algorithm, as given in: 
% C.C. Moallemi and B. Van Roy. “Convergence of the min-sum algorithm for convex optimization.” Posted: May 
%2007.
%
%
% Written by Danny Bickson.
% updated: 4-Auguest-2008
%
% input: A - square matrix nxn
% b - vector nx1
% max_iter - maximal number of iterations
% epsilon - convergence threshold
% output: x - vector of size nx1, which is the solution to linear systems of equations A x = b
%         Pf - vector of size nx1, which is an approximation to the main
%         diagonal of inv(A)



function [x]=gabpms(A,b,rounds,epsilon)
    assert(size(A,1) == size(A,2));
    
    gamma = zeros(size(A,1),size(A,1));
    z = zeros(size(A,1),size(A,1));
    old_gamma = gamma;
    old_z = z;
    
    x=zeros(1,size(A,1));
    
    for k=1:rounds
       for i=1:size(A,1)
           for j=1:size(A,2)
               if (i ~= j && A(i,j) ~= 0)
                    gamma(i,j) = 1/(1-sum(old_gamma(:,i).*(A(:,i).^2)) + old_gamma(j,i)*A(j,i)^2);
                    assert(gamma(i,j) ~=0);
                    z(i,j) = A(i,j)*gamma(i,j)*(b(i) - sum(old_z(:,i)) + old_z(j,i));
               end
           end
        end
        gamma
        z
        
         % Stage 3 - convergence detection
         if (sum(sum((z - old_z).^2)) < epsilon)
             disp(['Min-Sum GABP converged in round ', num2str(k)]);
             break;
         end
        old_gamma = gamma;
        old_z = z;
        
           
    end
    
    
    for i=1:size(A,1)
        x(i) = 1/(1-sum((A(:,i).^2).*gamma(:,i)))*(b(i) - sum(z(:,i)));
    end


end
