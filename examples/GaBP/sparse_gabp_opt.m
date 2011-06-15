%This is an implementation of the sparse optimized (scalar) Gaussian BP algorithm
% Written by Danny Bickson, CMU, HUJI and IBM Haifa Labs.
%
% Please report bugs to: danny.bickson@gmail.com
%
% See algorithm description in:
% Gaussian Belief Propagation: Theory and Application. D. Bickson. Ph.D. Thesis. Submitted to the senate of the Hebrew University of Jerusalem, October 2008. Revised July 2009.  
% available online on: http://arxiv.org/abs/0811.2518 
%
% Input: A - information matrix mxm, (assumed to be symmetric) 
% b - shift vector 1xm
% max_round - max iteration
% epsilon - convergence threshold
% d - required block size
%
% Output: The solution for the inference problem
%        vector h of size 1xm s.t. h = max(exp(-1/2x'Ax +x'b)). Equivalently, h = inv(A) * b;
%        J - vector of the precision values Pii (the diagonal of the matrix A^-1)
%        r - round number
%

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
function [h,J,r,C] = sparse_gabp_opt(A,b,maxround,epsilon,retn,warm)
m=length(A);
C=sparse(maxround,length(b));
%return values
h=zeros(1,m);
J=zeros(1,m);
if (~warm)
    Mh = sparse(m,m);
    MJ = sparse(m,m);
else % warm start
    load data;
end
conv = false;
% algorithm rounds
for r=1:maxround
    %disp(['starting sparse async GBP round ', num2str(r)]); 
       
	for i=1:m
		% sum up all mean and percision values got from neighbors
		h(i) = b(i) + sum(Mh(:,i));  %(7)
	    
        %variance can not be zero (must be a diagonally dominant matrix)!
        %assert(A(i,i) ~= 0);
        J(i) = A(i,i) + sum(MJ(:,i));
	
        %val = sparse(1,m);
        neighbors = retn{i};
        val(neighbors) = (-A(neighbors, i) ./ (J(i) - MJ(neighbors,i)));
        Mh(i,neighbors) =  val(neighbors) .* (h(i) - Mh(neighbors,i))';
        MJ(i,neighbors) = val(neighbors) .* A(i, neighbors);
       
           
        %end
        
        %h
    end
    
     for i=1:m
         h(i) = b(i) + sum(Mh(:,i));  %(9)
         J(i) = A(i,i) + sum(MJ(:,i));
     end
     J=1./J;
     h=h.*J;
    
     C(r,:)=h;
    
     %diffMj = MJ - old_MJ;
    %if (max(abs(diffMj)) < max_diff)
    if (r > 2 && ((norm(C(r,:) - C(r-1,:))/norm(C(r,:))) < epsilon))
        disp(['Async GBP (MJ) Converged afeter ', num2str(r), ' rounds ']); 
        
        for i=1:m
            h(i) = b(i) + sum(Mh(:,i));  %(9)
            J(i) = A(i,i) + sum(MJ(:,i));
        end
       conv = true;
        break;
    end
end
if (conv == false)
    disp(['GBP (MJ) did not converge in ', num2str(r), ' rounds ']); 
    for i=1:m
        h(i) = b(i) + sum(Mh(:,i));  %(9)
        J(i) = A(i,i) + sum(MJ(:,i));
    end
end
%if (~warm)
    save data Mh MJ;
%end
J = 1./J;
h=h.*J;
end
