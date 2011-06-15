

%This is an implementation of the Gaussian BP algorithm (sparse sync version)
%Written by Danny Bickson
%See: http://books.nips.cc/papers/files/nips18/NIPS2005_0210.pdf
%Equations 7,8,9
%Input: A - sparse information matrix mxm, (assumed to be symmetric) 
%b - shift vector 1xm
%maxround - max number of iterations
%epsilon - convergence detection
%retn - cell array of neighbors
%warm - 

%Output: The solution for the inference problem
%        vector h of size 1xm s.t. h = max(1/2h'Ah +h'b)
%        J - vector of the values Pii (approximation of the diagonal of the matrix A^-1)
%        r - iteration count

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
function [h,J,r] = sparse_gabp(A,b,maxround,epsilon)

m=length(A);
retn = cell(m,1);
C=sparse(maxround,length(b));

%learn matrix topology only once
 for i=1:length(A)
    neighbors = find (A(i,:) ~= 0);
    neighbors(neighbors == i) = [];
    retn{i} = neighbors;
 end

%return values
h=zeros(1,m);
J=zeros(1,m);

%messages arrays
Mh = sparse(m,m);
MJ = sparse(m,m);

conv = false;

% algorithm rounds
for r=1:maxround
%    disp(['starting sparse async GBP round ', num2str(r)]); 
       
	for i=1:m
		% sum up all mean and percision values got from neighbors
		h(i) = b(i) + sum(Mh(:,i));  %(7)
        %variance can not be zero 
        %assert(A(i,i) ~= 0);
        J(i) = A(i,i) + sum(MJ(:,i));

        val = sparse(1,m);
        val(retn{i}) = (-A(retn{i}, i) ./ (J(i) - MJ(retn{i},i)));
        Mh(i,retn{i}) = val(retn{i}) .* (h(i) - Mh(retn{i},i))';
        MJ(i,retn{i}) = val(retn{i}) .* A(i, retn{i});
    end

    
     for i=1:m
         h(i) = b(i) + sum(Mh(:,i));  %(9)
         J(i) = A(i,i) + sum(MJ(:,i));
     end
     J=1./J;
     h=h.*J;
    
     C(r,:)=h;
    if (r>2)
    disp(['norm is ', num2str(  (norm(C(r,:) - C(r-1,:))/norm(C(r,:))))]);
    end

    h(1:100)    
    if (r > 2 && ((norm(C(r,:) - C(r-1,:))/norm(C(r,:))) < epsilon))
        disp(['Async GBP (MJ) Converged afbter ', num2str(r), ' rounds ']); 
        
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


J = 1./J;
h=h.*J;
end




