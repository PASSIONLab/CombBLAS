%This is an implementation of the Gaussian BP algorithm (sparse sync version)
%Written by Danny Bickson
%See: http://books.nips.cc/papers/files/nips18/NIPS2005_0210.pdf
%Equations 7,8,9
%Input: A - sparse information matrix mxm, (assumed to be symmetric) 
%b - shift vector 1xm
%Output: The solution for the inference problem
%        vector h of size 1xm s.t. h = max(1/2h'Ah +h'b)
%        J - vector of the values Pii (approximation of the diagonal of the matrix A^-1)
function [h,J,r,C] = sparse_GBP(A,b,maxround,epsilon,retn,warm)
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
    disp(['starting sparse async GBP round ', num2str(r)]); 
       
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
