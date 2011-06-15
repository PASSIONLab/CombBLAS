%This is an implementation of the Block (vectoric) Gaussian BP algorithm
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
function [h,J,r,C] = blockGBP(A,b,maxround,epsilon,d)
assert(d>=1);

m=length(A);
Mh=zeros(m,m);
MJ=zeros(m,m);
h=zeros(1,m);
J=zeros(m,m);

conv = false;
C=zeros(maxround,m);
% algorithm rounds
for r=1:maxround
    disp(['starting Block GBP round ', num2str(r)]); 
    old_MJ = MJ;
    old_Mh = Mh;
    MJ = zeros(m,m);
    Mh = zeros(m,m);
	% for each node
   for i=1:m/d
       
        srow = d*(i-1)+1;
        erow = d*i;
        irange=srow:erow;
		% sum up all mean and percision values got from neighbors
		h(irange) = b(irange) + sum(old_Mh(:,irange));  %(7)
        
        assert(sum(diag(A(irange,irange))) ~= 0);
        J(irange,irange)=A(irange,irange);
        % for each neighbor
         for j=1:m/d
              jsrow = d*(j-1)+1;
              jerow = d*j;
              jrange = jsrow:jerow;
              J(irange,irange) = J(irange,irange) + old_MJ(jrange,irange);
        end
		% send message to all neighbors
	for j=1:m/d
        jsrow = d*(j-1)+1;
        jerow = d*j;
        jrange = jsrow:jerow;
        
			if (i ~= j && sum(sum(A(irange,jrange))) ~= 0)
				h_j = h(irange) - sum(old_Mh(jrange,irange));
				J_j = J(irange,irange) - old_MJ(jrange,irange);
				%disp([num2str(i), '.', num2str(j), 'h_i', num2str(h_j), 'j_j', num2str(J_j)]);
 %               assert(A(i,j) == A(j,i));
			    assert(trace(J_j) ~= 0);
                Mh(irange,jrange) = diag((-A(jrange,irange) / J_j)* h_j'); %(8)
				MJ(irange,jrange) = (-A(jrange,irange) / J_j) * A(irange,jrange);
            end
    end
		%disp(['h is : ', num2str(h-b)]);
		%disp(['J is : ', num2str(J)]);
			
        
        %h
   end
    max_diff  = epsilon;
   
    %Mh
    %MJ
	%sum(Mh)
	%sum(Mh')
    
    J=inv(J);
    h=(J*h')';
    C(r,:)=h;
    disp(['residual norm ',num2str(norm(A*h'-b'))]);
    %diffMj = MJ - old_MJ;
    %if (max(abs(diffMj)) < max_diff)
   if (r > 2 && ((norm(C(r,:) - C(r-1,:))/norm(C(r,:))) < epsilon))
        disp(['Block GBP (MJ) Converged afeter ', num2str(r), ' rounds ']); 
        conv = true;
		break;
   end
end
if (conv == false)
	disp(['Block GBP (MJ) Did not converge in ', num2str(r), ' rounds ']);
end
%assert(J ~= 0);
%J = 1./J;
%h=h.*J;
disp(['GBP result h is: ', num2str(h)]);
disp(['GBP result J is: ', num2str(diag(J)')]);
