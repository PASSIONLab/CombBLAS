%This is an implementation of the Block (scalar) Gaussian BP algorithm
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


function [h,J,r,C] = GBP(A,b,maxround,epsilon)
m=length(A);
%messages
Mh=zeros(m,m);
MJ=zeros(m,m);
%return values
h=zeros(1,m);
J=zeros(1,m);
%
conv = false;
C=zeros(maxround,m);
% algorithm rounds
for r=1:maxround
    disp(['starting GBP round ', num2str(r)]); 
    old_MJ = MJ;
    old_Mh = Mh;
    % for each node
    for i=1:m
	% sum up all mean and percision values got from neighbors
	h(i) = b(i) + sum(old_Mh(:,i));  %(7)
        
	%variance can not be zero (must be a diagonally dominant matrix)!
        assert(A(i,i) ~= 0);
        J(i) = A(i,i) + sum(old_MJ(:,i));
	% send message to all neighbors
	for j=1:m
	   if (i ~= j && A(i,j) ~= 0)
		h_j = h(i) - old_Mh(j,i);
		J_j = J(i) - old_MJ(j,i);
		assert(J_j ~= 0);
                
                Mh(i,j) = (-A(j,i) / J_j)* h_j; %(8)
		MJ(i,j) = (-A(j,i) / J_j) * A(i,j);
            end
       end
     end
    max_diff  = epsilon;
   
    for i=1:m
        h(i) = b(i) + sum(Mh(:,i));  %(9)
        J(i) = A(i,i) + sum(MJ(:,i));
    end
    J=1./J;
    h=h.*J;
  
    C(r,:)=h;
    disp(['residual norm ',num2str(norm(A*(h./J)'-b))]);
    if (r > 2 && ((norm(C(r,:) - C(r-1,:))/norm(C(r,:))) < epsilon))
        disp(['GBP (MJ) Converged afeter ', num2str(r), ' rounds ']); 
        conv = true;
		break;
   end
end
if (conv == false)
	disp(['GBP (MJ) Did not converge in ', num2str(r), ' rounds ']);
end

disp(['GBP result h is: ', num2str(h)]);
disp(['GBP result J is: ', num2str(J)]);
