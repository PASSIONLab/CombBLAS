%This is an implementation of the Gaussian BP algorithm
% Written by Danny Bickson
%See: http://books.nips.cc/papers/files/nips18/NIPS2005_0210.pdf
%Equations 7,8,9
%Input: A - information matrix mxm, (assumed to be symmetric) and
%diagonally dominant.
%B - shift vector 1xm
%C - constraint that h <= C, for all i=1:m
%Output: The solution for the inference problem
%        vector h of size 1xm s.t. h = max(1/2h'Ah +h'b)
%        J - vector of the values Pii (the diagonal of the matrix A^-1)
function [h,J,r,C] = blockGBP2(A,b,maxround,epsilon,d)
%format long e;
%assert(length(A) == length(b));
m=length(A);
%messages
Mh=zeros(m,m);
MJ=zeros(m,m);
%return values
h=zeros(1,m);
J=zeros(m,m);
%
conv = false;
C=zeros(maxround,m);
% algorithm rounds
for r=1:maxround
    disp(['starting GBP round ', num2str(r)]); 
    %old_MJ = MJ;
    %old_Mh = Mh;
    %MJ = zeros(m,m);
    %Mh = zeros(m,m);
	% for each node
   for i=1:m/d
       
        srow = d*(i-1)+1;
        erow = d*i;
        irange=srow:erow;
		% sum up all mean and percision values got from neighbors
		h(irange) = b(irange) + sum(Mh(:,irange));  %(7)
        
		%variance can not be zero (must be a diagonally dominant matrix)!
        assert(sum(diag(A(irange,irange))) ~= 0);
        J(irange,irange)=A(irange,irange);
        for j=1:m/d
              jsrow = d*(j-1)+1;
              jerow = d*j;
              jrange = jsrow:jerow;
              J(irange,irange) = J(irange,irange) + MJ(jrange,irange);
        end
		% send message to all neighbors
	for j=1:m/d
        jsrow = d*(j-1)+1;
        jerow = d*j;
        jrange = jsrow:jerow;
        
			if (i ~= j && sum(sum(A(irange,jrange))) ~= 0)
				h_j = h(irange) - sum(Mh(jrange,irange));
				J_j = J(irange,irange) - MJ(jrange,irange);
				%disp([num2str(i), '.', num2str(j), 'h_i', num2str(h_j), 'j_j', num2str(J_j)]);
 %               assert(A(i,j) == A(j,i));
			    assert(trace(J_j) ~= 0);
                Mh(irange,jrange) = diag((-A(jrange,irange) / J_j)* h_j'); %(8)
				MJ(irange,jrange) = (-A(jrange,irange) / J_j) * A(irange,jrange);
            end
    end
   end

    
    J=inv(J);
    h=(J*h')';
    C(r,:)=h;
    res = norm(A*h'-b');
    disp(['residual norm ',num2str(res)]);
    if (res < epsilon)
        break;
    end
 
   %if (r > 2 && ((norm(C(r,:) - C(r-1,:))/norm(C(r,:))) < epsilon))
   %     disp(['GBP (MJ) Converged afeter ', num2str(r), ' rounds ']); 
   %     conv = true;
   %		break;
   %   end
end
if (conv == false)
	disp(['GBP (MJ) Did not converge in ', num2str(r), ' rounds ']);
end
disp(['GBP result h is: ', num2str(h)]);
disp(['GBP result J is: ', num2str(diag(J)')]);
