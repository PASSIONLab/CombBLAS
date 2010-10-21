function B = sprefm(A,I,J)
% SPREFM : Sparse matrix indexing using sparse matrix multiplication
%
% B = sprefm(A,I,J) has the effect of B = A(I,J) for integer vectors I, J
%
% John R. Gilbert, September 3, 2010

if min(size(I))>1 || min(size(J))>1 || any(round(I)~=I) || any(round(J)~=J)
    error('indices must be vectors of integers');
end;
I = I(:);
J = J(:);
ni = size(I,1);
nj = size(J,1);
[nra,nca] = size(A);
if min(I)<1 || max(I)>nra || min(J)<1 || max(J)>nca
    error('index out of range');
end;
P = sparse(1:ni,I,1,ni,nra);
Q = sparse(J,1:nj,1,nca,nj);
B = P*(A*Q);