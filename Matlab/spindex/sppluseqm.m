function C = sppluseqm(A,I,J,B)
% SPPLUSEQM : Sparse matrix increment using sparse matrix multiplication
%
% For integer vectors I and J,
% A = sppluseqm(A,I,J,B) has the effect of A(I,J) += B
% C = sppluseqm(A,I,J,B) has the effect of C = A; C(I,J) = C(I,J) + B
%
% John R. Gilbert, September 6, 2010

if min(size(I))>1 || min(size(J))>1 || any(round(I)~=I) || any(round(J)~=J)
    error('indices must be vectors of integers');
end;
I = I(:);
J = J(:);
[nra,nca] = size(A);
[nrb,ncb] = size(B);
if (size(I,1)~=nrb || size(J,1)~=ncb) && (nrb*ncb~=1)
    error('bad dimensions for increment matrix');
end;
if min(I)<1 || max(I)>nra || min(J)<1 || max(J)>nca
    error('index out of range');
end;
P = sparse(I,1:nrb,1,nra,nrb);
Q = sparse(1:ncb,J,1,ncb,nca);
C = A + P*(B*Q);