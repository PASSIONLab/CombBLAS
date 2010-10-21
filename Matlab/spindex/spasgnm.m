function C = sppluseqm(A,I,J,B)
% SPASGNM : Sparse matrix left indexing using sparse matrix multiplication
%
% For integer vectors I and J,
% A = spasgnm(A,I,J,B) has the effect of A(I,J) = B
% C = spasgnm(A,I,J,B) has the effect of C = A; C(I,J) = B
%
% John R. Gilbert, September 6, 2010

if min(size(I))>1 || min(size(J))>1 || any(round(I)~=I) || any(round(J)~=J)
    error('indices must be vectors of integers');
end;
if any(diff(sort(I))==0) || any(diff(sort(J))==0)
    error('duplicate indices are not allowed');
end;
I = I(:);
J = J(:);
[nra,nca] = size(A);
[nrb,ncb] = size(B);
if (size(I,1)~=nrb || size(J,1)~=ncb) && (nrb*ncb~=1)
    error('bad dimensions for increment matrix');
end;
if min(I)<1 || min(J)<1
    error('index out of range');
end;
if max(I)>nra
    K = sparse(1:nra,1:nra,1,max(I),nra); % or K = speye(max(I),nra)
    A = K*A;
    nra = max(I);
end;
if max(J)>nca
    L = sparse(1:nca,1:nca,1,nca,max(J)); % or L = speye(nca,max(J))
    A = A*L;
    nca = max(J);
end;
P = sparse(I,1:nrb,1,nra,nrb);
Q = sparse(1:ncb,J,1,ncb,nca);
R = sparse(I,I,1,nra,nra);
S = sparse(J,J,1,nca,nca);
C = A - R*(A*S) + P*(B*Q);