function A = hcatm(B,C)
% HCATM : horizontal matrix concatenation using matrix multiplication
%
% A = hcatm(B,C) has the effect of A = [B C]
%
% John R. Gilbert, September 6, 2010

[nrb, ncb] = size(B);
[nrc, ncc] = size(C);
if nrb~=nrc
    error('inputs must have same number of rows');
end;
P = sparse(1:ncb,1:ncb,1,ncb,ncb+ncc);
Q = sparse(1:ncc,(1:ncc)+ncb,1,ncc,ncb+ncc);
if issparse(B)==issparse(C)
    A = B*P + C*Q;
elseif issparse(B)
    A = B*P + sparse(C*Q);
else
    A = sparse(B*P) + C*Q;
end;

