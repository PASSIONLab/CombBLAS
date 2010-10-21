function A = vcatm(B,C)
% VCATM : vertical matrix concatenation using matrix multiplication
%
% A = vcatm(B,C) has the effect of A = [B ; C]
%
% John R. Gilbert, September 6, 2010

[nrb, ncb] = size(B);
[nrc, ncc] = size(C);
if ncb~=ncc
    error('inputs must have same number of columns');
end;
P = sparse(1:nrb,1:nrb,1,nrb+nrc,nrb);
Q = sparse((1:nrc)+nrb,1:nrc,1,nrb+nrc,nrc);
if issparse(B)==issparse(C)
    A = P*B + Q*C;
elseif issparse(B)
    A = P*B + sparse(Q*C);
else
    A = sparse(P*B) + Q*C;
end;