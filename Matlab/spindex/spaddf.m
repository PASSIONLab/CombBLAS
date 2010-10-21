function A = spaddf(B,C)
% SPADDF : matrix addition using find & sparse
%
% A = spaddf(B,C) has the effect of A = B + C
%
% John R. Gilbert, September 8, 2010

[nrb,ncb] = size(B);
[nrc,ncc] = size(C);
if ncb~=ncc || nrb~=nrc
    error('inputs must have same dimensions');
end;
[IB,JB,VB] = find(B);
[IC,JC,VC] = find(C);
A = sparse([IB;IC],[JB;JC],[VB;VC],nrb,ncb);
if (~issparse(B)) || (~issparse(C))
    A = full(A);
end;