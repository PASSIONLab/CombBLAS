function A = vcatf(B,C)
% VCATF : vertical matrix concatenation using find & sparse
%
% A = vcatf(B,C) has the effect of A = [B ; C]
%
% John R. Gilbert, September 8, 2010

[nrb,ncb] = size(B);
[nrc,ncc] = size(C);
if ncb~=ncc
    error('inputs must have same number of columns');
end;
[IB,JB,VB] = find(B);
[IC,JC,VC] = find(C);
A = sparse([IB;IC+nrb],[JB;JC],[VB;VC],nrb+nrc,ncb);
if (~issparse(B)) && (~issparse(C))
    A = full(A);
end;