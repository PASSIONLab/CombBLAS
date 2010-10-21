function A = hcatf(B,C)
% HCATF : horizontal matrix concatenation using find & sparse
%
% A = hcatf(B,C) has the effect of A = [B C]
%
% John R. Gilbert, September 8, 2010

[nrb,ncb] = size(B);
[nrc,ncc] = size(C);
if nrb~=nrc
    error('inputs must have same number of rows');
end;
[IB,JB,VB] = find(B);
[IC,JC,VC] = find(C);
A = sparse([IB;IC],[JB;JC+ncb],[VB;VC],nrb,ncb+ncc);
if (~issparse(B)) && (~issparse(C))
    A = full(A);
end;