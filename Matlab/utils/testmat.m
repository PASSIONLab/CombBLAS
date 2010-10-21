function A = testmat(nr,nc)
% TESTMAT : create test matrix for indexing functions
%
% A = testmat(nr,nc) returns a full nr-by-nc matrix whose entries
%                  can be read as i,j coordinates
%
% John R. Gilbert,  4 Sep 2010

if nargin < 1
    nr = 7;
end;
if nargin < 2
    nc = nr;
end;
Rows = (1:nr)'*ones(1,nc);
Cols = ones(nr,1)*(1:nc);
scale = 10^(1+floor(log10(nc)));
A = scale*Rows + Cols;