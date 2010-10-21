function [ii, jj, Aij, vperm, tperm] = rmat(scale,density,vpermute,tpermute)
% RMAT500 : generate power-law directed graph with R-MAT algorithm
%
% A = rmat(scale)   returns a graph (adj matrix) with 2^scale vertices,
%                   with vertex numbers not randomized.
% [I J Aij] = rmat(scale)  returns triplets, with vertex numbers
%                          and triplet order randomized.
% [I J Aij vperm tperm] = rmat(scale) also returns the randomizing perms.
%
% rmat(scale,density,vpermute,tpermute) 
%   has about density * 2^scale edges (default density is 8)
%   randomizes vertex numbers only if vpermute = true (default true)
%   randomizes triple order only if tpermute = true (default true)


% Implementation of the Recursive MATrix (R-MAT) power-law graph
% generation algorithm (Chakrabati, Zhan & Faloutsos).
% This is a "nice" implementation in that  it is completely embarrassingly 
% parallel and does not require ever forming the adjacency matrix.
% Original by Jeremy Kepner, repackaged by John R. Gilbert, summer 2006

if nargin < 2 || isempty(density)
    density = 8;
end;
if nargin < 3
    vpermute = true;
end;

% Set number of vertices.
lgNv = scale;
Nv = 2^lgNv;

% Set number of edges.
Ne = density * Nv;

% Set R-MAT probabilities.
% Create a single parameter family.
% Parameters can't be symmetric in order
% for it to be a power law distribution.
p = 0.6; 
a = p;  b = (1 - a)/3;  c = b;  d = b;

% Create index arrays.
ii = ones(Ne,1);
jj = ones(Ne,1);
% Loop over each order of bit.
ab = a+b;
c_norm = c./(c+d);
a_norm = a./(a+b);
for ib = 1:lgNv
  % Compare with probabilities and set bits of indices.
  ii_bit = rand(Ne,1) > ab;
  jj_bit = rand(Ne,1) > ( c_norm.*ii_bit  + a_norm.*not(ii_bit) );
  ii = ii + (2^(ib-1)).*ii_bit;
  jj = jj + (2^(ib-1)).*jj_bit;
end

if nargout <= 1
    % return adjacency matrix
    ii = sparse(ii,jj,1,Nv,Nv);
else
    % return triples
    if tpermute
        % permute the order of triples
        tperm = randperm(Ne);
        ii = ii(tperm);
        jj = jj(tperm);
    else
        tperm = [];
    end;
    if vpermute
        % permute the vertex numbers
        vperm = randperm(Nv);
        ii = vperm(ii)
        jj = vperm(jj);
    else
        vperm = [];
    end;
    Aij = ones(Ne,1);
end;