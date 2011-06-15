function [] = test_ALS()

A = [ 0 1 0 2; 0 3 0 2; 1 1 0 0; 2 1 0 1];
A=sparse(A);
[a,b,c]=find(A);


% a list of non-zero positions i,j of sparse factorized matrix A
TTr.subs = [a b];
% a list of matrix values
TTr.vals = c;
% size of factorized matrix A
TTr.size = size(A);

D=3; % dimension of matrices U,V s.t. U'V=~ A
max_iter = 10; % number of iterations
ALS(TTr,[], D, max_iter,1,1);