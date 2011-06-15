% Written by Danny Bickson, CMU
% Matlab code for running Lanczos algorithm

function [] = test_lanczos()

A=rand(5,12); A=A*A'
disp(['eigenvalues are ', num2str(flipud(eig(A))')]);
for i=2:4
    disp(['running for ', num2str(i), ' rounds']);
    lanczos(A,i)
end
end