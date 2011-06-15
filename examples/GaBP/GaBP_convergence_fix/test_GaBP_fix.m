% Example for running the fix_GaBP code
% Given a system of linear equation Ax = b, the algorithm
% solves x = inv(A)*b iteratively, even when the sufficient convergence
% conditions of gaussian BP (or other iterative algorithms like Jacobi, Gauss-Seidel etc. ) 
% do not hold.

% Input: A - symmetric matrix of size nxn
%        b - vector of size nx1
%        max_iter - maximal number of iterations 
%        epsilon - convergence threshold
% Output:
%        x - vector of size nx1, x = inv(A)*b

% Algorithm is described in the paper:
% "Fixing convergence of GaBP algorithm" by
% J. K. Johnson, D. Bickson and D. Dolev
% In ISIT 2009
% http://arxiv.org/abs/0901.4192
% Code written by Danny Bickson
% May 2009

function []=test_GaBP_fix()
    
    A=rand(3,8);
    A=A*A';
    b= ones(3,1);
    x = fix_GaBP(A,b,100,1e-10);
    disp('Fixed result is ');
    x'
    disp('Matrlab result is ');
    (inv(A)*b)'

end