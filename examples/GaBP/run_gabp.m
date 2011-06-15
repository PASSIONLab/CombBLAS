% example for running the gabp algorithm, for computing the solution to Ax = b
% Written by Danny Bickson

% Initialize

%format long;
n = 3;
A = [1 0.3 0.1;0.3 1 0.1;0.1 0.1 1];
b = [1 1 1]';
x = inv(A)*b;
max_iter = 20;
epsilon = 0.000001;

[x1, p] = gabp(A, b, max_iter, epsilon);

disp('x computed by gabp is: ');
x1
disp('x computed by matrix inversion is : ');
x'
disp('diag(inv(A)) computed by gabp is: (this is an approximation!) ');
p
disp('diag(inv(A)) computed by matrix inverse is: ');
diag(inv(A))'