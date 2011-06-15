function [] = test_blockGBP2()

%A=rand(12,10);
%A=A'*A+20*diag(10);
%b=rand(10,1);
rand('state', 1);
A=[1 0.5 0.2; 0.5 1 -0.2; 0.2 0.5 1];
b=[1 1 1]';
%[h,J,r]=blockGBP(A,b', 100,1e-5,1);
A\b

%[h1,J1,r1]=blockGBP(A,b', 100,1e-5,3);
A\b


A=rand(12,10);
A=A'*A+40*eye(10);
b=rand(10,1);
%[h1,J1,r1]=blockGBP(A,b', 100,1e-5,1);
(A\b)'
[h1,J1,r1]=blockGBP2(A,b', 100,1e-7,2);
(A\b)'

%save_c_gl('blockgbp', A, b, A\b);
end