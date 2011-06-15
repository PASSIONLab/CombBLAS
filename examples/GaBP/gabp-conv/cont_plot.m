% Suplamentary material for the paper:
% "Message Passing Multi-user detection"
% By Danny Bickson and Carlos Guestrin, CMU
% Submitted to ISIT 2010, January 2010.
%
% Code written by Danny Bickson.
%
% This code Produces figure 1.


function [] = cont_plot()
rand('state',106);
k=2;n=3;
while(1)
    A=rand(n,k);
    A(A<=0.5)=-1;
    A(A>0.5)=1;
    if (rank(A'*A)== k)
        break;
    end
end



y1=A*ones(k,1);
y=[zeros(k,1)' y1']';
B=[zeros(k) A'; A eye(n)];
assert(rank(B)==n+k);

%let the variance converge first
GBP3(B,y,250,1e-5,1);
%and then run the means
[x,J,J2,J_j,C,cost,msg_norm]=GBP3(B,y,250,1e-5,2);
disp(['diagonal should be ', num2str(diag(inv(B))')]);
disp(['answer should be ', num2str((B\y)')]);

%verify precalculated variances
assert(norm(J_j(1,n)-(k-n))<1e-6);
assert(norm(J_j(n,1)-1/(1-(k-1)/(n-1)))<1e-6);
assert(norm(J(1)+1/(n*(1-(k-1)/(n-1))))<1e-6);
assert(norm(J(n)-(1/(1-k/(k-n))))<1e-6);


C=A'*A;
max(abs(eig(diag(diag(C))-abs(C))))/(n-1)
figure;
hold on;
box('on');
semilogy(0:19,msg_norm(1:1:20),'r-o',0:19,cost(1:1:20),'g-+');
legend('||h^t-h^{t-1}||_2^2 Msg Norm','||Ax^t-y||_2^2 Residual');
xlabel('Iteration number', 'FontSize', 14);
ylabel('Solution Norm','FontSize', 14);
end



