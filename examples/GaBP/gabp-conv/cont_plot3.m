% Suplamentary material for the paper:
% "Message Passing Multi-user detection"
% By Danny Bickson and Carlos Guestrin, CMU
% Submitted to ISIT 2010, January 2010.
%
% Code written by Danny Bickson.
%
% This code Produces figure 2.

function []=cont_plot3()
rand('state',112);
k=3;n=5;
while(1)
    A=rand(n,k);
    A(A<=0.5)=-1;
    A(A>0.5)=1;
    %verify matrix is full column rank
    if (rank(A*A')== k)
        break;
    end
end

y1=A*ones(k,1);
y=[zeros(k,1)' y1']';
B=[zeros(k) A'; A eye(n)];
assert(rank(B)==n+k);
[x,J,J2,J_j,C,cost]=GBP2(B,y,250,1e-5,1);
disp(['diagonal should be ', num2str(diag(inv(B))')]);
disp(['answer should be ', num2str((B\y)')]);

%Verify variances behave as they are predicted numerically
assert(norm(J_j(1,n)-(k-n))<1e-6);
assert(norm(J_j(n,1)-1/(1-(k-1)/(n-1)))<1e-6);
assert(norm(J(1)+1/(n*(1-(k-1)/(n-1))))<1e-6);
assert(norm(J(n)-(1/(1-k/(k-n))))<1e-6);



[t1 t2]=meshgrid(0:0.1:2,0:0.1:2);
t=[t1(:) t2(:)];
for i=1:length(t)
   tt(i,1) =  norm(A*[t(i,:) x(3)]'-y1);
end

figure(1);
contour(t1,t2,reshape(tt,size(t1)),[0.0:0.1:2]);
axis([0.4 1.6 0.4 1.6]);
hold on;
plot(C([1:2:25],2),C([1:2:25],1),'-*');


end




