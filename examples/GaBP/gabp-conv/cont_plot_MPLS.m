
% Suplamentary material for the paper:
% "Message Passing Multi-user detection"
% By Danny Bickson and Carlos Guestrin, CMU
% Submitted to ISIT 2010, January 2010.
%
% Code written by Danny Bickson.
%
% This script provies a running example of the basic algorithm

function [] =cont_plot_MPLS()

rand('state',99);
k=5;n=8;
while(1)
    A=rand(n,k);
    A(A<=0.5)=-1;
    A(A>0.5)=1;
    %verify that the matrix is full column rank
    if (rank(A*A')== k)
        break;
    end
end


t=sqrt(n);
y1=A/t*ones(k,1);
y=[zeros(k,1)' y1']';
B=[zeros(k) A'/t; A/t eye(n)];

assert(rank(B)==n+k);
[h,r]=MPLS3(B,y,250,1e-5,t,k,n);
if (r==250)
    warning('GaBP did not converge, aborting');
    return;
end
disp(['diagonal should be ', num2str(diag(inv(B))')]);
disp(['answer should be ', num2str((A\y1)')]);
end
