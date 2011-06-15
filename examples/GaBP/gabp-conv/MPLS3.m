
% Suplamentary material for the paper:
% "Message Passing Multi-user detection"
% By Danny Bickson and Carlos Guestrin, CMU
% Submitted to ISIT 2010, January 2010.
%
% Code written by Danny Bickson.
%
% This is an implementation of the basic algorithm presented in the above
% paper

% Input:
% A - square matrix of the type [ 0 B'; B I ] of size mxm
% b - vector of the size mx1 containing [ 0^T y^T]
% maxround - maximal round to run
% epsilon - convergence threshold
% t - normalizing constant of B
% k - number of hidden nodes
% n - number of observed nodes

% Output
% h - the solution h=B/t\y
% other statistics..
function [h,J,r,J_j,C,cost,msg_norm] = MPLS3(A,b,maxround,epsilon,t,k,n)

m=length(A);
Mh=zeros(m);
old_Mh=Mh;
h_j=Mh;
J_j=ones(m);
h=zeros(1,m);
J=zeros(1,m);
format long;

conv = false;
C=zeros(maxround,m);
% algorithm rounds
for r=1:maxround

    for i=1:m
    
       %calc precomputed variances
        if (i > k)
            J(i) = n/(n-k);
        else
            J(i) = n/((1-n)*t^2);
        end

    end

	% for each node
   for i=1:m
		% send message to all neighbors
        for j=1:m
			if (i ~= j && A(i,j) ~= 0)
				h_j(i,j) = b(i) + sum(Mh(:,i))- Mh(j,i);
				%J_j(i,j) = J(i)- MJ(j,i);
			    if (i <= k)
                      Mh(i,j) = A(j,i) *t^2* h_j(i,j); %(8)
                else
                      Mh(i,j) = (A(j,i) / (1-n))* h_j(i,j); 
                end
            end
        end
       
        
   end
   
    msg_norm(r)=norm(Mh-old_Mh);
  	disp([num2str(r), ') norm x is : ', num2str(norm(((h./J)'))), ' norm Ax-y ', num2str(norm(A*((h./J)')-b)),...
            ' norm Ax',num2str(norm(A*(h./J)')), ' msg norm ', num2str(msg_norm(r))]);

    C(r,:)=h./J;
    cost(r)=norm(A*((h./J)')-b);

    if (r > 2 && (norm(Mh-old_Mh,'fro')<epsilon))
        disp(['MPLS Converged afeter ', num2str(r), ' rounds ']); 
            conv = true;
		break;
   end
   
   old_Mh = Mh; 
   
end
if (conv == false)
	disp(['MPLS Did not converge in ', num2str(r), ' rounds ']);
end

h = (b + sum(Mh)')';
h=h./J;
disp(['GBP result h is: ', num2str(h)]);
disp(['GBP result J is: ', num2str(J)]);
