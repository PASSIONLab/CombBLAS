% Suplamentary material for the paper:
% "Message Passing Multi-user detection"
% By Danny Bickson and Carlos Guestrin, CMU
% Submitted to ISIT 2010, January 2010.
%
% Code written by Danny Bickson.

function [h,J,r,J_j,C,cost,msg_norm] = GBP3(A,b,maxround,epsilon,command)

m=length(A);
Mh=zeros(m);

if (command==2)
    load MJ.mat;
else
    MJ=zeros(m,m);
end
old_Mh=Mh;
old_MJ=MJ;

h_j=Mh;
J_j=ones(m);
h=zeros(1,m);
J=zeros(1,m);

%
format long;

conv = false;
C=zeros(maxround,m);
% algorithm rounds
for r=1:maxround
    %disp(['starting GBP round ', num2str(r)]); 
  
	% for each node
   for i=1:m
		% sum up all mean and percision values got from neighbors
		h(i) = b(i) + sum(Mh(:,i));  %(7)
        J(i) = A(i,i) + sum(MJ(:,i));

		% send message to all neighbors
        for j=1:m
			if (i ~= j && A(i,j) ~= 0)
				h_j(i,j) = h(i) - Mh(j,i);
				J_j(i,j) = J(i)- MJ(j,i);
			    if (J_j(i,j) == 0) % avoid division by zero when diagonal is zero
                        Mh(i,j) = 0; MJ(i,j) = 0;
			    else
                		Mh(i,j) = (-A(j,i) / J_j(i,j))* h_j(i,j); %(8)
                        MJ(i,j) = (-A(j,i) / J_j(i,j)) * A(i,j);
                end
            end
        end
       
        
   end
   
    msg_norm(r)=norm(Mh-old_Mh,'fro');
  	disp([num2str(r), ') norm x is : ', num2str(norm(((h./J)'))), ' norm Ax-y ', num2str(norm(A*((h./J)')-b)),...
            ' norm Ax',num2str(norm(A*(h./J)')), ' msg norm ', num2str(msg_norm(r))]);

    C(r,:)=h./J;
    cost(r)=norm(A*((h./J)')-b);

    if (r > 2 && (norm(MJ-old_MJ) < epsilon) && (norm(Mh-old_Mh)<epsilon))
        disp(['GBP (h) Converged afeter ', num2str(r), ' rounds ']); 
        
        conv = true;
		break;
   end
   
   old_Mh = Mh; old_MJ = MJ;
   
end
if (conv == false)
	disp(['GBP (MJ) Did not converge in ', num2str(r), ' rounds ']);
end
if (command == 1)
       save MJ.mat MJ;
 end
J = 1./J;
h=h.*J;
disp(['GBP result h is: ', num2str(h)]);
disp(['GBP result J is: ', num2str(J)]);
