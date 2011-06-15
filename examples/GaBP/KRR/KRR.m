%Function for computing Kernel Ridge regression
%
% Written by Danny Bickson, CMU
% danny.bickson@gmail.com
% Thanks to Arthur Gretton!
%
% Inputs: X - a matrix of data of size rows x cols
%         y - a vector of size rows x 1
%         band - kernel bandwidth (optional)
%         lambda - regulatization (optional)
%         positive - force only positive weights (optional)
% Output: w - a vector of output weights



function [w] = KRR(X,y, band, lambda, positive)

if (~exist('band','var'))
    band = 1;
end
if (~exist('lambda','var'))
    lambda=0;
end
if (~exist('positive','var'))
    positive=0;
end


[rows,cols]=size(X);
for i=1:rows
    for j=1:rows
        K(i,j) = (1/sqrt(2*pi*band^2))*exp(-0.5*sum((X(i,:)-X(j,:)).^2)/band^2);
    end
end

f1 = -2*y'*K;


opts.maxIter = 1000000;
if (positive)
    w=quadprog(K'*(K+lambda*eye(length(K))), f1',[],[],[],[],zeros(length(K),1),ones(length(K),1));
else
    w = (K'*(K+lambda*eye(length(K))))\f1';
end

end

