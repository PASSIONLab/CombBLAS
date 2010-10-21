function Z = min_plus(X,Y,Z);
% MIN_PLUS : matrix multiplication with (min,+) instead of (+,*)
%
%     Z = min_plus(X,Y);
% or
%     Z = min_plus(X,Y,Z);
%
% Input:  X is an n1-by-n2 matrix
%         Y is an n2-by-n3 matrix
%         Z is an n1-by-n3 matrix
% Output: Z is an n1-by-n3 matrix
%
% This computation looks exactly like the matrix multiplication  
%        Z  =  X*Y;               [ for  Z = min_plus(X,Y);  ]
% or 
%        Z  =  Z + X*Y;           [ for Z = min_pluz(X,Y,Z); ]
% except that it uses "min" instead of "+" to accumulate rows,
% and it uses "+" instead of "*" to combine individual elements.
%
% This code is a simple triply nested loop; it can be sped up or
% parallelized using the same techniques as for matrix multiplication.
%
% John R. Gilbert, 17 February, 2010

[n1,n2] = size(X);
[n,n3] = size(Y);
if n2 ~= n, error('Inner dimensions of X and Y do not match'); end
if nargin == 3
    % Add X*Y to input matrix Z
    [nr,nc] = size(Z);
    if (nr~=n1) || (nc~=n3), error('Dimensions of Z are wrong'); end;
else
    % Just compute X*Y
    Z = Inf * ones(n1,n3);  % "Inf" replaces "0" when "min" replaces "+"
end;

for i = 1:n1
    for j = 1:n3
        for k = 1:n2
            % In normal matrix multiplication, the following line would be
            % Z(i,j) =      Z(i,j) + X(i,k)*Y(k,j)  ;
              Z(i,j) = min( Z(i,j) , X(i,k)+Y(k,j) );
        end;
    end;
end;
return;