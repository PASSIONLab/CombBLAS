function P = apsp(H)
% APSP : all-pairs shortest paths by recursive Kleene algorithm
%
% P = apsp(H);
%
% Input is matrix of step lengths:  
%   H(i,j) is length of step from i to j
% Output is matrix of shortest path lengths:  
%   P(i,j) is length of shortest path from i to j
%
% This code uses several different matrices, but
% an efficient (parallel or sequential) code would
% just use one matrix:  It would overwrite H with P,
% and would pass data to the recursive calls in-place
% instead of making copies of A, B, C, and D.
%
% Example: 
%
% The following input represents a graph with 4 nodes:
%
%      H = [   0   5   9 Inf ;
%            Inf   0   1 Inf ;
%            Inf Inf   0   2 ;
%            Inf   3 Inf   0 ]
%
% The shortest paths are given by:
%
%      P = [   0   5   6   8 ;
%            Inf   0   1   3 ;
%            Inf   5   0   2 ;
%            Inf   3   4   0 ]
%
% John Gilbert, 17 February 2010

[n,nc] = size(H);
if n ~= nc, error ('H must be square'); end;
if min(min(H)) < 0, error ('Entries in H must not be negative'); end;
if any(diag(H)~=0), error ('Diagonal entries in H must all be 0'); end;

% Base case:  If graph has at most 2 nodes, shortest paths are 1 hop.
if n <= 2 
    P = H;
    return;
end;

% Split H recursively as H = [A B ; C D]

n1 = floor(n/2);    % If n is odd, n1 will be one less than n-n1.
A = H(1:n1,1:n1);
B = H(1:n1,n1+1:n);
C = H(n1+1:n,1:n1);
D = H(n1+1:n,n1+1:n);

% Solve the problem recursively, 
% using "matrix multiplication" with (min,+) instead of (+,*)

A = apsp(A);          % recursive call, compute path lengths within A
B = min_plus(A,B);    % B = A*B;       now B includes paths through A
C = min_plus(C,A);    % C = C*A;       now C includes paths through A
D = min_plus(C,B,D);  % D = D + C*B;   now D includes paths through A
D = apsp(D);          % recursive call, compute path lengths within D
B = min_plus(B,D);    % B = B*D;       now B includes paths through D
C = min_plus(D,C);    % C = D*C:       now C includes paths through D
A = min_plus(B,C,A);  % A = A + B*C;   now A includes paths through D

% Reassemble the output matrix

P = [A B ; C D];
return;
