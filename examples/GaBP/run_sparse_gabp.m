%This program is free software: you can redistribute it and/or modify
%it under the terms of the GNU General Public License as published by
%the Free Software Foundation, either version 3 of the License, or
%(at your option) any later version.

%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.

%You should have received a copy of the GNU General Public License
%along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%example for running sparse GaBP code for solving x = inv(A)*b
% where A, b are given and A is sparse

A=sparse(4,4);
A(1,1) = 1;
A(2,2) = 1;
A(3,3) = 1;
A(4,4) = 1;
A(2,3) = 0.3;
A(2,4) = 0.4;
A(3,2) = 0.3;
A(4,2) = 0.4;
A(1,4) = -0.5;
A(4,1) = -0.5;

b=ones(1,4);

[h,J] = sparse_gabp(A,b,30,0.0000001);
disp(['sparse gabp result is: ']);
h
disp(['direct inversion result is: ']);
(inv(A)*b')'
disp(['Main diagonal of inv(A) is (approximation): ']);
J
disp(['Main diagonal of inv(A) using direct method']);
diag(inv(A)
