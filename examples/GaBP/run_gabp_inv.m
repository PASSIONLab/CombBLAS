% example for running the gabp algorithm, for computing inv(A), given A
% Written by Danny Bickson
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%format long;

%A = [1 0.3 0.1;0.3 1 0.1;0.1 0.1 1];
A=rand(100,555);
A=A*A'+eye(100)*250;

inv_direct = inv(A);
max_iter = 20;
epsilon = 0.000001;

[invA] = gabp_inv(A, max_iter, epsilon);

disp('inv(A) computed by gabp is: ');
invA
disp('inv(A) computed directly is : ');
inv_direct
