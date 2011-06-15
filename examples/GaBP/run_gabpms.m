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
                                                                     
                                                                     
                                             
function [] = run_gabpms()

A=[1.0000    0.5000    0.3000
 0.5000    1.0000    0.3000
 0.3000    0.3000    1.0000];

b=ones(3,1);

h = gabpms(A,b,20,0.0000000001);

disp(['solution computed by Min-SUM gabp']);
h
% [h,J] = gabp(A,b,20,0.00000000001);

disp(['solution via direct computation is: ']);
inv(A)*b
end


