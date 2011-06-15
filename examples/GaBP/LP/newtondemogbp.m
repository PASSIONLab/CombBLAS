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
% Supplemantry code for the paper:
% Polynomial Linear Programming with Gaussian Belief Propagation.
% By Danny Bickson, Yoav Tock, Ori Shental, Paul H. Seigel, Jack K. Wolf
% and Danny Dolev.
% Submitted to the Forty-Sixth Annual Allerton Conference on Communication, Control, and Computing
% September 23 - September 26, 2008, Allerton House, Monticello, Illinois
%
% Written by  Danny Bickson, IBM Haifa Research Lab
% 26-6-08
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example solution of a linear program using affine-scaling algorithm
% Algorithm is described in:
% Ilan Adler, Narendra Karmarkar, Mauricio G.C. Resende and Geraldo Veiga
% (1989). "An Implementation of Karmarkar's Algorithm for Linear Programming". 
% Mathematical Programming, Vol 44, p. 297–335.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We solve the following LP problem:
%    minimize(c'x)
%    subject to Ax <= b
% Problem setup is borrowed from:
% http://en.wikipedia.org/wiki/Karmarkar's_algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('..');

%problem input: A, b, c
A=[0 1; 0.2 1; 0.4 1; 0.6 1; 0.8 1; 1.0 1;1.2 1;1.4 1;1.6 1;1.8 1; 2 1];
b=[1 1.01 1.04 1.09 1.16 1.25 1.36 1.49 1.64 1.81 2]';
c=[1 1]';
%max number of Newton steps
max_round=30;
%solution vector (including intermediate solutions)
x=zeros(2,max_round+1);
%used by the affine-scaling algorithm
gamma = 0.5;

for r=1:max_round;
   %perform affine sclaing
   vk=b-A*x(:,r);
   D=diag(vk); 
    

   %matrix inversion task is done efficiently using the GaBP algorithm
   %documented in the file gabp.m
   %for comparing to matlab matrix inverse, uncomment the following two
   %lines, and add instead:  %hx=inv(A'*(inv(D)^2)*A)*c;
   [hx, j] = gabp(A'*(inv(D)^2)*A, c, 15, 0.0001);
   hx = hx';
   hv=-A*hx;
   % check feasibility
   if (hv >= 0)
       error ('unbounded');
   else
       alpha = gamma*min(-vk./hv);
       x(:,r+1)=x(:,r)+alpha*hx;
   end
    
end

% plot result
figure;
hold on;
plot(x(1,:), x(2,:),'-or');
vec = 0:0.05:1;
for i=1:11
    plot(vec, b(i) - A(i,1)*vec);
end
title('Newton method using GaBP');  
xtitle('x1');
ytitle('x2');
