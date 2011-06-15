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

% Function that demonstrates divergence of the GaBP
% algorithm, in the multiuser detection secnario.
% Mulituser detection algorithm is described in:
% Linear Detection via Belief Propagation. Danny Bickson, Danny Dolev, Ori
% Shental, Paul H. Siegel and Jack K. Wolf. In the 45th Annual Allerton Conference on Communication, Control, and 
%Computing, Allerton House, Illinois, Sept. 07.
% 
% Supplamentary material of the paper:
% "Fixing the convergence of the GaBP algorithm" by
% J. K. Johnson, M. Chetrkov, D. Bickson and D. Dolev
% Submitted to ISIT 2009
% Code written by Danny Bickson
% December 2008
function [] = multiuser_detection()

addpath('..');

N=256; %number of chips per symbol
K=96; %number of active users.
A=eye(K); %assuming equal power for now.
dB=-10;

sigma2=10.^(-dB/10); % from dB=10*log10(1/sigma2)
b= rand(K,1); %transmitted bits
b(b < 0.5) = -1;
b(b >= 0.5) = 1;

S = rand(N,K); %random spreading CDMA 
S(S < 0.5) = -1;
S(S >= 0.5) = 1;
S = S/sqrt(N);

a11 = eig(eye(K) - abs(S'*S));
disp(['spectral radius is: ' , num2str(max(abs(a11)))]);

n=sqrt(sigma2)*randn(N,1); %AWGN
r=S*A*b+n; %received signal
y=S'*r; %matched filter output
        
M=S'*S;% correlation matrix
		
[b_est,J,r1,resid] = asynch_GBP(M,y,50,1e-6); %calling the linear solver to iteratively compute b_est = inv(M)*y;

figure;
title('Divergence of GaBP for a multiuser detection problem with n=256,k=96');
xlabel('Iteration');
ylabel('Value of x_i');
plot(resid);


end
