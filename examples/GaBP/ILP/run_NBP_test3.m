

% example script that compares LDLC vs. ILP via NBP
% Written by Danny Bickson.
% updated: May-2009
%

function [x,y,rx] = run_NBP_test3()


rand('state',sum(100*clock));
randn('state',sum(100*clock));

% this command should be updated to the location of the KDE matlab package
% found on: http://ttic.uchicago.edu/~ihler/code/
addpath('..');

%this command points to the root fo he gabp-sec package
%found in 
addpath('..');

max_rounds = 10; % max number of iterations
epsilon = 1e-10; % convergence threshold

%unlike LDLC, we assume (a potentially) sparse encoding matrix G,
G=[0.4 -0.2 0.29; 0.2 0.44 -0.2; 0.3 -0.2 0.33];
BIGG = [ zeros(3,3) G; G' zeros(3,3)];
tx = [-1 1 -1]';
sigma = 0.1;

y = G*tx+randn(3,1)*sqrt(sigma);
self_pot_val = kde([-1 1],[.1 .1]);
self_pot(1) = self_pot_val;
self_pot(2) = self_pot_val;
self_pot(3) = self_pot_val;
self_pot(4) = kde(y(1), 1/sqrt(sigma));
self_pot(5) = kde(y(2), 1/sqrt(sigma));
self_pot(6) = kde(y(3), 1/sqrt(sigma));

x = NBP(BIGG,self_pot,max_rounds, epsilon,'gibbs2')

%comparing the result to iterative linear detection. See:
%Gaussian belief propagation based multiuser detection. D. Bickson, O.
%Shental, P. G. Siegel, J. K. Wolf, and D. Dolev, In IEEE Int. Symp. on Inform. 
%Theory (ISIT), Toronto, Canada, July 2008. 

disp(['rx']);
tx'

disp(['using linear detection']);
sign((inv(G)*y)')

ret = [max(x{4}) max(x{5}) max(x{6})]';
x = G*ret;
disp('ILP');
x = sign(x)'
disp(['number of ILP errors', num2str(sum(tx~=x'))]);

addpath('../LDLC/');
xLDLC = LDLC(G,y,10,1e-10,sigma);
disp(['number of LDLC errors', num2str(sum(tx~=xLDLC))]);
end

