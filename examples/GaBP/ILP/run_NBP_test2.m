

% example script that runs NBP for solving BLS (boolean least squares )
% problem
% Written by Danny Bickson.
% updated: 3-March-2009
%
%
function [x,y,transmission] = run_NBP_test2(flag)

if (nargin < 1)
    flag = 0;
end

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

H=[0.4 -0.2 0.29; 0.2 0.44 -0.2; 0.3 -0.2 0.33];
BIGH = [ zeros(3,3) H; H' zeros(3,3)];
% y =ones(3,1);
% BIGy=[zeros(3,1)' y']';
% 
% [x,p] = GBP(H,y,10,0.00000001);
% x
% p
transmit = [-1 1 -1]';

y = H*transmit+randn(3,1)*sqrt(0.1);
self_pot_val = kde([-1 1],[.1 .1]);
self_pot(1) = self_pot_val;
self_pot(2) = self_pot_val;
self_pot(3) = self_pot_val;
self_pot(4) = kde(y(1), 1);
self_pot(5) = kde(y(2), 1);
self_pot(6) = kde(y(3), 1);

x = NBP(BIGH,self_pot,max_rounds, epsilon,'gibbs2')

%comparing the result to iterative linear detection. See:
%Gaussian belief propagation based multiuser detection. D. Bickson, O.
%Shental, P. H. Siegel, J. K. Wolf, and D. Dolev, In IEEE Int. Symp. on Inform. 
%Theory (ISIT), Toronto, Canada, July 2008. 

disp(['transmission']);
transmit'

disp(['using linear detection']);
sign((inv(H)*y)')

ret = [max(x{4}) max(x{5}) max(x{6})]';
x = H*ret;
disp('ILP');
x = sign(x)'
end

