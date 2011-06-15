

% example script that test NBP
% with a setting of a single Gaussian in each self potential
% in this setting, the algorithm is identical to the Gaussian BP algorithm
% Written by Danny Bickson.
% updated: 18-Dec-2008
%
% Supplamentary material of the paper: 
% "Low density lattice decoder via non-parametric belief propagation"
% By D. Bickson, A. T. Ihler, and D. Dolev,
% Submitted to ISIT 2009.
%
function [x,y,transmission] = run_NBP_test(flag)

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

max_rounds = 100; % max number of iterations
epsilon = 0.001; % convergence threshold

H=[1 0.2 0.3; 0.2 1 -0.2; 0.3 -0.2 1];
BIGH = [ eye(3,3) H; H' eye(3,3)];
y =ones(3,1);
BIGy=[zeros(3,1)' y']';

[x,p] = GBP(H,y,10,0.00000001);
x
p
self_pot_val = kde(1,1);
self_pot(1) = self_pot_val;
self_pot(2) = self_pot_val;
self_pot(3) = self_pot_val;
x = NBP(H,self_pot,30, 1e-10,'exact')

%comparing the result to iterative linear detection. See:
%Gaussian belief propagation based multiuser detection. D. Bickson, O.
%Shental, P. H. Siegel, J. K. Wolf, and D. Dolev, In IEEE Int. Symp. on Inform. 
%Theory (ISIT), Toronto, Canada, July 2008. 

disp(['answer should be']);
(inv(H)*y)'

[getPoints(x{1}) getPoints(x{2}) getPoints(x{3})]
end

