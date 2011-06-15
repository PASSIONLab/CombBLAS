

% example script that runs the extended LDLC decoder for solving
% the multiuser detection problem
% Written by Danny Bickson.
% updated: 18-Dec-2008
%
% Supplamentary material of the paper: 
% "Low density lattice decoder via non-parametric belief propagation"
% By D. Bickson, A. T. Ihler, and D. Dolev,
% Submitted to ISIT 2009.
%
function [x,y,transmission] = run_LDLC(flag)


rand('state',sum(100*clock));

% this command should be updated to the location of the KDE matlab package
% found on: http://ttic.uchicago.edu/~ihler/code/
addpath('d:\gabp\ihler\');

%this command points to the root fo he gabp-sec package
%found in 
addpath('..');

% sparse decoding matrix H
% Note that H is not square! Specifically, not a magic square.
H = [-0.8000    0.7000    0.5000         0         0   -1.0000
         0         0    1.0000   -0.5000    1.0000   -0.8000
   -0.5000    1.0000         0   -0.8000         0    0.5000
    1.0000         0    0.8000         0    0.5000         0
         0   -0.5000         0    1.0000    0.8000         0];

cn = size(H,2); % check nodes
vn = size(H,1); % variable nodes    
     
G = H'*inv(H*H');

sigma_squared = .18;

transmission = sign(rand(vn,1)-0.5); % transmit random [-1,1]
y1 = G*(transmission); 
y = y1+sqrt(sigma_squared)*randn(cn,1); % random AWGN noise 
max_rounds = 100; % max number of iterations
epsilon = 0.001; % convergence threshold

% Calling the LDLC decoder

x = LDLC(H,y,max_rounds, epsilon, sigma_squared)';

%comparing the result to iterative linear detection. See:
%Gaussian belief propagation based multiuser detection. D. Bickson, O.
%Shental, P. H. Siegel, J. K. Wolf, and D. Dolev, In IEEE Int. Symp. on Inform. 
%Theory (ISIT), Toronto, Canada, July 2008. 
if (flag == 2)
    bigmat = [ eye(vn,vn) G'; G -eye(cn,cn)*sigma_squared ];
    bigvec = [ zeros(vn,1)' y']';
    y = gabp(bigmat,bigvec,max_rounds,epsilon);
    y = sign(y(1:vn));
end

disp(['answer should be']);
transmission'

end

