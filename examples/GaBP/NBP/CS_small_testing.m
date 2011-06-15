%Code written by Danny Bickson
%Code is found on: http://www.cs.huji.ac.il/labs/danss/p2p/gabp/index.html
%Example of solving boolean least squares via NBP
%The problem is argmin_x ( norm (Gx - y))  s.t. x \in {-1,1}
%G is a 3x3 matrix
% Note that unlike LDLC paper by Sommer et al., G is a sparse ENCODING matrix

function [G, y, mrecon, transmit] = CS_small_testing()


rand('state',sum(100*clock));
randn('state',sum(100*clock));


%G=[0 3.1 0.8; 2.5 0 -3.5; 9.3 -0.5 0];
%G=[0 0.1 0.8; 0.6 0 -0.5; 0.3 -0.5 0];
% G =  [0    0    0    0    0; 0     1     1    0    0;  0    0    0     1    0;...
%     0     1     1    0    0; 0    0     1    0     1; ...
%      1    0    0    0    0;...
%      1    0    0    0     1;...
%     0    0     1     1    0;...
%      1     1    0     1    0;...
%     0    0     1     1    0;...
%     0    0     1     1     1;...
%      1    0    0    0     1;...
%      1     1    0    0     1;...
%      1     1     1     1    0;...
%     0     1     1     1    0;...
%      1    0     1     1     1];

 
 G=[-1    -1    -1    -1    -1; -1     1     1    -1    -1;  -1    -1    -1     1    -1;...
   -1     1     1    -1    -1; -1    -1     1    -1     1; ...
    1    -1    -1    -1    -1;...
    1    -1    -1    -1     1;...
   -1    -1     1     1    -1;...
    1     1    -1     1    -1;...
   -1    -1     1     1    -1;...
   -1    -1     1     1     1;...
    1    -1    -1    -1     1;...
    1     1    -1    -1     1;...
    1     1     1     1    -1;...
   -1     1     1     1    -1;...
    1    -1     1     1     1];

 
% Transpose! In CS matrix is wider than its height
G = G'

n=size(G,1);

%addpath('../CSBP_matlab');
max_rounds = 30;; % max number of iterations


% Continuous signal

transmit = rand(size(G, 2),1);
%transmit(transmit<0.5) = -1;
%
%transmit(transmit>=0.5) = 1;
transmit


sigma = 0.1;
y = G*transmit+randn(n,1)*sigma;


epsilon=1e-70;
boundx=15;
%---------------
% compute signal node prior
%---------------
model_order = 1023;
xx=1:model_order;  xx=xx-(model_order+1)/2;  xx=xx/max(xx);
xx=xx*boundx; % values over which pdf is sampled

%pdf_prior= normpdf(xx,0.2,1.0) +  normpdf(xx,0.8,1.0);
lambda=0.1;
pdf_prior= laplace(xx,0,1);
pdf_prior=pdf_prior/sum(pdf_prior);

sigma = 1;
noise_prior=normpdf(xx,0,.1); %noise prior
noise_prior=noise_prior/sum(noise_prior);


disp ('STARTING THE DECODER...');

displayind = [ 1 2 3 4 5];

 [xrecon, mrecon, srecon]=NBP(G,transmit',y,...
     sigma, max_rounds, displayind ,epsilon,pdf_prior,noise_prior,xx);


fprintf('[   %f   %f   %f ]\n', transmit(displayind));



% Compare to log barrier
est = tvqc_logbarrier(G\y, G, [], y, 0.01, 1e-4,2);

fprintf('Log Barrier Error=%6.2f (L1 norm) \n',sum(abs(est-transmit)));
fprintf('Log Barrier Reconstruction Error=%6.2f (L1 norm) \n',sum(abs(G*est-y)));
fprintf('Log Barrier cost=%6.2f  \n',norm(G*est-y)+lambda*sum(abs(est)));
fprintf('QBP Error=%6.2f (L1 norm) \n',sum(abs(mrecon-transmit')));
fprintf('QBP Reconstruction Error=%6.2f (L1 norm) \n',sum(abs(G*mrecon'-y)));
fprintf('QBP Barrier cost=%6.2f  \n',norm(G*mrecon'-y)+lambda*sum(abs(mrecon)));
[transmit mrecon' est]
end