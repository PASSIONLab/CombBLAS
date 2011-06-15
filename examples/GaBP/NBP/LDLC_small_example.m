%Code written by Danny Bickson
%Code is found on: http://www.cs.huji.ac.il/labs/danss/p2p/gabp/index.html
%Example of solving boolean least squares via NBP
%The problem is argmin_x ( norm (Gx - y))  s.t. x \in {-1,1}
%G is a 3x3 matrix
% Note that unlike LDLC paper by Sommer et al., G is a sparse ENCODING matrix

function [] = LDLC_small_example()
clear
rand('state',sum(100*clock));
randn('state',sum(100*clock));


G=[0 0.5 0.8; 0.5 0 -0.5; 0.8 -0.5 0];
n=size(G,1);

%addpath('../CSBP_matlab');
max_rounds = 12; % max number of iterations

transmit = rand(n,1);
transmit(transmit<0.5) = -1;
transmit(transmit>=0.5) = 1;
sigma = 0.001;

y = G*transmit+randn(n,1)*sigma;


epsilon=1e-70;
boundx=5;
%---------------
% compute signal node prior
%---------------
model_order = 512;
xx=1:model_order;  xx=xx-(model_order+1)/2;  xx=xx/max(xx);
xx=xx*boundx; % values over which pdf is sampled

%pdf_prior= normpdf(xx,-1,.7) + normpdf(xx,1,.7);
pdf_prior= normpdf(xx,0,1);
pdf_prior=pdf_prior/sum(pdf_prior);

noise_prior=normpdf(xx,0,sigma); %noise prior
noise_prior=noise_prior/sum(noise_prior);


disp ('STARTING THE DECODER...');

displayind = [ 1 2 3];

 [xrecon, mrecon, srecon]=NBP(G,transmit',y,...
     sigma, max_rounds, displayind ,epsilon,pdf_prior,noise_prior,xx);


fprintf('[   %d   %1d   %d   %d]\n', transmit(displayind));
fprintf('success=%6.2f \n',sum((((mrecon>0)*2)-1)==transmit')/n);

fprintf('GaBP success=%6.2f  \n',sum(transmit == sign(inv(G)*y))/n);


end