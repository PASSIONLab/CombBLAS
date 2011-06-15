%% Code for testing Binary Least Squares problem
% Written By Danny Bickson
% The linear model is Gx+v=y
% The algorithm computes argmin_x ||y-Gx||_2^2 s.t. x \in {-1,1}
% Matrix G is of size 961x961.
% Note that unlike LDLC paper by Sommer et al., G is a sparse ENCODING matrix

%%


function [] = LDLC_large_example()
clear
rand('state',sum(100*clock));
randn('state',sum(100*clock));

%load('H_n_121_d_3_1.mat','H');
load('H_n_961_d_7.mat','H');
G = H;
n=size(G,1);

%addpath('../CSBP_matlab');
max_rounds = 20; % max number of iterations
epsilon = 1e-70; % 
x = rand(n,1);
x(x<0.5) = -1;
x(x>=0.5) = 1;
sigma = 0.000038;

y = G*x+randn(n,1)*sigma;
boundx=8;
%---------------
% compute signal node prior
%---------------
model_order = 255;
xx=1:model_order;  xx=xx-(model_order+1)/2;  xx=xx/max(xx);
xx=xx*boundx; % values over which pdf is sampled

%pdf_prior= normpdf(xx,-0,.15) + normpdf(xx,2.25,.15);
pdf_prior= normpdf(xx,-1.25,.15) + normpdf(xx,1.25,.15);
%pdf_prior= normpdf(xx,0,1);
pdf_prior=pdf_prior./sum(pdf_prior);

noise_prior=normpdf(xx,0,1); %noise prior
noise_prior=noise_prior./sum(noise_prior);


disp ('STARTING THE DECODER...');

 [xrecon, mrecon]=NBP(G,x',y,...
     sigma, max_rounds, [ 1 10 21 24] ,...
     epsilon,pdf_prior,noise_prior,xx);


fprintf('[   %d   %1d   %d   %d]\n', x([1 10 21 24]));
fprintf('NBP success=%6.4f \n',sum((((mrecon>0)*2)-1)==x')/n);

fprintf('Linear detection success=%6.24  \n',sum(x == sign(inv(G)*y))/n);


end