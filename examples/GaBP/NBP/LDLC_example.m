%Code written by Danny Bickson
%Code is found on: http://www.cs.huji.ac.il/labs/danss/p2p/gabp/index.html
%Example of solving boolean least squares via NBP
%The problem is argmin_x ( norm (Hx - y))  s.t. x \in {-1,1}
% G is 121 x 121 matrix. Ulike ldlc G is the sparse ENCODING matrix (and not the decoding)

function [] = LDLC_example()
clear
rand('state',sum(100*clock));
randn('state',sum(100*clock));

load('H_n_121_d_3_1.mat','H');
G = H;
n=size(H,1);

max_rounds = 12; % max number of iterations
epsilon = 1e-70; % 
x = rand(n,1); % transmit a random vector of {-1,1}
x(x<0.5) = -1;
x(x>=0.5) = 1;
sigma = 0.000038;

y = G*x+randn(n,1)*sigma;
boundx=100;
%---------------
% compute signal node prior
%---------------
model_order = 1543;
xx=1:model_order;  xx=xx-(model_order+1)/2;  xx=xx/max(xx);
xx=xx*boundx; % values over which pdf is sampled

pdf_prior= normpdf(xx,-1,.2) + normpdf(xx,1,.2);
%pdf_prior= normpdf(xx,0,1);
pdf_prior=pdf_prior/sum(pdf_prior);

noise_prior=normpdf(xx,0,1); %noise prior
noise_prior=noise_prior/sum(noise_prior);


disp ('STARTING THE DECODER...');
 [xrecon, mrecon, srecon]=NBP(G,x',y,...
     sigma, max_rounds, [ 1 10 21 24] ,...
     epsilon,pdf_prior,noise_prior,xx);


fprintf('[   %d   %1d   %d   %d]\n', x([1 10 21 24]));
fprintf('success=%6.4f \n',sum((((mrecon>0)*2)-1)==x')/121);

fprintf('GaBP success=%6.24  \n',sum(x == sign(inv(G)*y))/121);

end