%%Code written by Danny Bickson & Harel Avissar
%Code is found on: http://www.cs.huji.ac.il/labs/danss/p2p/gabp/index.html
%Supplamentary material for the paper:
%Low density lattice decoding via non-parametric belief propagation. D.
%Bickson, A. Ihler, H. Avissar and D. Dolev. Submitted to Allerton 2009.
%Paper available on: http://arxiv.org/abs/0901.3197

%% This script tests the NBP decoder with different levels of noise
%  On code length n=1000 with d=9 non zeros on each row and column
%  The problem solved is argmin_b norm( Gb - y ), b \in [-1,1].
clear
addpath('../NBP');

load n1000d9.mat G;
n=size(G,1);
d=unique(sum(abs(G)));
assert(length(d) == 1);
detG=det(G)
max_sigma_squared = 4*nthroot(detG, n/2)/(2*pi*exp(1)); %Potyrov equation as given in
% G. Poltyrev, “On coding without restrictions for the AWGN channel,” in
% IEEE Trans. Inform. Theory, vol. 40, Mar. 1994, pp. 409–417.
disp(['max sigma^2 for channel capacity is ', num2str(max_sigma_squared)]);

retry=10;
max_instance = 250;
max_rounds = 5; % max number of iterations

sigmas=fliplr([max_sigma_squared*0.25 max_sigma_squared*0.95]);
success = zeros(1, retry);

epsilon = 1e-30; % 
boundx=15;


model_order = 729;%1543;
xx=1:model_order;  xx=xx-(model_order+1)/2;  xx=xx/max(xx);
xx=xx*boundx; % values over which pdf is sampled

pdf_prior= normpdf(xx,-1,.2)+normpdf(xx,1,.2);
%pdf_prior= normpdf(xx,0,1);
pdf_prior=pdf_prior/sum(pdf_prior);

noise_prior=normpdf(xx,0,1); %noise prior
noise_prior=noise_prior/sum(noise_prior);

for loop_count=1:retry
    rand('state',loop_count);
    randn('state',loop_count);
    total = 0;
    for instance=1:max_instance
       
       x = rand(n,1); % transmit a random vector of {-1,1}
        x(x<0.5) = -1;
  	    x(x>=0.5) = 1;
        sigma = sigmas(loop_count);

        y = G*x+randn(n,1)*sqrt(sigma);

        disp (['STARTING THE DECODER... with sigma ', num2str(sigma), ' distance from capcity ' , num2str(-10*log10(sigma/max_sigma_squared)), ' dB']);
        tic
        [xrecon, mrecon, srecon]=NBP_opt(G,x',y,...
             sigma, max_rounds, [ 1 10 21 24] ,...
             epsilon,pdf_prior,noise_prior,xx);
        toc
        fprintf('[   %d   %1d   %d   %d]\n', x([1 10 21 24]));
        sucrate = sum((((mrecon>0)*2)-1)==x')/n;
        fprintf('success=%6.4f \n',sucrate);
        total = total + sucrate;
    end
    total = total/max_instance;
    disp(['Avg performance with ', num2str(sigma),' is ', num2str(total)]);
    success(loop_count) = total;
end
save ret.mat sigmas success;
%end
