function [] = main_gabp()
%-------------------
% main.m
% Main file for running compressed sensing via Belief Propagation
% Full original code is found on: http://www.ece.rice.edu/~drorb/CSBP/
%-------------------
% Original code by Shriram Sarvotham, 2006. 
% Cleaned up a bit by Dror Baron, November 2008.
%-------------------
% Decoder rewritten by Danny Bickson, March 2009.
% This code improves the perofrmance of the previous code by Dror Baron.
% You can test both implementations by uncommenting a line in the bottom.

clear
rand('state',sum(100*clock));
randn('state',sum(100*clock));
addpath('../CSBP_matlab');
%-------------------
% Signal parameters
%-------------------
%Signal and noise parameters
n=200;  %Signal length
k=20;	%Sparsity
SNR=100;	% input snr
sigma_1=sqrt(SNR);
sigma_0=1; % small signal coefficients
sigma_Z=1; % noise in the measurements y (noisy measurements)

%-------------------
% CS-LDPC matrix
%-------------------
l=20; % constant row weight
r=10; % constant column weight

%-------------------
% CS-BP parameters
%-------------------
gamma_mdbpf=0.35; %Damping for Belief Prop
gamma_mdbpb=0.35;
gamma_pdbp=0.0;
iter=10; % Number of iterations in Belief Prop
p=243; % Number of sampling points (FFT runs decently for this value)

%-------------------
% Generate signal
%-------------------
t1=cputime;
disp ('GENERATING THE SIGNAL X...');
[x, heavyind]=generatex_noisy(n, k, sigma_1, sigma_0);
x=(x/norm(x))*sqrt(k);
x=sigma_1*x;
disp(sprintf('l2 norm of x: %g', norm(x) )); 
  
%-------------------
% Generate measurement matrix
%-------------------
[phi]=gen_phi(n, l, r);
phisign=randn(size(phi));  phisign=sign(phisign);


%-------------------
% Run driver, which decodes the signal
%-------------------

[A]=phi_to_A(phi,phisign, n);

%-------------------
% create auxiliary data structures
%-------------------
aux_rows=l; 
[aux, aux_rows_actual]=get_aux(phi, phisign, n, l, aux_rows);
[self_indexN,self_indexM]=GetSelfIndices(phi, aux);   %Used in BP
[tmptmp, dispind]=sort(-abs(x));
dispind=[dispind(1:5), 1:5];
  
%-------------------
% Encode (compute measurements)
%-------------------
disp ('GENERATING THE MEASUREMENTS...');
measvec=encoder(phi, phisign, x);
disp(sprintf('Number of measurements=%d', length(measvec)));
measvec=measvec+sigma_Z*randn(length(measvec),1);	% add noise

epsilon=1e-70;
boundx=sigma_1*10;
%---------------
% compute signal node prior
%---------------
model_order = p;
xx=1:model_order;  xx=xx-(model_order+1)/2;  xx=xx/max(xx);
xx=xx*boundx; % values over which pdf is sampled

 pdf_prior= (k/n)*normpdf(xx,0,sigma_1);
 if (sigma_0 > epsilon)
   pdf_prior= pdf_prior  +    (1-k/n)*normpdf(xx,0,sigma_0);
end
pdf_prior=pdf_prior/sum(pdf_prior);

noise_prior=normpdf(xx,0,sigma_Z);	% noise prior
noise_prior=noise_prior/sum(noise_prior);

%-------------------
% Decode
%-------------------
disp ('STARTING THE DECODER...');
disp('Last column = norm(cur_sol - real_sol)');
disp('Column before last = norm(A*cur_sol - observation)');
% D. Bickson: older code - you can uncomment the below line if you want to test
% the NBP vs. CSBP algo.
% Download the full code from http://www.ece.rice.edu/~drorb/CSBP/
% addpath() the directory where you put the code
%   [xrecon, mrecon, srecon, pdf, pdf_xx]=decoder(x,measvec,n,k,l,max(aux_rows_actual),...
%       sigma_1,sigma_0,sigma_Z, iter, phi, phisign, aux,p,self_indexN, self_indexM, dispind,...
%       boundx, epsilon, gamma_pdbp, gamma_mdbpf, gamma_mdbpb,xx);

 [xrecon, mrecon, srecon]=NBP(A,x,measvec,sigma_Z, iter, dispind,epsilon,pdf_prior,noise_prior,xx);



function [A] = phi_to_A(phi,phisign,n)
    [n1,k1] = size(phi);
    A = sparse(n1,k1);
    for i=1:n1
        for j=1:k1
            A(i,phi(i,j)) = phisign(i,j);
        end
    end
end


end
