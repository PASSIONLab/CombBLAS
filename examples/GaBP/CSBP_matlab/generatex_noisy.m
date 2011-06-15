function [x, heavyind]=generatex_noisy(N, K, sig_signal, sig_noise);

%Generates a noisy version of a K-sparse signal
x=zeros(1,N);
x(1:K)=sig_signal*randn(1,K);
x(K+1:N)=sig_noise*randn(1,N-K);
ind=randperm(N);
x=x(ind);
heavyind=find(ind<=K);
