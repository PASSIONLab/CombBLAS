% main.m
% computes BER and MMSE bounds for range of epsilon, 
% where m, n, q, and sigma are fixed.
% Dror, 1/27/2011

clear

% parameters
q=0.1;  % percentage nonzeros in matrix 
m=250;  % number of measurements
n=500;  % length of signal
sigma=4;    % factor by which we multiply noise

snr=q*m*4/sigma^2;  % Danny's {-1,+1} input multiplies snr 4X

eps_all=0.01:0.01:0.2; % range of epsilon
L=length(eps_all);
ber_all=zeros(L,1);
mmse_all=zeros(L,1);

% main loop
for ind=1:L
    epsilon=eps_all(ind);   % current epsilon
    mu=m/(n*epsilon);       % measurements per active element
    eta=efficiency_db(epsilon,mu,snr) % degradation
    ber_all (ind)=ber(epsilon,eta*snr);
    mmse_all(ind)=mmse(epsilon,eta*snr);
end

% printouts
for ind=1:L
    fprintf('Eps=%6.3f, BER = %10.7f, MMSE = %10.7f\n',eps_all(ind),ber_all(ind),mmse_all(ind));
end

