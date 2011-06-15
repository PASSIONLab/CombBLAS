% Function that demonstrates how to fix the convergence of the GaBP
% algorithm, in the multiuser detection secnario

% Algorithm is described in the paper:
% "Fixing the convergence of the GaBP algorithm" by
% J. K. Johnson, D. Bickson and D. Dolev
% In ISIT 2009
% http://arxiv.org/abs/0901.4192
% Code written by Danny Bickson
% December 2008
function [] = multiuser_detection_fix()

rand('state',sum(100*clock));
randn('state',sum(100*clock));

addpath('..');

N=256; %number of chips per symbol
K=64; %number of active users.
A=eye(K); %assuming equal power for now.
dB=-10; %noise level
epsilon_outer_loop = 1e-10; %convergence epsilon for outer loop

sigma2=1; % from dB=10*log10(1/sigma2)
b= rand(K,1); %transmitted bits
b(b < 0.5) = -1;
b(b >= 0.5) = 1;

S = rand(N,K); %random spreading CDMA 
S(S < 0.5) = -1;
S(S >= 0.5) = 1;
S = S/sqrt(N);


n=sqrt(sigma2)*randn(N,1); %AWGN
r=S*A*b+n; %received signal
y=S'*r; %matched filter output
        
M=S'*S + sigma2*eye(K);% correlation matrix + regularization
C_N = eye(K) - M;
C_N = C_N/diag(diag(C_N));
a11 = eig(abs(C_N));
disp(['spectral radius is: ' , num2str(max(abs(a11)))]);

b_est2 = inv(M)*y;
inv_err=sum(b~=(b_est2 > 0))/K;
disp(['err using direct inversion is: ', num2str(inv_err)]);

xj = zeros(K,100);
xj(:,1) = y;

diagonal_loading = max(sum(abs(M)) - max(diag(M)));
Minc = M + eye(length(M)) *  diagonal_loading;
disp(['diagonal loading is ' num2str( diagonal_loading) ]);
disp(['condition number is ' num2str( cond(inv(diag(diag(Minc)))*Minc - eye(K))) ]);
a11 = eig(eye(length(M)) - Minc*inv(diag(diag(Minc))));
disp(['spectral radius is: ' , num2str(max(abs(a11)))]);

norms = ones(1,100);
for l=2:100
    % This is a single Newton step
    % Algorithm is described in
    % Linear Detection via Belief Propagation. Danny Bickson, Danny Dolev, Ori
    % Shental, Paul H. Siegel and Jack K. Wolf. In the 45th Annual Allerton Conference on Communication, Control, 
    % and Computing, Allerton House, Illinois, Sept. 07.
    [b_est,J,r1] = asynch_GBP(Minc, y - M*xj(:,l-1), 30, 1e-6);
    xj(:,l) = xj(:,l-1) + b_est';
    e = norm(y - M*xj(:,l));
    disp(['error norm for round ', num2str(l), ' is ', num2str(e)]);
    norms(l) = e;
    if (e < epsilon_outer_loop)
        outer = l;
        break;
     end
end

       
figure;
semilogy(3:l, norms(3:l));
title('Convergence of fixed GaBP iteration with n=256,k=64','FontSize',16);
xlabel('Newton step','FontSize',14);
ylabel('Error norm','FontSize',14);



end
