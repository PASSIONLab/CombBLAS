% Code written by Harel Avissar and Danny Bickson
% Suplamentary material for the paper: 
% Fault identification via non-parametric belief propagation.
% By D. Bickson, D. Baron, A. Ihler, H. Avissar, D. Dolev
% In IEEE Tran of Signal Processing.
% http://arxiv.org/abs/0908.2005
% Code available from http://www.cs.cmu.edu/~bickson/gabp/

%generates figure 6 in the above paper
function [res] = Generate_2D()
clear
avgT = 0;
totalR = 0;
LOFlag = 1;
K = 10;
n = 50;%15;
m = 30;%10;
Slevels = 2; %sparcity levels to check
Elevels = 1; %error levels to check
Sstart = 2;
Estart = 1;
algs = 8; %number of algorithems compare
% params are : 1- time      2- success rate         3- iterations to
% converge      4- number of times when found MAP solution better than real
% solution      5- ranking of solution          6- number of LO rounds
% 7- number of exact solution           8- number of times on top
INBP = 1; IIP = 2; ICoSaMP = 3; IGPSR = 4; IhIO = 5; ICS = 6; INON = 7; ILDLC = 8;%index for each alg
solutions = zeros(n, algs); %at each round
iterations = zeros(1, algs); %at each round
times = zeros(1, algs); %at each round
errors = zeros(1, algs); %for all rounds,will be shown every end of loop
%rand('state',76);%plot - 76
rand('state',1);
randn('state',1);

for er=Estart:Elevels
    
for sp=Sstart:Slevels
    
p = 0.05*er;%0.1
q = sp/10;%0.1;
skipped = 0;
test =1;
%tic;
for state=1:test
%load('H_n_121_d_3_1.mat','H');
    %%%H = rand(m,n);
    H = rand(m,n);
    %Hm = randn(m,n)*0.25+1;
    H(H<1-q) = 0;
    H(H>1-q+q/2) = -1;
    H(H>1-q) = 1;
    %H = H.*Hm;
    A = 2*H;%0.5*H;

    x = rand(n,1); % transmit a random vector of {-1,1}
    x(x>1-p) = 1;
    x(x<=1-p) = -1;
    [sortP, indP] = sort(x, 'descend');
    X = (x+1)/2;
    %x(x<0.5) = -1;
    %x(x>=0.5) = 1;
    sigma = 1;%0.4;
    Noise = randn(m,1)*sigma;

    y = H*x+Noise;
    Y = y+0.5*A*ones(n,1);
    
    prn = x;
    str = sprintf('Five biggest entries of x:[%d %d %d %d %d]', indP(1:5));
    disp(str);
    fprintf('Their values and corresponding convergence:\n');
    str = sprintf('[ %7.2f %7.2f %7.2f %7.2f %7.2f]', prn(indP(1:5)));
    disp(str);
    fprintf('NBP:\n');
    
    %% NBP
    max_rounds = 12;%12; % max number of iterations
    epsilon = 1e-20; % 
    boundx=n*(1.2*q+2*p);%n*q*2;%(1.2*q+2*p);
    model_order = 243;
    xx=1:model_order;  xx=xx-(model_order+1)/2;  xx=xx/max(xx);
    xx=xx*boundx; % values over which pdf is sampled
    pdf_prior= p*normpdf(xx,1,.1) + (1-p)*normpdf(xx,-1,.1);
    pdf_prior=pdf_prior/sum(pdf_prior);
    noise_prior=normpdf(xx,0,1); %noise prior
    noise_prior=noise_prior/sum(noise_prior);
    try
    tic;
    [xrecon, solutions(:,INBP), srecon, iterations(INBP),conNBP]=NBP(H,x',y,sigma, max_rounds, indP(1:5) ,epsilon,pdf_prior,noise_prior,xx,1,.1);
    times(INBP) = toc;
    iterations(INBP) = iterations(INBP)-1;
    catch ME
        skipped = skipped+1;
        errors(INBP) = errors(INBP)+1;
        %rethrow(ME);
        continue
    end
    fprintf('CS:\n');
    %% NON
    solutions(:,ICS) = solutions(:,INBP)*0;
    times(ICS) = 0;
    iterations(ICS) = 0;
    %% NBP_CS
    boundx=n*(1.2*q+2*p);
    model_order = 243;
    xx=1:model_order;  xx=xx-(model_order+1)/2;  xx=xx/max(xx);
    xx=xx*boundx; % values over which pdf is sampled
    pdf_prior= p*normpdf(xx,-1,3) + (1-p)*normpdf(xx,-1,.1);
    pdf_prior=pdf_prior/sum(pdf_prior);
    noise_prior=normpdf(xx,0,1); %noise prior
    noise_prior=noise_prior/sum(noise_prior);
    try
    tic;
    [xrecon, solutions(:,ICS), srecon, iterations(ICS),conCS]=NBP(H,x',y,sigma, max_rounds, indP(1:5) ,epsilon,pdf_prior,noise_prior,xx,1,.1);
    times(ICS) = toc;
    iterations(ICS) = iterations(ICS)-1;
    catch ME
        %skipped = skipped+1;
        solutions(:,ICS) = solutions(:,INBP)*0;
        times(ICS) = 0;
        iterations(ICS) = 0;
        errors(ICS) = errors(ICS)+1;
        %continue
    end
    fprintf('LDLC:\n');
    %% NBP_LDLC
    boundx=n*(1.2*q+2*p);
    model_order = 243;
    xx=1:model_order;  xx=xx-(model_order+1)/2;  xx=xx/max(xx);
    xx=xx*boundx; % values over which pdf is sampled
    pdf_prior= 0.5*normpdf(xx,1,.1) + 0.5*normpdf(xx,-1,.1);
    pdf_prior=pdf_prior/sum(pdf_prior);
    noise_prior=normpdf(xx,0,1); %noise prior
    noise_prior=noise_prior/sum(noise_prior);
    try
    tic;
    [xrecon, solutions(:,ILDLC), srecon, iterations(ILDLC),conLDLC]=NBP(H,x',y,sigma, max_rounds, indP(1:5) ,epsilon,pdf_prior,noise_prior,xx,1,.1);
    times(ILDLC) = toc;
    iterations(ILDLC) = iterations(ILDLC)-1;
    catch ME
        %skipped = skipped+1;
        solutions(:,ILDLC) = solutions(:,INBP)*0;
        times(ILDLC) = 0;
        iterations(ILDLC) = 0;
        errors(ILDLC) = errors(ILDLC)+1;
        %continue
    end
    fprintf('IP:\n');
  
end
end
end
figure;
plot([1], [1], 'd','MarkerSize', 10, 'MarkerEdgeColor', 'k','DisplayName','Solution','LineWidth',2);%, 'MarkerFaceColor', 'k');
hold on;
grid on;
plot(conNBP(indP(1),:), conNBP(indP(2),:), '-*b','LineWidth',2,'MarkerSize', 10, 'DisplayName','NBP');
plot(conIP(indP(1),:), conIP(indP(2),:), '-xr','LineWidth',2,'MarkerSize', 10, 'DisplayName','IP');
plot(conCS(indP(1),:), conCS(indP(2),:), '-+y','LineWidth',2,'MarkerSize', 10, 'DisplayName','CS');
plot(conLDLC(indP(1),:), conLDLC(indP(2),:),'-og','MarkerSize', 10, 'LineWidth',2,'DisplayName','LDLC');
legend('Solution', 'NBP','IP','CSBP','LDLC');
axis([-0.2 1.6 -0.8 1.1]);
box('on');


end