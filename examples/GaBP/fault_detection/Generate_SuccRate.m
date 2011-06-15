% Code written by Harel Avissar and Danny Bickson
% Suplamentary material for the paper: 
% Fault identification via non-parametric belief propagation.
% By D. Bickson, D. Baron, A. Ihler, H. Avissar, D. Dolev
% http://arxiv.org/abs/0908.2005
% In IEEE Tran on Signal Processing
% Code available from http://www.cs.cmu.edu/~bickson/gabp/

% code for creating figure 2,3,4,5

% if you want to compare to SDP, need to install CVX
%from  http://www.stanford.edu/~boyd/cvx/
addpath /afs/cs.cmu.edu/user/bickson/dolev/papers/gabp-src/fault_detection/cvx
addpath /afs/cs.cmu.edu/user/bickson/dolev/papers/gabp-src/fault_detection/cvx/structures
addpath /afs/cs.cmu.edu/user/bickson/dolev/papers/gabp-src/fault_detection/cvx/lib
addpath /afs/cs.cmu.edu/user/bickson/dolev/papers/gabp-src/fault_detection/cvx/functions
addpath /afs/cs.cmu.edu/user/bickson/dolev/papers/gabp-src/fault_detection/cvx/commands
addpath /afs/cs.cmu.edu/user/bickson/dolev/papers/gabp-src/fault_detection/cvx/builtins

%if you want to compare to discrete BP, 
%requires talya meltzer inference package to run
%package is found on: http://www.cs.huji.ac.il/~talyam/inference.html
addpath('../fault_det_plot/max-prod/');
addpath('../fault_det_plot/max-prod/c_inference/');

% if you want to compare to factor grpah BP,
% requires libDAI obtainable from
% http://people.kyb.tuebingen.mpg.de/jorism/libDAI/
addpath('../libDAI/matlab');

clear all;
avgT = 0;
totalR = 0;
LOFlag = 1;% local optimization flag, 0=off, 1 = on
K = 10;
n = 100;  % number of hidden variables
m = 50; % number of observations

Slevels = 1; %sparsity levels to check
Sstart = 1;% loop over Sstart:Slevels
Elevels = 5; %error levels to check
Estart = 1; % loop over Estart:Elevels
fault_step = 0.03;
test = 1000; %number of tests to average - be careful here 1000 runs may be slow!

algs = 11; %number of algorithems compare
sigma=1;%noise level
params = 25; %number of parameters to monitor on each alg
PTIME = 1; %run time
PSUCCRATE = 2; %success rate
PITERATIONS = 3; %number of iterations
PMAPGTR = 4; %success rate of larger likelihood than true sol
PRANK = 5; % ranking of solution 
PLOROUND = 6; % number of local optimization rounds 
PEXACT = 7; %number of exact solutions found 
PTOP = 8; % number of times on top 
PSUCCRATELO = 9; % sucess rate with local optimization
PFULLSOLUTION = 10; %records the full solution 
PFULLSOLUTIONLO = 11; %records the full solution with lo
POSX = 12;% records the original fault pattern X
PMAPGTRLO = 13;% number of times the local optimization has higher likelihood than true solution
POSLX = 14; % likelihood of solution x
POSLSOL = 15; %likelihood of solution
POSLSOLO = 16; %likelihood of solution with local optimization
POSP = 17; %probability of fault
POSQ = 18; %sparsity of the matrix A
POSSIGMA= 19; % noise level
POSM = 20; % size of A
POSN = 21; % size of A
PSEED = 22; % random seed
POSPREPORTED = 23; % value of p reported to the application
PLOTIME = 24;% local optimization time
PFRACTIONAL = 25; %fractional solution computed before transformation to binary/bipolar values

INBP = 1; IIP = 2; ICoSaMP = 3; IGPSR = 4; IhIO = 5; ICSBP = 6; INON = 7; ILDLC = 8; ISDP = 9; IMRF = 10; IFACTORBP = 11; %index for each alg
names = {'NBP','IP','CoSaMP','GPSR','hardIO','CS','NON','LDLC','SDP','Max-prod','BP-Factor'};
colors = ['b', 'r', 'c', 'm', 'k', 'g', 'y', 'b', 'r', 'g','k'];
ticks={'o','*','+','h','d','s','.','^','p','*','o'}; 
Lines = {'-','-','-','-','-','-','-','-','--','--','+'};

% boolean flag decides which algorithms to run
dorun = zeros(1,algs);
%dorun(1:10)=1;
dorun(ISDP)=1;

disp('running with parameters');
disp(['fault step: ', num2str(fault_step)]);
disp(['num of tests: ', num2str(test)]);
disp(['sigma: ', num2str(sigma)]);
disp(['Local optimization flag: ', num2str(LOFlag)]);
disp(['Sparsity levels tested: ', num2str(Slevels)]);
disp(['Error levels tested: ', num2str(Elevels)]);
disp(['Do run vector is: ', num2str(dorun)]);

%initialize data structure to hold running results
results = cell(Elevels,Slevels,algs,params); %of comparisons
for er=Estart:Elevels
for sp=Sstart:Slevels
for t=1:algs
    for ipp=1:params
        results{er,sp,t,ipp} = 0; 
    end
end
end
end
solutions = zeros(n, algs); %at each round
best = zeros(1, algs); %at each round
iterations = zeros(1, algs); %at each round
times = zeros(1, algs); %at each round
errors = zeros(1, algs); %for all rounds,will be shown every end of loop

assert(sum(dorun) > 0);

for er=Estart:Elevels
    
for sp=Sstart:Slevels


q=0.2; % matrix A sparsity level
%q = 0.1+(sp-1)/10;%0.1;
%q = 0.05+(0.1*(sp-1));
p=0.03+(er-1)*fault_step; % fault signature sparsity
skipped = 0; % number of errors

disp(['p is: ', num2str(p), ' q is: ', num2str(q)]);


% for each test
for state=1:test

% create a random problem
    rand('state',state);
    randn('state',state);
    
    H = rand(m,n);
    
    H(H<1-q) = 0;
    H(H>1-q+q/2) = -1;
    H(H>1-q) = 1;
   
    A = 2*H;%0.5*H;

    x = rand(n,1); % transmit a random vector of {-1,1}
    x(x>1-p) = 1;
    x(x<=1-p) = -1;
    X = (x+1)/2;
  
      Noise = randn(m,1)*sigma;
    y = H*x+Noise;
    Y = y+0.5*A*ones(n,1);
    
    prn = x;
        
    %% NBP
	if (dorun(INBP))
        rand('state',state);
        randn('state',state);
    
        max_rounds = 8;%12; % max number of iterations
        epsilon = 1e-20; % 
        boundx = n*1.2*q; % quantization bounds
        if (max(abs(y))>boundx)
            boundx = 1.3*max(abs(y));
        end
        model_order = 729; % how many quantization points
        xx=1:model_order;  xx=xx-(model_order+1)/2;  xx=xx/max(xx);
        xx=xx*boundx; % values over which pdf is sampled
        pdf_prior= 2*p*normpdf(xx,1,.05) + (1-(2*p))*normpdf(xx,-1,.05);% TODO!!!

        pdf_prior=pdf_prior/sum(pdf_prior); %normalize prior
        noise_prior=normpdf(xx,0,sigma); %noise prior, a gaussian 
        noise_prior=noise_prior/sum(noise_prior); %normalize prior
        try
        tic;
        [xrecon, solutions(:,INBP), srecon, iterations(INBP),conNBP]=NBP(H,x',y,sigma, max_rounds, [1 2 3 7] ,epsilon,pdf_prior,noise_prior,xx,0,1);
        times(INBP) = toc;
        iterations(INBP) = iterations(INBP)-1;
        catch ME
            skipped = skipped+1;
            errors(INBP) = errors(INBP)+1;
            ME.stack
            %continue
        end
        %% NON
     
    end

    
    solutions(:,INON) = solutions(:,INBP)*0;
    times(INON) = 0;
    iterations(INON) = 0;
    
    %% NBP_CS
	if (dorun(ICSBP))
        rand('state',state);
        randn('state',state);
        max_rounds = 8;
        epsilon=1e-30;
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
        [xrecon, solutions(:,ICSBP), srecon, iterations(ICSBP),conCS]=NBP(H,x',y,sigma, max_rounds, [] ,epsilon,pdf_prior,noise_prior,xx,0,1);
        times(ICSBP) = toc;
        iterations(ICSBP) = iterations(ICSBP)-1;
        catch ME
            %skipped = skipped+1;
            solutions(:,ICSBP) = solutions(:,INBP)*0;
            times(ICSBP) = 0;
            iterations(ICSBP) = 0;
            errors(ICSBP) = errors(ICSBP)+1;
            ME.stack
        end
	end

    %% NBP_LDLC
    if (dorun(ILDLC))
        boundx=n*(1.2*q+2*p);
        model_order = 243;
        xx=1:model_order;  xx=xx-(model_order+1)/2;  xx=xx/max(xx);
        xx=xx*boundx; % values over which pdf is sampled
        pdf_prior= 0.5*normpdf(xx,1,.1) + 0.5*normpdf(xx,-1,.1);
        pdf_prior=pdf_prior/sum(pdf_prior);
        noise_prior=normpdf(xx,0,1); %noise prior
        noise_prior=noise_prior/sum(noise_prior);
        max_rounds=8;
        epsilon = 1e-30;
        try
        tic;
        [xrecon, solutions(:,ILDLC), srecon, iterations(ILDLC),conLDLC]=NBP(H,x',y,sigma, max_rounds, [] ,epsilon,pdf_prior,noise_prior,xx,0,1);
        times(ILDLC) = toc;
        iterations(ILDLC) = iterations(ILDLC)-1;
        catch ME
            skipped = skipped+1;
            solutions(:,ILDLC) = solutions(:,INBP)*0;
            times(ILDLC) = 0;
            iterations(ILDLC) = 0;
            errors(ILDLC) = errors(ILDLC)+1;
            ME
        end
    end
    
    %% ----------- ARMAP solution only ------------------------------------
    if (dorun(IIP))
        rand('state',state);
        randn('state',state);
    
        try
        tic;
        [sol,iterations(IIP),conIP] = armap_gabp(A,X,Y,p*ones(n,1),sigma,1/(2*n),0,'verbose',0);
        solutions(:,IIP) = 2*(sol-0.5);
        times(IIP) = toc;
        catch ME
            skipped = skipped+1;
            errors(IIP) = errors(IIP)+1;
            solutions(:,IIP) = solutions(:,INBP)*0;
            times(IIP) = 0;
            iterations(IIP) = 0;
 ME
            %continue
        end
    end
    %% ----------- CoSaMP ------------------------------------
    
    if (dorun(ICoSaMP))
        rand('state',state);
        randn('state',state);
    
        try
        tic;
        max_rounds = 100;
        [sol,iterations(ICoSaMP), conCoSaMP] = CoSaMP(Y,A,2*p*n,max_rounds,0);
        solutions(:,ICoSaMP) = 2*(sol-0.5);
        times(ICoSaMP) = toc;
        catch ME
            skipped = skipped+1;
            errors(ICoSaMP) = errors(ICoSaMP)+1;
             solutions(:,ICoSaMP) = solutions(:,INBP)*0;
            times(ICoSaMP) = 0;
            iterations(ICoSaMP) = 0;
            ME
            %continue
        end
    end
    
    %% ----------- GPSR ------------------------------------
    
    if (dorun(IGPSR))
        rand('state',state);
        randn('state',state);
    
        try
        tic;
        [sol,iterations(IGPSR), conGPSR] = GPSR_BB(Y,A,0.3,0);
        solutions(:,IGPSR) = 2*(sol-0.5);
        times(IGPSR) = toc;
        catch ME
            skipped = skipped+1;
            errors(IGPSR) = errors(IGPSR)+1;
             solutions(:,IGPSR) = solutions(:,INBP)*0;
            times(IGPSR) = 0;
            iterations(IGPSR) = 0;
%rethrow(ME);
            %continue
        end
    end
    %% ----------- hardIO ------------------------------------
    
    if (dorun(IhIO))
        rand('state',state);
        randn('state',state);
    
        try
        [sol,iterations(IhIO), conhIO] = hard_l0_Mterm(Y,A,n,2*n*p,0);
        solutions(:,IhIO) = 2*(sol-0.5);
        times(IhIO) = toc;
        catch ME
            skipped = skipped+1;
            errors(IhIO) = errors(IhIO)+1;
             solutions(:,IhIO) = solutions(:,INBP)*0;
            times(IhIO) = 0;
            iterations(IhIO) = 0;
ME
            %continue
        end
    end
    
    
    %% MRF ---- max product belief propagation
    if (dorun(IMRF))
        rand('state',state);
        randn('state',state);
    
        lambda = log((1-p)/p)*ones(n,1);
        %ad = H'*H;
        ad=A'*A;


        la = cell(n,n);
        lo = cell(1,n);
        for in1=1:n
            for in2=1:n
                if (in1~=in2 && ad(in1,in2)~=0)
                    phi_ij = ad(in1,in2)/(sigma*sigma);
                    la{in1,in2} = [1,1;1,exp(-phi_ij)];%/sum(sum([1,1;1,exp(-phiij)]));
                end
            end
            %tmp = H'*y/(sigma*sigma);
            tmp = A'*Y/(sigma^2);
            phi_ii = ad(in1,in1)/(2*sigma*sigma)+(log((1-p)/p)-tmp(in1));
            lo{1,in1} = [1;exp(-phi_ii)]/(1+exp(-phi_ii));
            ad(in1,in1) = 0;
        end
        rho=ad/10;
        ad = ad.*(ones(n,n)-eye(n,n));
        ad(ad>0) = 1;
        ad(ad<0) = 1;
        %ad = ones(n,n)-eye(n);
        tic;
        %[bel, iterations(IMRF)] = inference(ad,la,lo,'loopy','sum_or_max', 1, 'trw',1);%,'trw',1);
        % if this fails, you probably did not download and install Talya
        % Meltzer inference package from http://www.cs.huji.ac.il/~talyam/inference.html
        [bel, iterations(IMRF)] = inference(ad,la,lo,'loopy','max_iter', 50);
        for in1=1:n
            %bel{in1}
            solutions(in1,IMRF) = bel{in1}(2);
        end
        solutions(:,IMRF) = 2*(solutions(:,IMRF)-0.5);
         times(IMRF) = toc;
    end
    
    
    %% cvx for SDP
    if (dorun(ISDP))
        rand('state',state);
        randn('state',state);
    
        tic;
        %if this code fails, you did not install CVX software
        cvx_quiet(true);
        lambda = log((1-p)/p)*ones(n,1);
         cvx_begin
             variable g(n)
             minimize((1/(2*sigma^2))*square_pos(norm(A*g-Y,2))+lambda'*g)
             g >= 0;
             g <= 1;
         cvx_end
         solutions(:,ISDP) = 2*(g-0.5);
         iterations(ISDP) = 0;
         times(ISDP) = toc;
    end
    
    % factor graph belief propagation
    if (dorun(IFACTORBP))
        rand('state',state);
        randn('state',state);
    
       lambda = log((1-p)/p)*ones(n,1);
       %A - is the linear relation matrix
       %x - is the solution
       %y - is the observation
       Nv=size(H,2); Nf=size(H,1);
       psi=cell(1,Nf+Nv); Edges=sparse([],[],[],Nv+Nf,Nv+Nf,round(4*Nv*Nf*q)); ii=1;
       for i=1:Nf,
         vars = find(H(i,:)); nvars=length(vars); sizes=2*ones(1,nvars);
         tmp.Member=vars-1;
         fprintf('Creating factor with %d vars\n',nvars);
         if (nvars>20) fprintf('Are you sure? (paused)\n'); continue; end;
         Ptable = zeros(sizes);
         for j=1:numel(Ptable)                          % fill in the table of likelihoods
           idx=cell(1,nvars); [idx{:}]=ind2sub(sizes,j); idx=2*[idx{:}]-3;
           xtmp=zeros(Nv,1); xtmp(vars)=idx;            % for each config, eval signature
           Ptable(j) = -((y(i)-H(i,:)*xtmp)/sigma).^2;  % and p(yi | expected signature)
         end;
         if (nvars)                                     % if it's an empty factor drop it
           Ptable = Ptable-max(Ptable(:));              % otherwise, normalize it a bit
           tmp.P=exp(Ptable); psi{ii}=tmp; ii=ii+1;     % and store it as a factor
         end;
       end;
       for i=1:Nv,                                      % add local, unary factors
         tmp.Member=i-1;
         tmp.P=exp([log(1-p);log(p)]); psi{ii}=tmp; ii=ii+1;
       end;
       psi=psi(1:ii-1);
       fprintf('Running BP...\n');
       tic,
       [lZ,qbp,mdbp,qvbp,qfbp]=dai(psi, 'BP', '[inference=SUMPROD,updates=SEQMAX,tol=1e-9,maxiter=20,logdomain=0]');
       bels=zeros(2,Nv); for i=1:Nv, bels(:,i)=qvbp{i}.P; end;
       iterations(IFACTORBP)=20;                        % not really correct
       solutions(:,IFACTORBP) = 2*(bels(2,:)'-0.5);     % go from probability to bipolar representation
       times(IFACTORBP) = toc;                          % save time values


        
    end
    
    
%% all went well, we can enter results of this round
for t=1:algs
    if (dorun(t))
        results{er,sp,t,PTIME} = results{er,sp,t,1}+times(t);% time
        results{er,sp,t,PITERATIONS} = results{er,sp,t,PITERATIONS}+iterations(t); %number of iterations
        tic;
        [xamb1 lamb1 xamb2 lamb2 rounds] = Round_and_Local(solutions(:, t),K,H,p,n,y,sigma,LOFlag);
        time = toc;
        results{er,sp,t,PLOTIME} = results{er,sp,t,PLOTIME}+time;% time
        avgT = avgT+time;
        totalR = totalR+rounds;
        results{er,sp,t,PLOROUND} = results{er,sp,t,PLOROUND}+rounds; %number of rounds
      
        if (state ==1)
            results{er,sp,t,PFULLSOLUTION}= xamb1(:,1);
            results{er,sp,t,PFULLSOLUTIONLO}= xamb2(:,1);    
            results{er,sp,t,POSX} = x;
            results{er,sp,t,PFRACTIONAL} = solutions(:,t);
        else
            results{er,sp,t,PFULLSOLUTION}= [results{er,sp,t,PFULLSOLUTION} xamb1(:,1)];
            results{er,sp,t,PFULLSOLUTIONLO}= [ results{er,sp,t,PFULLSOLUTIONLO} xamb2(:,1)];    
            results{er,sp,t,POSX} = [results{er,sp,t,POSX} x];
            results{er,sp,t,PFRACTIONAL} = [results{er,sp,t,PFRACTIONAL} solutions(:,t)];
        end
        if x== xamb1(:,1) %exact result obtained withut local optimization
            results{er,sp,t,PSUCCRATE} = results{er,sp,t,PSUCCRATE}+1;
            results{er,sp,t,PEXACT} = results{er,sp,t,PEXACT}+1; % exact counter
        end;
        if x== xamb2(:,1) % local optimization found exact result
            results{er,sp,t,PSUCCRATELO} = results{er,sp,t,PSUCCRATELO}+1;
        end;
        if (state == 1)
            results{er,sp,t,POSLX} = L(H, p, n, x, y, sigma);
            results{er,sp,t,POSLSOL} = L(H, p, n, xamb1(:,1), y, sigma);
            results{er,sp,t,POSLSOLO} = L(H, p, n, xamb2(:,1), y, sigma);
        else
            results{er,sp,t,POSLX} = [results{er,sp,t,POSLX} L(H, p, n, x, y, sigma)];
            results{er,sp,t,POSLSOL} = [ results{er,sp,t,POSLSOL}  L(H, p, n, xamb1(:,1), y, sigma)];
            results{er,sp,t,POSLSOLO} = [results{er,sp,t,POSLSOLO} L(H, p, n, xamb2(:,1), y, sigma)];
        end
        if L(H, p, n, x, y, sigma)>L(H, p, n, xamb1(:,1), y, sigma)
            %results{er,sp,t,PSUCCRATE} = results{er,sp,t,PSUCCRATE}+1;
            results{er,sp,t,PMAPGTR} = results{er,sp,t,PMAPGTR}+1;
        end;
        if L(H, p, n, x, y, sigma)>L(H, p, n, xamb2(:,1), y, sigma)
            %results{er,sp,t,PSUCCRATELO} = results{er,sp,t,PSUCCRATELO}+1;
             results{er,sp,t,PMAPGTRLO} = results{er,sp,t,PMAPGTRLO}+1;
        end;
        results{er, sp, t, POSSIGMA} = sigma;
        results{er, sp, t, POSQ} = q;
        results{er, sp, t, POSP} = p;
        results{er, sp, t, POSM} = m;
        results{er, sp, t, POSN} = n;
        results{er, sp, t, PSEED} = state;
        results{er, sp, t, POSPREPORTED} = p;
        best(t) = lamb1(1);
    end;
end


%% Calc relative rating of algs
for t=1:algs
    results{er,sp,t,PRANK} = results{er,sp,t,PRANK}+sum(unique(sort(best))<=best(t));
    if (sum(unique(sort(best))<=best(t))==1)
        results{er,sp,t,PTOP} = results{er,sp,t,PTOP}+1;
    end
end

end

%% print this rounds (er+sp) results
test = test-skipped;
fprintf('error level     %d     sp level     %d     failed     %d  \n',er,sp,skipped);
for t=1:algs
if (dorun(t))
   fprintf('%s :Time- %7.4f     succRate- %6.2f   succRateLO %6.2f  Iter- %6.2f  MAP- %d  Place- %7.4f  LO- %6.2f GTR- %7.4f GTRLO - %6.2f LOTIME = %6.2f \n', ...
       names{t}, results{er,sp,t,1}/test, results{er,sp,t,2}/test, results{er,sp,t,PSUCCRATELO}/test, ...
       results{er,sp,t,3}/test,results{er,sp,t,4},results{er,sp,t,5}/test,results{er,sp,t,6}/test,...
       results{er,sp,t,PMAPGTR}/test,results{er,sp,t,PMAPGTRLO}/test, results{er,sp,t,PLOTIME}/test);
end
end

if (sum(errors) > 0)
fprintf('errors: ');
for t=1:algs
   fprintf('%s: %d  ', names{t},errors(t));    
end
fprintf('\n');
end

if (skipped > 0)
   disp(['skipped: ', num2str(skipped)]);
end
errors = zeros(1, algs);
if skipped>=10
    %assert(false);
   disp('warning: more than 10 assertions!');
end

end
end

if LOFlag
    fprintf('Average time per LO round: %7.4f\n',avgT/totalR);
end
 
token = date(); 
 a1=clock();  
 matfilename = sprintf('/mnt/bigbrofs/usr7/bickson/%sresults%d-%d-%d.mat', token,a1(4),a1(5),a1(6));
 save(matfilename,'results','token','names','dorun');
 matfilename2 = sprintf('/mnt/bigbrofs/usr6/bickson/%sresults%d-%d-%d.mat', token,a1(4),a1(5),a1(6));
 save(matfilename2,'results','token','names','dorun');
 disp(['saved results file to', matfilename]);
 disp(['saved results file to', matfilename2]);

