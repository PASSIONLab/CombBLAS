%Code written by Harel Avissar and Danny Bickson
%Suplamentary material for the paper: 
%Distributed fault identification via non-parametric belief propagation.
%By D. Bickson, H. Avissar, D. Dolev, S. P. Boyd, A. Ihler and D. Baron.
%Submitted for publication. Aug 2009.
%Code available from http://www.cs.huji.ac.il/labs/danss/p2p/gabp/

%Generates figure 4 in the above paper
function Generate_NBPvIP()
clear
n = 15;%15;
m = 10;%10;
Slevels = 5; %sparcity levels to check
Elevels = 2; %error levels to check
Sstart = 5;
Estart = 2;
algs = 8; %number of algorithems compare
% params are : 1- time      2- success rate         3- iterations to
% converge      4- number of times when found MAP solution better than real
% solution      5- ranking of solution
% alg INON is not here since its not very interesting
INBP = 1; IIP = 2; ICoSaMP = 3; IGPSR = 4; IhIO = 5; ICS = 6;ILDLC = 8; %index for each alg
solutions = zeros(n, algs); %at each round
iterations = zeros(1, algs); %at each round
times = zeros(1, algs); %at each round
errors = zeros(1, algs); %for all rounds,will be shown every end of loop
rand('state',76);%plot - 76
randn('state',1);
paintNBP = false;%true;
paintNBPvCoSaMP = false;
paintNBPvhIO = false;
paintNBPvGPSR = false;
paintNBPvIP = true;
paintNBPvLDLC = false;
paintNBPvCS = false;

for er=Estart:Elevels
    
for sp=Sstart:Slevels
    
p = 0.05*er;%0.1
q = sp/10;%0.1;
skipped = 0;
test = 1;
for state=1:test
    H = rand(m,n);
    H(H<1-q) = 0;
    H(H>1-q+q/2) = -1;
    H(H>1-q) = 1;
    A = 2*H;

    max_rounds = 12; % max number of iterations
    epsilon = 1e-20; % 
    x = rand(n,1); % transmit a random vector of {-1,1}
    x(x>1-p) = 1;
    x(x<=1-p) = -1;
    X = (x+1)/2;
    %x(x<0.5) = -1;
    %x(x>=0.5) = 1;
    sigma = 1;%0.4;
    Noise = randn(m,1)*sigma;

    y = H*x+Noise;
    Y = y+0.5*A*ones(n,1);
    
    prn = x;
    str = sprintf('[ %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f]', prn([ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]));
    disp(str);
    fprintf('\n');
    
    %% NBP
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
    [xrecon, solutions(:,INBP), srecon, iterations(INBP),conNBP]=NBP(H,x',y,sigma, max_rounds, [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15] ,epsilon,pdf_prior,noise_prior,xx,1,1);
    times(INBP) = toc;
    iterations(INBP) = iterations(INBP)-1;
    catch ME
        skipped = skipped+1;
        errors(INBP) = errors(INBP)+1;
        rethrow(ME);
        continue
    end    
    fprintf('\n');
    if paintNBP
        pos = sum(conNBP(:,iterations(INBP))>0);
        neg = sum(conNBP(:,iterations(INBP))<0);
        posDelta = ceil(255/pos);
        negDelta = ceil(255/neg);
        viewingDelta = 0.06;
        [sortNBP, indNBP] = sort(conNBP(:,iterations(INBP)),'descend');
        curX = [0:iterations(INBP)  (iterations(INBP) +0.84)];
        hold on;
        set(gca,'XTick',0:1:(iterations(INBP)+1));
        set(gca,'YTick',[-1.5 -1 0 1 1.5]);
        grid on;
        xlabel('Iteration number');
        ylabel('Xi Value');
        title('Plot of converging NBP solution');
        for index=1:pos
            curY = [0 conNBP(indNBP(index),:) conNBP(indNBP(index),iterations(INBP))];
            curColor = ceil((index-1+0.9)*posDelta);
            name = sprintf('X%d',indNBP(index));
            plot(curX,curY,'-','LineWidth',2,'Color',[0 0 curColor/255],'DisplayName',name);
            curX(iterations(INBP)+2)=curX(iterations(INBP)+2)-viewingDelta;
        end
        curX = [0:iterations(INBP)  (iterations(INBP))];
        for index=1:neg
            curY = [0 conNBP(indNBP(index+pos),:) conNBP(indNBP(index+pos),iterations(INBP))];
            curColor = 255-ceil((index-1+0.5)*negDelta);
            name = sprintf('X%d',indNBP(index+pos));
            plot(curX,curY,'-','LineWidth',2,'Color',[curColor/255 0 0],'DisplayName',name);
            curX(iterations(INBP)+2)=curX(iterations(INBP)+2)+viewingDelta;
        end
        plot([0 0],[-1.5 1.5],'-b');
        
    end
    %% NBP_CS
    boundx=n*(1.2*q+2*p);
    model_order = 243;
    xx=1:model_order;  xx=xx-(model_order+1)/2;  xx=xx/max(xx);
    xx=xx*boundx; % values over which pdf is sampled
    pdf_prior= p*normpdf(xx,-1,2) + (1-p)*normpdf(xx,-1,.1);
    pdf_prior=pdf_prior/sum(pdf_prior);
    noise_prior=normpdf(xx,0,1); %noise prior
    noise_prior=noise_prior/sum(noise_prior);
    try
    tic;
    [xrecon, solutions(:,ICS), srecon, iterations(ICS),conCS]=NBP(H,x',y,sigma, max_rounds, [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15] ,epsilon,pdf_prior,noise_prior,xx,1,0.1);
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
    fprintf('\n');
    if paintNBPvCS
        PlotComp(conCS, iterations(ICS), 'CS', conNBP, iterations(INBP), 'NBP');
    end
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
    [xrecon, solutions(:,ILDLC), srecon, iterations(ILDLC),conLDLC]=NBP(H,x',y,sigma, max_rounds, [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15] ,epsilon,pdf_prior,noise_prior,xx,1,0.1);
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
    fprintf('\n');
    if paintNBPvLDLC
        PlotComp(conLDLC, iterations(ILDLC), 'LDLC', conNBP, iterations(INBP), 'NBP');
    end
    %% ----------- ARMAP solution only ------------------------------------
    try
    tic;
    [sol,iterations(IIP),conIP] = armap_gabp(A,X,Y,p*ones(n,1),sigma,1/(2*n),0,'verbose',1);
    solutions(:,IIP) = 2*(sol-0.5);
    times(IIP) = toc;
    catch ME
        skipped = skipped+1;
        errors(IIP) = errors(IIP)+1;
        continue
    end
    fprintf('\n');
    if paintNBPvIP
        PlotComp(conIP, iterations(IIP), 'IP', conNBP, iterations(INBP), 'NBP');
    end
    %% ----------- CoSaMP ------------------------------------
    try
    tic;
    [sol,iterations(ICoSaMP), conCoSaMP] = CoSaMP(Y,A,p*n,max_rounds,1);
    solutions(:,ICoSaMP) = 2*(sol-0.5);
    times(ICoSaMP) = toc;
    catch ME
        skipped = skipped+1;
        errors(ICoSaMP) = errors(ICoSaMP)+1;
        continue
    end
    fprintf('\n');
    if paintNBPvCoSaMP
        PlotComp(conCoSaMP, iterations(ICoSaMP), 'CoSaMP', conNBP, iterations(INBP), 'NBP');
    end
    %% ----------- GPSR ------------------------------------
    try
    tic;
    [sol,iterations(IGPSR), conGPSR] = GPSR_BB(Y,A,0.3,1);
    solutions(:,IGPSR) = 2*(sol-0.5);
    times(IGPSR) = toc;
    catch ME
        skipped = skipped+1;
        errors(IGPSR) = errors(IGPSR)+1;
        rethrow(ME);
        %continue
    end
    fprintf('\n');
    if paintNBPvGPSR
        PlotComp(conGPSR, iterations(IGPSR), 'GPSR', conNBP, iterations(INBP), 'NBP');
    end
    %% ----------- hardIO ------------------------------------
    try
    [sol,iterations(IhIO), conhIO] = hard_l0_Mterm(Y,A,n,n*p,1);
    solutions(:,IhIO) = 2*(sol-0.5);
    times(IhIO) = toc;
    catch ME
        skipped = skipped+1;
        errors(IhIO) = errors(IhIO)+1;
        continue
    end
    fprintf('\n');
    if paintNBPvhIO
        PlotComp(conhIO, iterations(IhIO), 'hardIO', conNBP, iterations(INBP), 'NBP');
    end

end

end
end


%% this function plots a comparison of two convergence sets.
%%% it assumes left converged to right solution and plots right against it. 
%%% all number values are correct for the n=15 examples, must be
%%% recalculated otherwise!
function PlotComp(conRight, iterRight, nameRight, conLeft, iterLeft, nameLeft)
        pos = sum(conLeft(:,iterLeft)>0);
        neg = sum(conLeft(:,iterLeft)<0);
        posDelta = ceil(255/pos);
        negDelta = ceil(255/neg);
        viewingDelta = 0.06;
        [sortLeft, indLeft] = sort(conLeft(:,iterLeft),'descend');
        curX = [- (iterLeft+0.84) -iterLeft:iterRight (iterRight +0.84)];
        hold on;
        set(gca,'XTick',-(iterLeft+1):1:iterRight+1);
        set(gca,'YTick',[-1.5 -1 0 1 1.5]);
        grid on;
        xlabel('Iteration number');
        ylabel('Xi Value');
        tit = sprintf('Plot of converging %s vs %s solution', nameLeft, nameRight);
        title(tit);
        for index=1:pos
            curY = [conLeft(indLeft(index),iterLeft) conLeft(indLeft(index),iterLeft:-1:1) 0 conRight(indLeft(index),:) conRight(indLeft(index),iterRight)];
            curColor = ceil((index-1+0.9)*posDelta);
            name = sprintf('X%d',indLeft(index));
            plot(curX,curY,'-','LineWidth',2,'Color',[0 0 curColor/255],'DisplayName',name);
            curX(1)=curX(1)+viewingDelta;
            curX(iterLeft+iterRight+3)=curX(iterLeft+iterRight+3)-viewingDelta;
        end
        curX = [- (iterLeft) -iterLeft:iterRight (iterRight)];
        for index=1:neg
            curY = [conLeft(indLeft(index+pos),iterLeft) conLeft(indLeft(index+pos),iterLeft:-1:1) 0 conRight(indLeft(index+pos),:) conRight(indLeft(index+pos),iterRight)];
            curColor = 255-ceil((index-1+0.5)*negDelta);
            name = sprintf('X%d',indLeft(index+pos));
            plot(curX,curY,'-','LineWidth',2,'Color',[curColor/255 0 0],'DisplayName',name);
            curX(1)=curX(1)-viewingDelta;
            curX(iterLeft+iterRight+3)=curX(iterLeft+iterRight+3)+viewingDelta;
        end
        for index=1:neg
            curY = [conLeft(indLeft(n-index+1),iterLeft) conLeft(indLeft(n-index+1),iterLeft:-1:1) 0 conRight(indLeft(n-index+1),:) conRight(indLeft(n-index+1),iterRight)];
            curColor = ceil((index-1+0.5)*negDelta);
            name = sprintf('X%d',indLeft(n-index+1));
            hobj=line(curX,curY,'LineWidth',2,'Color',[curColor/255 0 0],'DisplayName',name);
            hAnnotation = get(hobj,'Annotation');
            hLegendEntry = get(hAnnotation','LegendInformation');
            set(hLegendEntry,'IconDisplayStyle','off');
            curX(1)=curX(1)+viewingDelta;
            curX(iterLeft+iterRight+3)=curX(iterLeft+iterRight+3)-viewingDelta;
        end
end

%fprintf('GaBP success=%6.24  \n',sum(x == sign(inv(H)*y))/121);
end