function [xx,iter, convergence] = CoSaMP(uu, PP, s, Its, show);
% CoSaMP Compressive Sampling (??) Matching Pursuit
%--- 
% USAGE  xx= CoSaMP(uu, PP, s)
%
%           uu              : compressive samples of a signal
%           PP              : sampling matrix
%           s               : sparsity of xx
%
%
%   Output vector xx is an s-sparse approximation of the signal 
%   that generated the compressive samples in uu via
%               uu= PP x ss+ nn
%   where uu: (M x 1), PP: (M x N), and nn: (N x 1). 
%
%
%
%     -see also Joel Tropp 
%
%--------------------------------------------------------------------
% 2/4/2008, by Volkan Cevher. volkan@rice.edu. For FUN. -Rice. 
%--------------------------------------------------------------------
% NOTES: 
%       1. First version. The sampling operator does not support functions.
%
%--------------------------------------------------------------------
if nargin~=5, error('Incorrect number of inputs!!!'); end
%---
uu= uu(:);
M= length(uu);
%---
[Mt,N]= size(PP); if or(ne(Mt,M),ne(Mt,M)); error('Inconsistent sampling matrix!!!'); end
% I am not checking the transpose cases (yet).
%---
% Its= s;
aa= zeros(N,Its);
conv = zeros(N,Its);
% aa= zeros(N,1);
% yy= aa;
PPt= PP';
%---Main
kk=1;
% maxiter= 1000;
 maxiter= 20;
iternum = 0;
verbose= 0;
%tol= 1e-3;
tol= 0.1;
% rr= uu;
% normrr= inf;
% normrr1= norm(rr);
% while  and(and(le(kk,Its), (normrr1/norm(uu))> 1e-2),
% le(normrr1,normrr)),tic
%tic;
while  le(kk,Its),
%     normrr= normrr1;
    rr= uu- PP*aa(:,kk);
%     normrr1= norm(rr);
%     rr= uu- PP*aa;
  yy= PPt*rr;
    %---
%     yyi=yy;
%     yyi(aa(:,kk)>0)= aa(aa(:,kk)>0,kk)';
%     %---
%     ymax= max(abs(yyi));
%     L1= zeros(size(yyi));
%     L2= zeros(size(yyi));
%     L1(yyi>=thr)= 1;
%     L1(yyi<thr)= exp(log(2)/thr*yyi(yyi<thr))-1;
%     L2(yyi<thr)= 1;
%     L2(yyi>=thr)= exp(- 3/(ymax-thr)*(yyi(yyi>=thr)-thr));
%     
%     
%     
%     gch = GraphCut('open', Dc, 10*Sc); %, exp(-Vc*5), exp(-Hc*5));
%     [gch L] = GraphCut('expand',gch);   
%     gch = GraphCut('close', gch);
% 
%     
%     
    %---
%    [tmp,ww]= sort(abs(yy),'descend');
% db doesnt have "descend"
[tmp,ww]= sort(abs(yy));
tmp=flipud(tmp);
ww=flipud(ww);
%     [tmp,ww]= sort((yy),'descend');
    tt= union(find(ne(aa(:,kk),0)),ww(1:(2*s)));
    
%     tt= union(find(ne(aa,0)),ww(1:(2*s)));
    %---
%     bb= zeros(N,1);
%     bb(tt)= pinv(PP(:,tt))*uu;
    %----
    [x, res, it] = cgsolve(PP(:,tt)'*PP(:,tt), PP(:,tt)'*uu, tol, maxiter, verbose);
    iternum = iternum + 1;
    bb= zeros(N,1);
    bb(tt)= x;
    %---
%    [tmp,ww2]= sort(abs(bb),'descend');
% db as before
[tmp,ww2]= sort(abs(bb));
tmp=flipud(tmp);
ww2=flipud(ww2);
%     [tmp,ww2]= sort((bb),'descend');
    %---
    kk= kk+1;
    aa(ww2(1:s),kk)= bb(ww2(1:s));
    prn = 2*(aa(:,kk)-0.5);
    conv(:,kk-1) = prn;    
    str = sprintf('[ %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f]', prn([ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]));
    if show
        disp(str);
    end
%     figure(kk)
%     stem(aa(:,kk))
%     hold on
%     stem(gt,'*')
%     hold off
%     aa(ww(1:s))= tmp(1:s);

% db removes vc printout
%[kk,toc,norm(rr)]

%% additional check to stop when no change is made
if kk>1
    if norm(aa(:,kk-1)-aa(:,kk))<tol
        break;
    end
end

end
iter = iternum;
xx= aa(:,kk);
convergence = conv(:,1:iter);
%xx(:,kk:end)=[];