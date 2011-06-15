% Original code by Liang Xiong, CMU
% Code cleaned by Danny Bickson, CMU

% Inputs:
% training - training data (see test_ALS for format)
% testing - test data
% D - dimension of factorized matrices U and V
% maxIter - maximum number of iterations

% Output:
% Factorized matrices U,V, s.t. facorized matrix A =~ U'V
function [U, V, objective]=ALS(training, testing, D, maxIter, regularizationU, regularizationV)

M=training.size(1);
N=training.size(2);

subs=training.subs;
vals=training.vals;
L=length(vals);


%check the index
check(min(subs) >= 1);
check(max(subs) <= [M N]);

fprintf('ALS for matrix (%d, %d):%d.\n', M, N, L);
fprintf('regularizationU=%g, regularizationV=%g\n', regularizationU, regularizationV);

fprintf('Initialization...');

%U=ones(D, M)*0.1; %TODO
U=ones(D,M)*0.1;
%for i=1:M
%    U(
%end
V=ones(D, N)*0.1;

te=~isempty(testing);

if te
  subsTe=testing.subs;
  valsTe=testing.vals;
  LTe=length(valsTe);
  clear testing;
  
  yTe=predict(subsTe, U, V);
  rmseTe=rmse(yTe - valsTe);
else
  rmseTe=nan;
end

yTr=predict(subs, U, V);
rmseTr=rmse(yTr - vals);

fprintf('complete. RMSE=%0.4f/%0.4f.\n', rmseTr, rmseTe);

fprintf('Pre-calculating the index...');
subU=subs(:, 1);
subV=subs(:, 2);
fprintf('U');Usub=GroupIndex(subU);
fprintf('V');Vsub=GroupIndex(subV);
fprintf('. complete.\n');


eDU=eye(D);
eDV=eye(D);

tic;
for iter=1:maxIter
  fprintf('-Iter%d... ', iter);
  
  fprintf('U');
  for ind=1:M
    filter=Usub{ind};
    Q=V(:, subV(filter));
    U(:, ind)=(Q*Q' + eDU*regularizationU*length(filter))\(Q*vals(filter));
  end
  
  fprintf('V');
  for jnd=1:N
    filter=Vsub{jnd};
    Q=U(:, subU(filter));
    V(:, jnd)=(Q*Q' + eDV*regularizationV*length(filter))\(Q*vals(filter));
  end
  
  yTr=predict(subs, U, V);
  res=yTr - vals;
  objective=sum(res.^2) + regularizationU*sum(U(:).^2) + regularizationV*sum(V(:).^2);
  objective=objective*0.5;
  rmseTr=rmse(res);
  
  if te
    yTe=predict(subsTe, U, V);
    rmseTe=rmse(yTe - valsTe);
  else
    rmseTe=nan;
  end
  
  fprintf('. objective=%g, RMSE=%0.4f/%0.4f. Time to finish=%0.2fhr.\n', objective, ...
          rmseTr, rmseTe, (maxIter - iter)*toc/iter/3600);


end

function [y]=predict(subs, U, V)
    for i=1:size(subs,1)
        y(i,1) = U(:,subs(i,1))'*V(:,subs(i,2));
    end
end

function [gI]=GroupIndex(I)
    I=I(:);
    n=max(I);
    gI=group(uint32(I), 1:length(I), n);
end

function [r]=rmse(err)
    r=sqrt(mean(err.^2));
end
end
