function indextimetest(nreps)
% INDEXTIMETEST : compare timings of matrix indexing with sprefm and A(I,J)
%
% indextimetest(nreps)  does nreps(i) (default 100) reps of test #i
% 
% John R. Gilbert, 5 Sep 2010

if nargin < 1
    nreps = 100;
end;

bigscale = 17;
smallscale = 10;
ntests = 0;

fprintf('\nComparing times for A(I,J) and sprefm(A,I,J)\n\n');
tstart = tic;

ntests = ntests+1;
nr = nreps(min(ntests,length(nreps)));
fprintf('Test %d: Single element (%d reps)\n',ntests,nr);
t0 = tic;
Abig = rmat(bigscale);
nbig = length(Abig);
tg = toc(t0);
fprintf('  Generate %6d-vertex Rmat: %5.2f sec\n',nbig,tg);
t0 = tic;
for i = 1:nr;
    r = ceil(nbig*rand);
    c = ceil(nbig*rand);
end;
tl = toc(t0);
fprintf('  Loop and random overhead:    %5.2f sec\n',tl);
t0 = tic;
for i = 1:nr;
    r = ceil(nbig*rand);
    c = ceil(nbig*rand);
    a = Abig(r,c);
end;
ti = toc(t0);
fprintf('  Native indexing:             %5.2f sec\n',ti);
t0 = tic;
for i = 1:nr;
    r = ceil(nbig*rand);
    c = ceil(nbig*rand);
    a = sprefm(Abig,r,c);
end;
tm = toc(t0);
fprintf('  Indexing with spgemm:        %5.2f sec\n',tm);
ratio1 = (tm-tl)/(ti-tl);
fprintf('  Ratio spgemm / native:  %5.2f\n',ratio1);

ntests = ntests+1;
nr = nreps(min(ntests,length(nreps)));
fprintf('Test %d: Single row (%d reps)\n',ntests,nr);
t0 = tic;
for i = 1:nr;
    r = ceil(nbig*rand);
end;
tl = toc(t0);
fprintf('  Loop and random overhead:    %5.2f sec\n',tl);
t0 = tic;
for i = 1:nr;
    r = ceil(nbig*rand);
    a = Abig(r,:);
end;
ti = toc(t0);
fprintf('  Native indexing:             %5.2f sec\n',ti);
t0 = tic;
for i = 1:nr;
    r = ceil(nbig*rand);
    c = 1:nbig;
    a = sprefm(Abig,r,c);
end;
tm = toc(t0);
fprintf('  Indexing with spgemm:        %5.2f sec\n',tm);
ratio2 = (tm-tl)/(ti-tl);
fprintf('  Ratio spgemm / native:  %5.2f\n',ratio2);

ntests = ntests+1;
nr = nreps(min(ntests,length(nreps)));
fprintf('Test %d: Single column (%d reps)\n',ntests,nr);
t0 = tic;
for i = 1:nr;
    c = ceil(nbig*rand);
end;
tl = toc(t0);
fprintf('  Loop and random overhead:    %5.2f sec\n',tl);
t0 = tic;
for i = 1:nr;
    c = ceil(nbig*rand);
    a = Abig(:,c);
end;
ti = toc(t0);
fprintf('  Native indexing:             %5.2f sec\n',ti);
t0 = tic;
for i = 1:nr;
    r = 1:nbig;
    c = ceil(nbig*rand);
    a = sprefm(Abig,r,c);
end;
tm = toc(t0);
fprintf('  Indexing with spgemm:        %5.2f sec\n',tm);
ratio3 = (tm-tl)/(ti-tl);
fprintf('  Ratio spgemm / native:  %5.2f\n',ratio3);

ntests = ntests+1;
nr = nreps(min(ntests,length(nreps)));
fprintf('Test %d: Random 1000-by-1000 submatrix of big Rmat (%d reps)\n',ntests,nr);
t0 = tic;
for i = 1:nr;
    R = ceil(nbig*rand(1,1000));
    C = ceil(nbig*rand(1,1000));
end;
tl = toc(t0);
fprintf('  Loop and random overhead:    %5.2f sec\n',tl);
t0 = tic;
for i = 1:nr;
    R = ceil(nbig*rand(1,1000));
    C = ceil(nbig*rand(1,1000));
    A = Abig(R,C);
end;
ti = toc(t0);
fprintf('  Native indexing:             %5.2f sec\n',ti);
t0 = tic;
for i = 1:nr;
    R = ceil(nbig*rand(1,1000));
    C = ceil(nbig*rand(1,1000));;
    A = sprefm(Abig,R,C);
end;
tm = toc(t0);
fprintf('  Indexing with spgemm:        %5.2f sec\n',tm);
ratio4 = (tm-tl)/(ti-tl);
fprintf('  Ratio spgemm / native:  %5.2f\n',ratio4);

ntests = ntests+1;
nr = nreps(min(ntests,length(nreps)));
fprintf('Test %d: Random 1000-by-1000 submatrix of small Rmat (%d reps)\n',ntests,nr);
t0 = tic;
Asmall = rmat(smallscale);
nsmall = length(Asmall);
tg = toc(t0);
fprintf('  Generate %6d-vertex Rmat: %5.2f sec\n',nsmall,tg);
t0 = tic;
for i = 1:nr;
    R = ceil(nsmall*rand(1,1000));
    C = ceil(nsmall*rand(1,1000));
end;
tl = toc(t0);
fprintf('  Loop and random overhead:    %5.2f sec\n',tl);
t0 = tic;
for i = 1:nr;
    R = ceil(nsmall*rand(1,1000));
    C = ceil(nsmall*rand(1,1000));
    A = Asmall(R,C);
end;
ti = toc(t0);
fprintf('  Native indexing:             %5.2f sec\n',ti);
t0 = tic;
for i = 1:nr;
    R = ceil(nsmall*rand(1,1000));
    C = ceil(nsmall*rand(1,1000));
    A = sprefm(Asmall,R,C);
end;
tm = toc(t0);
fprintf('  Indexing with spgemm:        %5.2f sec\n',tm);
ratio5 = (tm-tl)/(ti-tl);
fprintf('  Ratio spgemm / native:  %5.2f\n',ratio5);

fprintf('Summary of execution time ratios, spgemm / native indexing\n')
fprintf('  Single element: %7.2f\n',ratio1);
fprintf('  Single row:     %7.2f\n',ratio2);
fprintf('  Single column:  %7.2f\n',ratio3);
fprintf('  Submatrix 1:    %7.2f\n',ratio4);
fprintf('  Submatrix 2:    %7.2f\n',ratio5);
fprintf('\nTotal elapsed time: %5.2f sec\n\n',toc(tstart));