function rstats(scales)
% RSTATS : collect stats about RMAT graphs of given sizes
%
% rstats(scales);
%
% John Gilbert, 27 Sep 2010

nb = 200;                % batch size to average over
maxd = 20;               % max distance to go out
fringe = zeros(1,maxd);  % size of fringe at given distance
fprintf('\nRMAT statistics at different scales\n');
fprintf('\nScale     Nv       Ne Diam     d1    d2    d3    d4    d5    d6    d7    d8    d9   d10\n')
for k = scales
    A = rmat(k);
    nv = length(A);
    ne = nnz(A);
    B = sparse(ceil(nv*rand(1,nb)),1:nb,1,nv,nb);
    for d = 1:maxd
        BB = A*B | B;
        F = BB - B;
        B = BB;
        fringe(d) = nnz(F);
        if fringe(d)==0
            break
        end;
    end;
    if fringe(d)==0
        nf = d-1;
        diam = d-1;
    else
        nf = d;
        diam = nan;
    end;
    
    fprintf('  %3d %6d %8d %4d ',k,nv,ne,diam);
    fprintf('%6d',round(fringe(1:nf)/nb));
    fprintf('\n');
end;