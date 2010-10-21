% PARALLELRMATSCRIPT : Script to simulate parallel RMAT
%
% This compares two ways of building an RMAT graph:
%      First, build it all at once (as if on one processor)
%      Second, build np graphs of 1/np the edge density
%          and put their triples together (as if on np processors)
%
% This test assumes np divdes edgefactor*2^scale; 
% for the real code, generate ceil(edgefactor*2^scale/np) triples 
% on each processor but only use edgefactor*2^scale of them in all.
%
% John Gilbert, 20 Oct 2010

edgefactor = 16;
scale = 15;
np = 128;

nv = 2^scale;
ne = edgefactor * nv;

% The official way
IJ1 = kronecker_generator(scale,edgefactor);
A1 = sparse(IJ1(1,:)+1,IJ1(2,:)+1,ones(1,ne),nv,nv);

% The multinode way
IJ2 = zeros(2,ne);
for p = 1:np
    myrange = ((p-1)*ne/np) + (1:ne/np);
    IJ2(:,myrange) = kronecker_generator(scale,edgefactor/np);
end;
A2 = sparse(IJ2(1,:)+1,IJ2(2,:)+1,ones(1,ne),nv,nv);

% Qualitative comparison of matrices
figure; spy(A1); title('Single-node matrix');
figure; spy(A2); title(sprintf('%d-node matrix',np));
figure;
plot(sum(A1),sum(A2),'b.',sum(A1,2),sum(A2,2),'r.');
legend('column counts','row counts','Location','northwest'); 
title('Correlation between matrices');
xlabel('Single-node matrix'); ylabel('Multi-node matrix');
figure; 
subplot(2,1,1); 
s = full(sum(A1));
top = ceil(log10(nv));
es = 0:top/20:top;
N = histc(log10(s),es);
plot(es,log10(N),'.');
title('Single-node matrix column counts');
subplot(2,1,2); 
s = full(sum(A2));
N = histc(log10(s),es);
plot(es,log10(N),'.');
title('Multi-node matrix column counts');

