function genRMatBetw(scale, dim)
%%
% Create an RMAT matrix (8 nonzeros per column) with given scale - in parallel !
% Partition it to a grid of dimensions (dim) x (dim)
% Write each submatrix into a file
% Written by : Aydin BULUC. Jan 22, 2008
% This is a new version that chops of the remainders of the last block 
%   row instead of calling a different generator for each grid
% Best for speedup experiments where sqrt(p) does not divide the 
%   matrix dimension
% Avoids memory problems by not forming the output at all and generating
%   matrices one at a time.
% Modified: November 13, 2008
%   Now generates square grids of the form {1x1}, {2x2},...,{dim x dim)
%%



G1 = rmat(scale*p, true);
A = grsparse(G1);
AT = A';				% transpose the graph
AT = logical(mod(AT,8) > 0);		% filter 1/8 of the edges

fprintf('Dimensions %d %d\n', ppfront(size(A,1)), ppfront(size(A,2)));


gridsize = 1;
while gridsize <= dim
    dimx = floor(ppfront(size(A,1))/ gridsize);
    dimy = floor(ppfront(size(A,2))/ gridsize);   
    % Grid dimensions does not need to divide the matrix dimensions evenly 
    
    pdimx = 1;
    pdimy = 1;
    k = 0;
    for i = 1:gridsize
        for j = 1:gridsize
            dirname = ['p', num2str(gridsize*gridsize), '/proc',num2str(k)];
            if exist(dirname, 'dir') ~= 0
                rmdir(dirname,'s')
            end
            mkdir(dirname);
            cd(dirname);

            subA = A(pdimx:pdimx+dimx-1, pdimy:pdimy+dimy-1);
            [i1,j1,k1] = find(ppfront(subA));

            fid = fopen(['input1_',num2str(k)], 'w+');
            fprintf(fid, '%d\t%d\t%d\n',size(subA,1), size(subA,2), nnz(subA)); 
            for m = 1:nnz(subA)
                fprintf(fid, '%d\t%d\t%d\n',i1(m),j1(m),k1(m)); 
            end

            fclose(fid);
        
            cd('../..');
            pdimy = pdimy + dimy;
            k = k+1;
        end
        pdimx = pdimx + dimx;
        pdimy = 1;
    end
    gridsize = gridsize * 2;
end

