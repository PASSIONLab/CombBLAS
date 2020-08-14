% for titan
% class = 'ER' or  'G500' or 'SSCA'
% scale = number of rows/cols = 2^scale
function batchGenRop(maxCore, mtxfile)

nameParts = strsplit(mtxfile, '/');
nameParts = strsplit(nameParts{end},'.');
mtxname = nameParts{1};

fileName = sprintf('%s/batchRop_%s_%d',mtxname, mtxname, maxCore);
fileID = fopen(fileName,'w');
fprintf(fileID,'#PBS -q debug\n');
fprintf(fileID,'#PBS -l mppwidth=%d\n', maxCore);
fprintf(fileID,'#PBS -l walltime=00:30:00\n');
fprintf(fileID,'#PBS -N Rop_%s_%d\n', mtxname, maxCore);
fprintf(fileID,'#PBS -j oe\n\n');

fprintf(fileID, 'cd $PBS_O_WORKDIR\n');
fprintf(fileID,'export MPICH_GNI_COLL_OPT_OFF=MPI_Alltoallv\n');
fprintf(fileID, 'IN=%s\n',mtxfile);
% machine specifc infp 
% for edison

layers = [1,2,4,8,16];
threads = [1,3,6,12];
coresPerNode = 24;
coresPerSocket = 12;

for t = threads
    fprintf(fileID, '\nexport OMP_NUM_THREADS=%d\n', t);
    if(t==coresPerSocket) 
        cc = 'numa_node';
    else
        cc = 'depth';
    end
        
    for c = layers
        dim1 = floor(sqrt(maxCore/(t*c)));
        dim2 = dim1;
        ncores = dim1*dim2*c*t;
        nprocs = dim1*dim2*c;
        N = min(nprocs, coresPerNode/t);
        S = min(nprocs, coresPerSocket/t);
        if(dim1>=1)
            if(t<=coresPerSocket)
                fprintf(fileID,'aprun -n %d -d %d -N %d -S %d -cc %s ../../RestrictionOp %d %d %d input $IN\n', nprocs, t, N, S, cc, dim1, dim2, c);
            else
                fprintf(fileID,'aprun -n %d -d %d -N %d ../../RestrictionOp %d %d %d input $IN\n', nprocs, t, N, dim1, dim2, c);
        %fprintf(fileID,'%d\t %d\t %d\t %d\t %d\t %d\t\n', ncores, nprocs, dim1, dim2, c, t);
            end
        end
    end
end

fclose(fileID);

%cores = 0:15;
%cores = 2.^cores;
%for core = cores
%    batchGenRop(core, '/scratch2/scratchdirs/azad/spGEMM_matrices/nlpkkt160.mtx');
%end
    