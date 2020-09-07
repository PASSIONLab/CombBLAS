% class = 'ER' or  'G500' or 'SSCA'
% scale = number of rows/cols = 2^scale
function batchGen(maxCore, class, scale)

fileName = sprintf('batch_%s_%d_%d', class, scale, maxCore);
fileID = fopen(fileName,'w');
fprintf(fileID,'#PBS -q debug\n');
fprintf(fileID,'#PBS -l mppwidth=%d\n', maxCore);
fprintf(fileID,'#PBS -l walltime=00:30:00\n');
fprintf(fileID,'#PBS -N spGEMMexp_%s_%d_%d\n', class, scale, maxCore);
fprintf(fileID,'#PBS -j oe\n');
fprintf(fileID,'cd $PBS_O_WORKDIR\n');



% problem specific stats
%scale = 24;
%class = 'ER';
deg = 16;


% machine specifc infp 
% for edision
layers = [1,2,4,8,12,16];
threads = [1,3,6,12];
coresPerNode = 24;
coresPerSocket = 12;

for t = threads
    fprintf(fileID, '\nexport OMP_NUM_THREADS=%d\n', t);
    if(t==12) 
        cc = 'numa_node';
    else
        cc = 'depth';
    end
        
    for c = layers
        dim1 = floor(sqrt(maxCore/(t*c)));
        dim2 = dim1;
        ncores = dim1*dim2*c*t;
        nprocs = dim1*dim2*c;
        N = coresPerNode/t;
        S = coresPerSocket/t;
        fprintf(fileID,'aprun -n %d -d %d -N %d -S %d -cc %s ./mpipspgemm %d %d %d %s %d %d column\n', nprocs, t, N, S, cc, dim1, dim2, c, class, scale, deg);
        %fprintf(fileID,'%d\t %d\t %d\t %d\t %d\t %d\t\n', ncores, nprocs, dim1, dim2, c, t);
    end
end

fclose(fileID);
    