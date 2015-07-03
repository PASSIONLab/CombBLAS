
function  multwrite(fileA, fileB, fileC)
% Reads two matrix market files, multiply the sparse matrices and 
% writes the resultant sparse matrix into another file in a matrix market
% format
A  = mmread(fileA);
B  = mmread(fileB);
C = A * B;
mmwrite(fileC, C);
