function writematrix(A,name)

[i,j,k] = find(A);
fid = fopen(name, 'w+');
fprintf(fid, '%d\t%d\t%d\n',size(A,1), size(A,2), nnz(A)); 
for m = 1:nnz(A)
	fprintf(fid, '%d\t%d\t%.3f\n',i(m),j(m),k(m)); 
end
fclose(fid);

