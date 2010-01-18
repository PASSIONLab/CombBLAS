function writerowvector(v,name)

[i,j,k] = find(v);
fid = fopen(name, 'w+');
fprintf(fid, '%d\t%d\n',size(v,1), nnz(v)); 
for m = 1:nnz(v)
	fprintf(fid, '%d\t%.3f\n',i(m),k(m)); 
end
fclose(fid);
