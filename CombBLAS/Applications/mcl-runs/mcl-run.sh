# input: cage10.mcl: matrix market file without the header
# output: cage10.out: 1 line for each cluster 

mcl cage10.mcl --abc -I 2 -p 0.01 -S 5 -R 6 -pct 0.9 -o cage10.out
