#permute

./mpermute -fin /project/projectdirs/m1982/ariful/symmetric/thermal2/thermal2.mtx  -fout thermal2_rcm.bin -permute rcm
./mpermute -fin /project/projectdirs/m1982/ariful/symmetric/thermal2/thermal2.mtx  -fout thermal2.bin


./mpermute -fin /project/projectdirs/m1982/ariful/symmetric/af_shell4/af_shell4.mtx  -fout af_shell4.bin

./mpermute -fin /project/projectdirs/m1982/ariful/symmetric/parabolic_fem/parabolic_fem.mtx -fout parabolic_fem_rcm.bin -permute rcm
#no permute 
./mpermute -fin /project/projectdirs/m1982/ariful/symmetric/parabolic_fem/parabolic_fem.mtx -fout parabolic_fem_nopermute.bin
#srun -n 4 ./ex18 -f0 cage10.bin -permute nd -ksp_type cg


srun -n 4 ./ex18 -f0 parabolic_fem_rcm.bin -ksp_type cg

srun -n 4 ./ex18 -f0 parabolic_fem_nopermute.bin -ksp_type cg
 
