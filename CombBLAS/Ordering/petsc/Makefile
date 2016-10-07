CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/ksp/ksp/examples/tutorials/
MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk
NP               = 1

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


ex18: ex18.o  chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_KSP_LIB}
	${RM} ex18.o

mmconvert: mmconvert.o  chkopts
	-${CLINKER} -o mmconvert mmconvert.o  
	${RM} mmconvert.o

mpermute: mpermute.o 
	-${CLINKER} -o mpermute mpermute.o  
	${RM} mpermute.o
#---------------------------------------------------------------------------
runex18:
	-@${MPIEXEC} -n 3 ./ex18 -m 39 -n 18 -ksp_monitor_short -permute nd > ex18_1.tmp 2>&1; \
	   ${DIFF} output/ex18_1.out ex18_1.tmp || printf "${PWD}\nPossible problem with ex18_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex18_1.tmp
runex18_2:
	-@${MPIEXEC} -n 3 ./ex18 -m 39 -n 18 -ksp_monitor_short -permute rcm > ex18_2.tmp 2>&1; \
	   ${DIFF} output/ex18_2.out ex18_2.tmp || printf "${PWD}\nPossible problem with ex18_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex18_2.tmp
runex18_3:
	-@${MPIEXEC} -n 3 ./ex18 -m 13 -n 17 -ksp_monitor_short -ksp_type cg -ksp_cg_single_reduction > ex18_3.tmp 2>&1; \
	   ${DIFF} output/ex18_3.out ex18_3.tmp || printf "${PWD}\nPossible problem with ex18_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex18_3.tmp
