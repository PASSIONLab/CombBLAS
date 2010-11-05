#       script intended to run after testscript.py to validate the results
#	NOTE:  need to change testscript not to delete all the vars

import pyCombBLAS as pcb
Applybroken = True


try:            # partially handle the case where the main script did not run
        A
except NameError:
        A = pcb.pySpParMat();
        A.load("/home/alugowski/matrices/rmat_scale14.mtx");
        levels = pcb.pyDenseParVec(A.getnrow(), 0);
        if Applybroken:
                for i in range(1,A.getnrow()+1):
                        levels.SetElement(i,1);
        else:
                levels.Apply(set(1));
        levels.SetElement(12345, -2);
        levels.SetElement(5,0);
        levels.GetElement(1);
        levels.GetElement(5);
        levels.GetElement(12345);

tmp1 = levels.FindInds_GreaterThan(-1);                 # levels > -1
tmp2 = pcb.pyDenseParVec(A.getnrow(),0);
if Applybroken:
        for i in range(1,A.getnrow()+1):
                tmp2.SetElement(i,0);
else:
        tmp2.Apply(set(0));
tmp2 -= levels;                                         # == -levels
tmp2ndx = tmp2.FindInds_GreaterThan(-1);                # levels < 1
root = pcb.EWiseMult(tmp1, tmp2ndx, True, 0);           # (levels < 1) & (levels > -1)

visited = pcb.pySpParVec.zeros(A.getncol());
visited.SetElement(root,1);

fringe = pcb.pySpParVec.zeros(A.getncol());
fringe.SetElement(root,1);