#       script intended to run after testscript.py to validate the results
#	NOTE:  need to change testscript not to delete all the vars

import pyCombBLAS as pcb
Applybroken = True


try:            # partially handle the case where the main script did not run
        A
except NameError:
        A = pcb.pySpParMat();
        A.load("../../CombBLAS/TESTDATA/SCALE16BTW-TRANSBOOL/input.txt");
        #A.load("/home/alugowski/matrices/rmat_scale14.mtx");
	nrowA = A.getnrow();
	#nrowA = 16384;
	root = 5;
        levels = pcb.pyDenseParVec(nrowA, 1);
        levels.SetElement(root,0);
        levels.SetElement(12345, -2);
        levels.GetElement(1)
        levels.GetElement(5)
        levels.GetElement(12345)
	parents = pcb.pyDenseParVec(nrowA, 5);
	parents.SetElement(root,-1)
	parents.SetElement(12345,-2)

tmp1 = levels.FindInds_GreaterThan(-1);                 # levels > -1
tmp2 = pcb.pyDenseParVec(nrowA,0);
tmp2 -= levels;                                         # == -levels
tmp2ndx = tmp2.FindInds_GreaterThan(-1);                # levels < 1
root = pcb.EWiseMult(tmp2ndx.sparse(), tmp1, True, 0).GetElement(0);           # (levels < 1) & (levels > -1)
#tmp1 = levels.FindInds_NotEqual(0);
#tmp2 = tmp1.GetElement(0);

# Spect test #1:  !!NOTE!! Not completed!!
print "starting spec test#1"
visited = pcb.pySpParVec.zeros(nrowA);
visited.SetElement(root,1);

fringe = pcb.pySpParVec.zeros(nrowA);
fringe.SetElement(root,1);

cycle = False;
#while fringe.FindInds_NotEqual(0)).getnnz() > 0 and not cycle:
#	pass

# Spect test #2

print "starting spec test#2"
#treeEdges = ((parents<> -2) & (parents <> -1)).nonzero();
tmp1 = parents.Find_NotEqual(-2)
tmp2 = parents.Find_NotEqual(-1)
tmp3 = pcb.pyDenseParVec(nrowA,0);
tmp3.ApplyMasked_SetTo(tmp2,1);
treeEdges = pcb.EWiseMult(tmp1,tmp3,False,0);

#for next line, really need just the nonzeros in treeEdges used
treeI = parents.SubsRef(treeEdges.dense());
treeJ = pcb.pyDenseParVec.range(treeEdges.getnnz(),0);
pass
