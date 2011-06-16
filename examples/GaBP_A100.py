# Input: A - information matrix mxm, (assumed to be symmetric) 
# b - shift vector 1xm
# max_round - max iteration
# epsilon - convergence threshold
#
# Output:currently this function doesn't return anything. vector ha would be the
# x of Ax = b

import time
import math
import sys
import getopt
import kdt

k = -1

if (len(sys.argv) > 1):
	outfile = sys.argv[1]
	for i in range(1, len(sys.argv)):
		a = sys.argv[i]
		if a == '-k':
			k = int(sys.argv[i+1])

if k > 2:
	import ModelProblemGen
	
	if (kdt.master()):
		print "Generating %d-by-%d model problem..."%(k,k)
		
	A, b = ModelProblemGen.getModelProbem(k)
else:
	A = kdt.DiGraph.load('thermal2/thermal2.mtx');
	b = kdt.ParVec.load('thermal2/thermal2_b.mtx');

nvert = A.nvert()

def gabp(A, b, maxround, epsilon):
	copy_time=0
	scale_time=0
	t_time=0
	add_time=0
	mul_divide_time=0
	sum_time=0
	cmp_time=0
	
	t1 = time.time()
	m = A.nvert()
	pv = kdt.ParVec(0)
	#Mh,MJ init to m by m all-zero matrices
	Mh = kdt.DiGraph(pv,pv,pv,m)
	MJ = Mh.copy()
	
	conv = False
	
	stencil=A.copy()
	stencil.ones()
	stencil.removeSelfLoops()
	#stencil.reverseEdges()
	#print stencil
	
	#create an m*m identity matrix
	pi = kdt.ParVec.range(0,m)
	pw = kdt.ParVec(m,1)
	eye = kdt.DiGraph(pi,pi,pw,m)
	
	diagA = A*eye
	[piDiagA,pjDiagA,peDiagA] = diagA.toParVec()
	h = b.copy()
	J = peDiagA.copy()
	
	
	ha = h / J
	r=1
	t2 = time.time()
	init_time = t2-t1
	rel_norm = 40000
	while r<=maxround:
		if kdt.master():
			print "starting GBP round %d, relnorm=%f"%(r, rel_norm)
		preRes = ha
	
		t3 = time.time()
		# COPY
		Mhtemp = stencil.copy()
		MJtemp = stencil.copy()
	
		t4 = time.time()
		copy_time += (t4-t3)
	
		# SCALE
		Mhtemp.scale(h)	# default direction: dir=kdt.DiGraph.Out, which scales rows
		MJtemp.scale(J)
#		print MJtemp.toParVec()
		
		t5 = time.time()
		scale_time += t5-t4
		
#		if kdt.master():
#			print "scale time: %f" % (t5-t4)

		# TRANSPOSE
		Mh.reverseEdges()
		MJ.reverseEdges()
		t6 = time.time()
		t_time += t6-t5
		
		# ADD
		h_m = Mhtemp + -Mh
		J_m = MJtemp + -MJ
		t7 = time.time()
		add_time += t7-t6
		
		# MUL/DIVIDE
		val = -A / J_m
		Mh = val * h_m
		MJ = val * A
		
		t8 = time.time()
		mul_divide_time += t8-t7
		
		# SUM
		Mh.removeSelfLoops()
		MJ.removeSelfLoops()
		h = b + Mh.sum(kdt.DiGraph.In)
		J = peDiagA + MJ.sum(kdt.DiGraph.In)
		
		t9 = time.time()
		sum_time += t9-t8
		
		Ja = 1.0/J
		ha=h*Ja

		ha_norm = ha.norm(2)
		if (ha_norm == 0.0):
			rel_norm = 0
		else:
			rel_norm = (ha-preRes).norm(2)/ha_norm
			
		#rel_norm = (ha-preRes).norm(2) #Adam
		
		t10 = time.time()
		cmp_time += t10-t9

 #	if kdt.master():
#		print "rel_norm %f after round %d"% (rel_norm,r)   	
		if r > 2 and rel_norm<epsilon:
			y = kdt.SpParVec(m)
			y._spv=A._spm.SpMV_PlusTimes(ha.toSpParVec()._spv)
			real_norm = (y-b).toParVec().norm(2)
			if kdt.master():
				after = time.time()
				print "GBP Converged after %d rounds, reached rel_norm %f real_norm %f"% (r,rel_norm,real_norm)
				print "run time %fs"%(after-t1)
			conv = True
			break
		r += 1
	after=time.time()
	if kdt.master():
		print "init time:   %fs"%init_time
		print "copy time:   %fs"%copy_time
		print "scale_time:  %fs"%scale_time
		print "t_time:      %fs"%t_time
		print "add_time:    %fs"%add_time
		print "m_d_time:    %fs"%mul_divide_time
		print "sum_time:    %fs"%sum_time
		print "cmp_time:    %fs"%cmp_time
		print "total_time:  %fs"%(after-t1)
	if conv==False:
		y = kdt.SpParVec(m)
		y._spv=A._spm.SpMV_PlusTimes(ha.toSpParVec()._spv)
		real_norm = (y-b).toParVec().norm(2)
		if kdt.master():
			print "GBP did not converge in %d rounds, reached rel_norm %f real_norm %f"%(r-1,rel_norm,real_norm)
			print "run time %fs"%(after-t1)
	#print ha
	#print Ja
	
	if False:
		if kdt.master():
			print "writing resulting vector x to x.mtx"
		X = kdt.DiGraph(kdt.ParVec.range(nvert), kdt.ParVec.ones(nvert)-1, ha, nvert)
		X.save("x.mtx")
	return

gabp(A,b,10000,1e-3)
