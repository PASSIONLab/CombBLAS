import kdt
import time


for i in [2**x for x in range(7, 8)]:
    print "i=%d" % i
    G1 = kdt.DiGraph.twoDTorus(i)
    #[i1,j1,v1] = G1.toParVec()
    if False:
        before = time.time();
        bcExact = G1.centrality('exactBC')
        exactTime = time.time() - before;
        if ((bcExact - bcExact[0]) > 1e-15).any():
            if kdt.master():
                print "not all vertices have same value for bcExact"
	bc100min = bcExact.min()
	bc100max = bcExact.max()
	bc100mean = bcExact.mean()
	bc100std = bcExact.std()
        if kdt.master():
            print "bcExact[0] = %f, min=%f, max=%f, mean=%f, std=%f" % (bcExact[0], bc100min, bc100max, bc100mean, bc100std)
            print "   took %4.3f seconds" % exactTime
    if False:
        before = time.time();
        bcApprox100 = G1.centrality('approxBC', sample=1.0)
        approx100Time = time.time() - before;
        if ((bcApprox100 - bcApprox100[0]) > 1e-15).any():
            if kdt.master():
                print "not all vertices have same value for bcApprox100"
	bc100min = bcApprox100.min()
	bc100max = bcApprox100.max()
	bc100mean = bcApprox100.mean()
	bc100std = bcApprox100.std()
        if kdt.master():
            print "bcApprox100[0] = %f, min=%f, max=%f, mean=%f, std=%f" % (bcApprox100[0], bc100min, bc100max, bc100mean, bc100std)
            print "   took %4.3f seconds" % approx100Time
    if False:
        before = time.time();
        bcApprox050 = G1.centrality('approxBC', sample=0.5, BCdebug=1)
        approx050Time = time.time() - before;
        if ((bcApprox050 - bcApprox050[0]) > 1e-15).any():
            if kdt.master():
                print "not all vertices have same value for bcApprox050"
	bc050min = bcApprox050.min()
	bc050max = bcApprox050.max()
	bc050mean = bcApprox050.mean()
	bc050std = bcApprox050.std()
        if kdt.master():
            print "bcApprox050[0] = %f, min=%f, max=%f, mean=%f, std=%f" % (bcApprox050[0], bc050min, bc050max, bc050mean, bc050std)
            print "   took %4.3f seconds" % approx050Time
    if True:
        before = time.time();
        bcApprox005 = G1.centrality('approxBC', sample=0.05, BCdebug=1)
        approx005Time = time.time() - before;
        if ((bcApprox005 - bcApprox005[0]) > 1e-15).any():
            if kdt.master():
                print "not all vertices have same value for bcApprox005"
	bc005min = bcApprox005.min()
	bc005max = bcApprox005.max()
	bc005mean = bcApprox005.mean()
	bc005std = bcApprox005.std()
        if kdt.master():
            print "bcApprox005[0] = %f, min=%f, max=%f, mean=%f, std=%f" % (bcApprox005[0], bc005min, bc005max, bc005mean, bc005std)
            print "   took %4.3f seconds" % approx005Time
