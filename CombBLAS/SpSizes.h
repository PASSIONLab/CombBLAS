#ifndef _SP_SIZES_H
#define _SP_SIZES_H

#include "SpDefines.h"

class SpSizes
{
public:
	SpSizes(ITYPE nprocs): procs(nprocs)
	{
		nrows = new ITYPE[nprocs];
		ncols = new ITYPE[nprocs];
		nzcs = new ITYPE[nprocs];
		nnzs = new ITYPE[nprocs];
	}
	~SpSizes()
	{
		delete [] nrows;
		delete [] ncols;
		delete [] nzcs;
		delete [] nnzs;
	}
	SpSizes (const SpSizes & rhs)
	{
		for(int i=0; i<procs; i++)
		{
			nrows[i] = rhs.nrows[i];
			ncols[i] = rhs.ncols[i];	
			nzcs[i] = rhs.nzcs[i];
			nnzs[i] = rhs.nnzs[i];
		}
	}
	SpSizes & operator=(const SpSizes & rhs)	
	{
		if(this != &rhs)		
		{
			for(int i=0; i<procs; i++)
			{
				nrows[i] = rhs.nrows[i];
				ncols[i] = rhs.ncols[i];	
				nzcs[i] = rhs.nzcs[i];
				nnzs[i] = rhs.nnzs[i];
			}
		}
		return *this;
	}
	ITYPE procs;
	ITYPE * nrows;
	ITYPE * ncols;
	ITYPE * nzcs;
	ITYPE * nnzs;
};

#endif
