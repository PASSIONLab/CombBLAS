/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library  /
/  version 2.3 --------------------------------------------------/
/  date: 01/18/2009 ---------------------------------------------/
/  author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
\****************************************************************/

#ifndef _ISECT_H
#define _ISECT_H

/**
  * Carries information about the intersecting col(A) and row(B) indices of matrix operands
  **/ 
template <class IT>
class Isect
{
public:
	IT index;	// col-row index
	IT size;	
	IT start;
	IT current;	// current pointer

	bool operator < (const Isect & rhs) const
	{ return (index < rhs.index); }
	bool operator > (const Isect & rhs) const
	{ return (index > rhs.index); }
	bool operator == (const Isect & rhs) const
	{ return (index == rhs.index); }
};


#endif

