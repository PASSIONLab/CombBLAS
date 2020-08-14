/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 11/15/2016 --------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, Adam Lugowski ------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2016, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */


#ifndef _CSC_H
#define _CSC_H

#include <cstdlib>
#include <vector>
#include <limits>
#include <cassert>
#include "SpDefs.h"
#include "SpHelper.h"

namespace combblas {

template <class IT, class NT>
class Csc
{
public:
    typedef NT value_type;
    typedef IT index_type;
    Csc ();
    Csc (IT size,IT nCol);
    Csc (const Csc<IT,NT> & rhs);				// copy constructor
    ~Csc();
    
    Csc<IT,NT> & operator=(const Csc<IT,NT> & rhs);	// assignment operator
    void Resize(IT nsize);

	// index-based pruning
	template <typename UnaryOperation,
			  typename GlobalIT>
	Csc<IT, NT> *
	PruneI(UnaryOperation unary_op, bool inPlace,
		   GlobalIT rowOffset, GlobalIT colOffset);

	void
	Split(Csc<IT, NT> * &A, Csc<IT, NT> * &B, IT cut);

	void
	Merge(const Csc<IT, NT> *A, const Csc<IT, NT> *B, IT cut);


    IT * jc ;	    //	col pointers, size n+1
    IT * ir ;	    //  row indices, size nzmax
    NT * num;		//  generic values, size nzmax
    IT n;			//  number of columns
    IT nz;
};

}

#include "csc.cpp"	// Template member function definitions need to be known to the compiler

#endif
