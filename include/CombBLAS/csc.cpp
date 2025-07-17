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


#include "csc.h"
#include <cassert>

namespace combblas {

// Constructing empty Csc objects (size = 0) are not allowed.
template <class IT, class NT>
Csc<IT, NT>::Csc () :
	jc(NULL), ir(NULL), num(NULL), n(0), nz(0)
{
}
	
	
template <class IT, class NT>
Csc<IT,NT>::Csc (IT size, IT nCol): nz(size),n(nCol)
{
    assert(size != 0 && n != 0);
    num = new NT[nz];
    ir = new IT[nz];
    jc = new IT[n+1];
}

template <class IT, class NT>
Csc<IT,NT>::Csc (const Csc<IT,NT> & rhs): n(rhs.n), nz(rhs.nz)
{
    if(nz > 0)
    {
        ir	= new IT[nz];
        num	= new NT[nz];
        std::copy(rhs.ir, rhs.ir+nz, ir); // copy(first, last, result)
        std::copy(rhs.num, rhs.num+nz, num);
    }
    jc	= new IT[n+1];
    std::copy(rhs.jc, rhs.jc+n+1, jc);
}

template <class IT, class NT>
Csc<IT,NT> & Csc<IT,NT>::operator= (const Csc<IT,NT> & rhs)
{
    if(this != &rhs)
    {
        if(nz > 0)	// if the existing object is not empty
        {
            // make it empty
            delete [] num;
            delete [] ir;
        }
        delete [] jc;
        
        nz	= rhs.nz;
        n	= rhs.n;
        if(nz > 0)	// if the copied object is not empty
        {
            ir		= new IT[nz];
            num	= new NT[nz];
            std::copy(rhs.ir, rhs.ir+nz, ir);
            std::copy(rhs.num, rhs.num+nz, num);
        }
        jc	= new IT[n+1];
        std::copy(rhs.jc, rhs.jc+n+1, jc);
    }
    return *this;
}

template <class IT, class NT>
Csc<IT,NT>::~Csc()
{
    if( nz > 0)
    {
        delete [] num;
        delete [] ir;
    }
    delete [] jc;
}

//! Does not change the dimension
template <class IT, class NT>
void Csc<IT,NT>::Resize(IT nsize)
{
    if(nsize == nz)
    {
        // No need to do anything!
        return;
    }
    else if(nsize == 0)
    {
        delete []  num;
        delete []  ir;
        nz = 0;
        return;
    }
    
    NT * tmpnum = num;
    IT * tmpir = ir;
    num	= new NT[nsize];
    ir	= new IT[nsize];
    
    if(nsize > nz)	// Grow it
    {
        std::copy(tmpir, tmpir + nz, ir);   //copy all old elements
        std::copy(tmpnum, tmpnum + nz, num);
    }
    else	// Shrink it
    {
        std::copy(tmpir, tmpir + nsize, ir);   // copy only a portion of the old elements
        std::copy(tmpnum, tmpnum + nsize, num);
    }
    delete [] tmpnum;		// delete the memory pointed by previous pointers
    delete [] tmpir;
    nz = nsize;
}



template <class IT, class NT>
template <typename UnaryOperation, typename GlobalIT>
Csc<IT, NT> *
Csc<IT, NT>::PruneI (UnaryOperation unary_op,
					 bool			inPlace,
					 GlobalIT		rowOffset,
					 GlobalIT		colOffset
					 )
{
	IT prunednnz = 0;
	for (IT i = 0; i < n; ++i)
	{
		for (IT j = jc[i]; j < jc[i+1]; ++j)
		{
			if (!(unary_op(std::make_tuple(rowOffset + ir[j],
										   colOffset + i, num[j]))))
				++prunednnz;
		}
	}

	IT	*oldjc	= jc;
	IT	*oldir	= ir;
	NT	*oldnum = num;
	jc			= new IT[n + 1];
	ir			= new IT[prunednnz];
	num			= new NT[prunednnz];

	IT	cnnz = 0;
	jc[0]	 = 0;
	for (IT i = 0; i < n; ++i)
	{
		for (IT j = oldjc[i]; j < oldjc[i+1]; ++j)
		{
			if (!(unary_op(std::make_tuple(rowOffset + oldir[j],
										   colOffset + i, oldnum[j]))))
			{
				ir[cnnz]	= oldir[j];
				num[cnnz++] = oldnum[j];
			}
		}
		jc[i+1] = cnnz;
	}

	assert (cnnz == prunednnz);

	Csc<IT, NT> *ret = NULL;
	if (inPlace)
	{
		DeleteAll(oldnum, oldir, oldjc);
		nz = cnnz;
	}
	else
	{
		ret = new Csc<IT, NT>();
		ret->jc	 = jc;
		ret->ir	 = ir;
		ret->num = num;
		ret->n	 = n;
		ret->nz	 = cnnz;

		jc	= oldjc;
		ir	= oldir;
		num = oldnum;
	}

	return ret;
}



template <class IT, class NT>
void
Csc<IT, NT>::Split (Csc<IT, NT> *	&A,
					Csc<IT, NT> *	&B,
					IT				 cut
					)
{
	// left
	if (jc[cut] == 0)
		A = NULL;
	else
	{
		A = new Csc<IT, NT>(jc[cut], cut);
		std::copy(jc, jc + cut + 1, A->jc);
		std::copy(ir, ir + jc[cut], A->ir);
		std::copy(num, num + jc[cut], A->num);
	}

	// right
	if (nz - jc[cut] == 0)
		B = NULL;
	else
	{
		B = new Csc<IT, NT>(nz - jc[cut], n - cut);
		std::copy(jc + cut, jc + n + 1, B->jc);
        auto jccut = jc[cut];
		transform(B->jc, B->jc + (n - cut + 1), B->jc,
		[jccut](IT val) { return val - jccut; }
		);
		std::copy(ir + jc[cut], ir + nz, B->ir);
		std::copy(num + jc[cut], num + nz, B->num);
	}
}



template <class IT, class NT>
void
Csc<IT, NT>::Merge (const Csc<IT, NT>	*A,
					const Csc<IT, NT>	*B,
					IT					 cut
					)
{
	assert (A != NULL && B != NULL);

	IT	cnz = A->nz + B->nz;
	IT	cn	= A->n + B->n;
	if (cnz > 0)
	{
		*this = Csc<IT, NT>(cnz, cn);

		std::copy(A->jc, A->jc + A->n, jc);
		std::copy(B->jc, B->jc + B->n + 1, jc + A->n);
		const IT offset = A->jc[A->n];
		std::transform(jc + A->n, jc + cn + 1, jc + A->n,
					   [offset](IT val) { return val + offset; });

		std::copy(A->ir, A->ir + A->nz, ir);
		std::copy(B->ir, B->ir + B->nz, ir + A->nz);

		std::copy(A->num, A->num + A->nz, num);
		std::copy(B->num, B->num + B->nz, num + A->nz);
	}
}




}
