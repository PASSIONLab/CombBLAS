/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.5 -------------------------------------------------*/
/* date: 10/09/2015 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, Adam Lugowski ------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2015, The Regents of the University of California
 
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


// Constructing empty Csc objects (size = 0) are not allowed.
template <class IT, class NT>
Csc<IT,NT>::Csc (IT size, IT nCol): nz(size),n(nCol)
{
    assert(size != 0 && n != 0);
    numx = new NT[nz];
    ir = new IT[nz];
    jc = new IT[n+1];
}

template <class IT, class NT>
Csc<IT,NT>::Csc (const Csc<IT,NT> & rhs): n(rhs.n), nz(rhs.nz)
{
    if(nz > 0)
    {
        ir		= new IT[nz];
        numx	= new NT[nz];
        std::copy(rhs.ir, rhs.ir+nz, ir); // copy(first, last, result)
        std::copy(rhs.numx, rhs.numx+nz, numx);
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
            delete [] numx;
            delete [] ir;
        }
        delete [] jc;
        
        nz	= rhs.nz;
        n	= rhs.n;
        if(nz > 0)	// if the copied object is not empty
        {
            ir		= new IT[nz];
            numx	= new NT[nz];
            std::copy(rhs.ir, rhs.ir+nz, ir);
            std::copy(rhs.numx, rhs.numx+nz, numx);
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
        delete [] numx;
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
        delete []  numx;
        delete []  ir;
        nz = 0;
        return;
    }
    
    NT * tmpnumx = numx;
    IT * tmpir = ir;
    numx	= new NT[nsize];
    ir	= new IT[nsize];
    
    if(nsize > nz)	// Grow it
    {
        std::copy(tmpir, tmpir + nz, ir);   //copy all old elements
        std::copy(tmpnumx, tmpnumx + nz, numx);
    }
    else	// Shrink it
    {
        std::copy(tmpir, tmpir + nsize, ir);   // copy only a portion of the old elements
        std::copy(tmpnumx, tmpnumx + nsize, numx);
    }
    delete [] tmpnumx;		// delete the memory pointed by previous pointers
    delete [] tmpir;
    nz = nsize;
}
