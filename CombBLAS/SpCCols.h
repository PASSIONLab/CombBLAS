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

#ifndef _SP_CCOLS_H_
#define _SP_CCOLS_H_

template <class IT, class NT>
class SpCCols: public SpMat<IT, NT, SpCCols<IT, NT> >
{
public:
    // Constructors :
    SpCCols ();
    SpCCols (IT size, IT nRow, IT nCol);
    SpCCols (const SpTuples<IT,NT> & rhs, bool transpose);

    SpCCols (const SpCCols<IT,NT> & rhs);					// Actual copy constructor
    SpCCols();

    // Member Functions and Operators:
    SpCCols<IT,NT> & operator= (const SpCCols<IT, NT> & rhs);
    SpCCols<IT,NT> & operator+= (const SpCCols<IT, NT> & rhs);

    void RowSplit(int numsplits);
    void ColSplit(int parts, vector< SpCCols<IT,NT> > & matrices); //!< \attention Destroys calling object (*this)
    
    void CreateImpl(const vector<IT> & essentials);
    void CreateImpl(IT size, IT nRow, IT nCol, tuple<IT, IT, NT> * mytuples);
    
    Arr<IT,NT> GetArrays() const;
    vector<IT> GetEssentials() const;
    const static IT esscount;
    
    IT getnrow() const { return m; }
    IT getncol() const { return n; }
    IT getnnz() const { return nnz; }
    int getnsplit() const { return splits; }

private:
    // Anonymous union
    union {
        Csc<IT, NT> * dcsc;
        Csc<IT, NT> ** dcscarr;
    };
    
    IT m;
    IT n;
    IT nnz;
    
    template <typename SR, typename IU, typename NU, typename RHS, typename LHS>
    friend void csc_gespmv_dense (const SpCCols<IU, NU> & A, const RHS * x, LHS * y); //!< dense vector
    
    template <typename SR, typename IU, typename NUM, typename IVT, typename OVT>
    friend int csc_gespmv_sparse (const SpCCols<IU, NUM> & A, const int32_t * indx, const IVT * numx, int32_t nnzx,
                                     int32_t * & sendindbuf, OVT * & sendnumbuf, int * & sdispls, int p_c);  //!< sparse vector
}

#include "SpCCols.cpp"
#endif

