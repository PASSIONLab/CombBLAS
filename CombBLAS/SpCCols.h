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

#include <cmath>
#include "SpMat.h"	// Best to include the base class first
#include "SpHelper.h"
#include "csc.h"

using namespace std;

template <class IT, class NT>
class SpCCols: public SpMat<IT, NT, SpCCols<IT, NT> >
{
public:
    typedef IT LocalIT;
    typedef NT LocalNT;
    
    // Constructors :
    SpCCols ();
    SpCCols (IT size, IT nRow, IT nCol);
    SpCCols (const SpTuples<IT,NT> & rhs, bool transpose);
    SpCCols (const SpDCCols<IT,NT> & rhs):nnz(0), n(0), m(0), splits(0), csc(NULL)
    {
        SpTuples<IT,NT> tuples(rhs);
        SpCCols<IT,NT> object(tuples, false);
        *this = object; // its members are already initialized by the initializer list
    }

    SpCCols (const SpCCols<IT,NT> & rhs);					// Actual copy constructor
    ~SpCCols();

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
    
    
    class SpColIter //! Iterate over (sparse) columns of the sparse matrix
    {
    public:
        class NzIter	//! Iterate over the nonzeros of the sparse column
        {
        public:
            NzIter(IT * ir = NULL, NT * num = NULL) : rid(ir), val(num) {}
            
            bool operator==(const NzIter & other)
            {
                return(rid == other.rid);	// compare pointers
            }
            bool operator!=(const NzIter & other)
            {
                return(rid != other.rid);
            }
            NzIter & operator++()	// prefix operator
            {
                ++rid;
                ++val;
                return(*this);
            }
            NzIter operator++(int)	// postfix operator
            {
                NzIter tmp(*this);
                ++(*this);
                return(tmp);
            }
            IT rowid() const	//!< Return the "local" rowid of the current nonzero entry.
            {
                return (*rid);
            }
            NT & value()		//!< value is returned by reference for possible updates
            {
                return (*val);
            }
        private:
            IT * rid;
            NT * val;
            
        };
        
        SpColIter(IT * jc = NULL) : begcptr(jc), curcptr(jc) {}
        
        bool operator==(const SpColIter& other)
        {
            return(curcptr == other.curcptr);	// compare current pointers
        }
        bool operator!=(const SpColIter& other)
        {
            return(curcptr != other.curcptr);
        }
        SpColIter& operator++()		// prefix operator
        {
            ++curcptr;
            return(*this);
        }
        SpColIter operator++(int)	// postfix operator
        {
            SpColIter tmp(*this);
            ++(*this);
            return(tmp);
        }
        IT colid() const	//!< Return the "local" colid of the current column.
        {
            return (curcptr-begcptr);
        }
        IT colptr() const   // only needed internally by ::begnz() below
        {
            return (*curcptr);
        }
        IT colptrnext() const   // only needed internally by ::begnz() below
        {
            return (*(curcptr+1));
        }
        IT nnz() const
        {
            return (colptrnext() - colptr());
        }
    private:
        IT * begcptr;
        IT * curcptr;
   	};
    
    SpColIter begcol()
    {
        if( nnz > 0 )
            return SpColIter(csc->jc);
        else
            return SpColIter(NULL);
    }
    SpColIter endcol()
    {
        if( nnz > 0 )
            return SpColIter(csc->jc + n);  // (csc->jc+n) should never execute because SpColIter::colptrnext() would point invalid
        else
            return SpColIter(NULL);
    }
    
    typename SpColIter::NzIter begnz(const SpColIter & ccol)	//!< Return the beginning iterator for the nonzeros of the current column
    {
        return typename SpColIter::NzIter( csc->ir + ccol.colptr(), csc->num + ccol.colptr() );
    }
    
    typename SpColIter::NzIter endnz(const SpColIter & ccol)	//!< Return the ending iterator for the nonzeros of the current column
    {
        return typename SpColIter::NzIter( csc->ir + ccol.colptrnext(), NULL );
    }

private:
    // Anonymous union
    union {
        Csc<IT, NT> * csc;
        Csc<IT, NT> ** cscarr;
    };
    
    IT m;
    IT n;
    IT nnz;
    
    int splits;	// for multithreading

    void CopyCsc(Csc<IT,NT> * source);
    
    template <typename SR, typename IU, typename NU, typename RHS, typename LHS>
    friend void csc_gespmv_dense (const SpCCols<IU, NU> & A, const RHS * x, LHS * y); //!< dense vector (not implemented)
    
    template <typename SR, typename IU, typename NUM, typename DER, typename IVT, typename OVT>
    friend int generic_gespmv_threaded (const SpMat<IU,NUM,DER> & A, const int32_t * indx, const IVT * numx, int32_t nnzx,
                                        int32_t * & sendindbuf, OVT * & sendnumbuf, int * & sdispls, int p_c); //<! sparse vector
};

#include "SpCCols.cpp"
#endif

