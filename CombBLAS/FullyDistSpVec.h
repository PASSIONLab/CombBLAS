/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.4 -------------------------------------------------*/
/* date: 1/17/2014 ---------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/
/*
 Copyright (c) 2010-2014, The Regents of the University of California
 
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

#ifndef _FULLY_DIST_SP_VEC_H_
#define _FULLY_DIST_SP_VEC_H_

#include <iostream>
#include <vector>
#include <utility>
#include "CommGrid.h"
#include "promote.h"
#include "SpParMat.h"
#include "FullyDist.h"
#include "Exception.h"
#include "OptBuf.h"
#include "CombBLAS.h"

template <class IT, class NT, class DER>
class SpParMat;

template <class IT>
class DistEdgeList;

template <class IU, class NU>
class FullyDistVec;

template <class IU, class NU>
class SparseVectorLocalIterator;

/** 
  * A sparse vector of length n (with nnz <= n of them being nonzeros) is distributed to 
  * "all the processors" in a way that "respects ordering" of the nonzero indices
  * Example: x = [5,1,6,2,9] for nnz(x)=5 and length(x)=12 
  *	we use 4 processors P_00, P_01, P_10, P_11
  * 	Then P_00 owns [1,2] (in the range [0,...,2]), P_01 ow`ns [5] (in the range [3,...,5]), and so on.
  * In the case of A(v,w) type sparse matrix indexing, this doesn't matter because n = nnz
  * 	After all, A(v,w) will have dimensions length(v) x length (w) 
  * 	v and w will be of numerical type (NT) "int" and their indices (IT) will be consecutive integers 
  * It is possibly that nonzero counts are distributed unevenly
  * Example: x=[1,2,3,4,5] and length(x) = 20, then P_00 would own all the nonzeros and the rest will hold empry vectors
  * Just like in SpParMat case, indices are local to processors (they belong to range [0,...,length-1] on each processor)
  * \warning Always create vectors with the right length, setting elements won't increase its length (similar to operator[] on std::vector)
 **/
template <class IT, class NT>
class FullyDistSpVec: public FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>
{
public:
	FullyDistSpVec ( );
	explicit FullyDistSpVec ( IT glen );
	FullyDistSpVec ( shared_ptr<CommGrid> grid);
	FullyDistSpVec ( shared_ptr<CommGrid> grid, IT glen);

    template <typename _UnaryOperation>
    FullyDistSpVec (const FullyDistVec<IT,NT> & rhs, _UnaryOperation unop);
	FullyDistSpVec (const FullyDistVec<IT,NT> & rhs);					// Conversion copy-constructor
    FullyDistSpVec (IT globalsize, const FullyDistVec<IT,IT> & inds,  const FullyDistVec<IT,NT> & vals, bool SumDuplicates = false);

    FullyDistSpVec<IT,NT> Invert (IT globallen);
    template <typename _BinaryOperationIdx, typename _BinaryOperationVal>
    FullyDistSpVec<IT,NT> Invert (IT globallen, _BinaryOperationIdx __binopIdx, _BinaryOperationVal __binopVal);
    template <typename _BinaryOperationIdx, typename _BinaryOperationVal>
    FullyDistSpVec<IT,NT> InvertRMA (IT globallen, _BinaryOperationIdx __binopIdx, _BinaryOperationVal __binopVal);

    
    template <typename NT1, typename _UnaryOperation>
    void Select (const FullyDistVec<IT,NT1> & denseVec, _UnaryOperation unop);
    template <typename NT1, typename _UnaryOperation1, typename _UnaryOperation2>
    FullyDistSpVec<IT,NT1> SelectNew (const FullyDistVec<IT,NT1> & denseVec, _UnaryOperation1 __unop1, _UnaryOperation2 __unop2);
    template <typename _UnaryOperation>
    void FilterByVal (FullyDistSpVec<IT,IT> Selector, _UnaryOperation __unop, bool filterByIndex);
    template <typename NT1>
    void Setminus (const FullyDistSpVec<IT,NT1> & other);

    //template <typename NT1, typename _UnaryOperation>
    //void Set (FullyDistSpVec<IT,NT1> Selector, _UnaryOperation __unop);

    template <typename NT1, typename _UnaryOperation, typename _BinaryOperation>
    void SelectApply (const FullyDistVec<IT,NT1> & denseVec, _UnaryOperation __unop, _BinaryOperation __binop);
    template <typename NT1, typename _UnaryOperation, typename _BinaryOperation>
    FullyDistSpVec<IT,NT> SelectApplyNew (const FullyDistVec<IT,NT1> & denseVec, _UnaryOperation __unop, _BinaryOperation __binop);


	//! like operator=, but instead of making a deep copy it just steals the contents. 
	//! Useful for places where the "victim" will be distroyed immediately after the call.
	void stealFrom(FullyDistSpVec<IT,NT> & victim); 
	FullyDistSpVec<IT,NT> &  operator=(const FullyDistSpVec< IT,NT > & rhs);
	FullyDistSpVec<IT,NT> &  operator=(const FullyDistVec< IT,NT > & rhs);	// convert from dense
    FullyDistSpVec<IT,NT> &  operator=(NT fixedval) // assign fixed value
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IT i=0; i < ind.size(); ++i)
            num[i] = fixedval;
        return *this;
    }
	FullyDistSpVec<IT,NT> & operator+=(const FullyDistSpVec<IT,NT> & rhs);
	FullyDistSpVec<IT,NT> & operator-=(const FullyDistSpVec<IT,NT> & rhs);

	class ScalarReadSaveHandler
	{
	public:
		NT getNoNum(IT index) { return static_cast<NT>(1); }

		template <typename c, typename t>
		NT read(std::basic_istream<c,t>& is, IT index)
		{
			NT v;
			is >> v;
			return v;
		}
	
		template <typename c, typename t>
		void save(std::basic_ostream<c,t>& os, const NT& v, IT index)
		{
			os << v;
		}
	};
	
    //! New (2014) version that can handle parallel IO and binary files
    template <class HANDLER>
    void ReadDistribute (const string & filename, int master, HANDLER handler, bool pario);
    void ReadDistribute (const string & filename, int master, bool pario) { ReadDistribute(filename, master, ScalarReadSaveHandler(), pario); }


    //! Obsolete version that only accepts an ifstream object and ascii files
	template <class HANDLER>
	ifstream& ReadDistribute (ifstream& infile, int master, HANDLER handler);	
	ifstream& ReadDistribute (ifstream& infile, int master) { return ReadDistribute(infile, master, ScalarReadSaveHandler()); }
	
	template <class HANDLER>
	void SaveGathered(ofstream& outfile, int master, HANDLER handler, bool printProcSplits = false);
	void SaveGathered(ofstream& outfile, int master) { SaveGathered(outfile, master, ScalarReadSaveHandler()); }


	template <typename NNT> operator FullyDistSpVec< IT,NNT > () const	//!< Type conversion operator
	{
		FullyDistSpVec<IT,NNT> CVT(commGrid);
		CVT.ind = vector<IT>(ind.begin(), ind.end());
		CVT.num = vector<NNT>(num.begin(), num.end());
		CVT.glen = glen;
		return CVT;
	}

	bool operator==(const FullyDistSpVec<IT,NT> & rhs) const
	{
		FullyDistVec<IT,NT> v =  *this;
		FullyDistVec<IT,NT> w =  rhs;
		return (v == w);
	}

	void PrintInfo(string vecname) const;
	void iota(IT globalsize, NT first);
	FullyDistVec<IT,NT> operator() (const FullyDistVec<IT,IT> & ri) const;	//!< SpRef (expects ri to be 0-based)
	void SetElement (IT indx, NT numx);	// element-wise assignment
	void DelElement (IT indx); // element-wise deletion
	NT operator[](IT indx);
	bool WasFound() const { return wasFound; }

	//! sort the vector itself, return the permutation vector (0-based)
	FullyDistSpVec<IT, IT> sort();	

#if __cplusplus > 199711L
	template <typename _BinaryOperation = minimum<NT> >
	FullyDistSpVec<IT, NT> Uniq(_BinaryOperation __binary_op = _BinaryOperation(), MPI_Op mympiop = MPI_MIN);
#else
	template <typename _BinaryOperation >
	FullyDistSpVec<IT, NT> Uniq(_BinaryOperation __binary_op, MPI_Op mympiop);
#endif


	IT getlocnnz() const 
	{
		return ind.size();
	}
	IT getnnz() const
	{
		IT totnnz = 0;
		IT locnnz = ind.size();
		MPI_Allreduce( &locnnz, &totnnz, 1, MPIType<IT>(), MPI_SUM, commGrid->GetWorld());
		return totnnz;
	}
	using FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>::LengthUntil;
	using FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>::MyLocLength;
	using FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>::MyRowLength;
	using FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>::TotalLength;
	using FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>::Owner;
	using FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>::RowLenUntil;

	void setNumToInd()
	{
		IT offset = LengthUntil();
		IT spsize = ind.size();
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for(IT i=0; i< spsize; ++i)
			num[i] = ind[i] + offset;
	}

	template <typename _Predicate>
	IT Count(_Predicate pred) const;	//!< Return the number of elements for which pred is true

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{
		//transform(num.begin(), num.end(), num.begin(), __unary_op);
        IT spsize = num.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IT i=0; i < spsize; ++i)
            num[i] = __unary_op(num[i]);
	}

	template <typename _BinaryOperation>
	void ApplyInd(_BinaryOperation __binary_op)
	{
		IT offset = LengthUntil();
		IT spsize = ind.size();
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for(IT i=0; i < spsize; ++i)
			num[i] = __binary_op(num[i], ind[i] + offset);
	}

    

	template <typename _BinaryOperation>
	NT Reduce(_BinaryOperation __binary_op, NT init) const;
	
	template <typename OUT, typename _BinaryOperation, typename _UnaryOperation>
	OUT Reduce(_BinaryOperation __binary_op, OUT default_val, _UnaryOperation __unary_op) const;

	void DebugPrint();
	shared_ptr<CommGrid> getcommgrid() const { return commGrid; }
	
	void Reset();
	NT GetLocalElement(IT indx);
	void BulkSet(IT inds[], int count);
    vector<IT> GetLocalInd (){vector<IT> rind = ind; return rind;};
    vector<NT> GetLocalNum (){vector<NT> rnum = num; return rnum;};
    
    template <typename _Predicate>
    FullyDistVec<IT,IT> FindInds(_Predicate pred) const;
    template <typename _Predicate>
    FullyDistVec<IT,NT> FindVals(_Predicate pred) const;


protected:
	using FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>::glen; 
	using FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>::commGrid;
    
private:
	vector< IT > ind;	// ind.size() give the number of nonzeros
	vector< NT > num;
	bool wasFound; // true if the last GetElement operation returned an actual value
    
#if __cplusplus > 199711L
	template <typename _BinaryOperation = minimum<NT> >
	FullyDistSpVec<IT, NT> UniqAll2All(_BinaryOperation __binary_op = _BinaryOperation(), MPI_Op mympiop = MPI_MIN);
    
    template <typename _BinaryOperation = minimum<NT> >
	FullyDistSpVec<IT, NT> Uniq2D(_BinaryOperation __binary_op = _BinaryOperation(), MPI_Op mympiop = MPI_MIN);
#else
	template <typename _BinaryOperation >
	FullyDistSpVec<IT, NT> UniqAll2All(_BinaryOperation __binary_op, MPI_Op mympiop);
    
    template <typename _BinaryOperation >
	FullyDistSpVec<IT, NT> Uniq2D(_BinaryOperation __binary_op, MPI_Op mympiop);
#endif
    

    template <class HANDLER>
    void ReadAllMine(FILE * binfile, IT * & inds, NT * & vals, vector< pair<IT,NT> > & localpairs, int * rcurptrs, int * ccurptrs, int * rdispls, int * cdispls,
                     IT buffperrowneigh, IT buffpercolneigh, IT entriestoread, HANDLER handler, int rankinrow);
    
    void HorizontalSend(IT * & inds, NT * & vals, IT * & tempinds, NT * & tempvals, vector < pair <IT,NT> > & localpairs,
                        int * rcurptrs, int * rdispls, IT buffperrowneigh, int recvcount, int rankinrow);
    
    void VerticalSend(IT * & inds, NT * & vals, vector< pair<IT,NT> > & localpairs, int * rcurptrs, int * ccurptrs, int * rdispls, int * cdispls,
                      IT buffperrowneigh, IT buffpercolneigh, int rankinrow);
    
    void BcastEssentials(MPI_Comm & world, IT & total_m, IT & total_n, IT & total_nnz, int master);
    
    void AllocateSetBuffers(IT * & myinds, NT * & myvals,  int * & rcurptrs, int * & ccurptrs, int rowneighs, int colneighs, IT buffpercolneigh);
    
	template <class IU, class NU>
	friend class FullyDistSpVec;

	template <class IU, class NU>
	friend class FullyDistVec;
	
	template <class IU, class NU, class UDER>
	friend class SpParMat;
	
	template <class IU, class NU>
	friend class SparseVectorLocalIterator;

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend FullyDistSpVec<IU,typename promote_trait<NUM,NUV>::T_promote> 
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,NUV> & x );

	template <typename SR, typename IU, typename NUM, typename UDER> 
	friend FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue);

	template <typename VT, typename IU, typename UDER>	// NoSR version (in BFSFriends.h)
	friend FullyDistSpVec<IU,VT>  SpMV (const SpParMat<IU,bool,UDER> & A, const FullyDistSpVec<IU,VT> & x, OptBuf<int32_t, VT > & optbuf);

	template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
	friend void SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IVT> & x, FullyDistSpVec<IU,OVT> & y,bool indexisvalue, OptBuf<int32_t, OVT > & optbuf);

	template <typename IU, typename NU1, typename NU2>
	friend FullyDistSpVec<IU,typename promote_trait<NU1,NU2>::T_promote> 
	EWiseMult (const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W , bool exclude, NU2 zero);

	template <typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
	friend FullyDistSpVec<IU,RET> 
	EWiseApply (const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W , _BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls, NU1 Vzero, const bool useExtendedBinOp);

    template <typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
    friend FullyDistSpVec<IU,RET>
    EWiseApply_threaded (const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W , _BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls, NU1 Vzero, const bool useExtendedBinOp);

    
	template <typename RET, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate>
	friend FullyDistSpVec<IU,RET>
	EWiseApply (const FullyDistSpVec<IU,NU1> & V, const FullyDistSpVec<IU,NU2> & W , _BinaryOperation _binary_op, _BinaryPredicate _doOp, bool allowVNulls, bool allowWNulls, NU1 Vzero, NU2 Wzero, const bool allowIntersect, const bool useExtendedBinOp);

	template <typename IU>
	friend void RandPerm(FullyDistSpVec<IU,IU> & V); 	// called on an existing object, randomly permutes it
	
	template <typename IU>
	friend void RenameVertices(DistEdgeList<IU> & DEL);
	
	//! Helper functions for sparse matrix X sparse vector
	template <typename SR, typename IU, typename OVT>
	friend void MergeContributions(FullyDistSpVec<IU,OVT> & y, int * & recvcnt, int * & rdispls, int32_t * & recvindbuf, OVT * & recvnumbuf, int rowneighs);

	template <typename IU, typename VT>
	friend void MergeContributions(FullyDistSpVec<IU,VT> & y, int * & recvcnt, int * & rdispls, int32_t * & recvindbuf, VT * & recvnumbuf, int rowneighs);
	
	template<typename IU, typename NV>
	friend void TransposeVector(MPI_Comm & World, const FullyDistSpVec<IU,NV> & x, int32_t & trxlocnz, IU & lenuntil, int32_t * & trxinds, NV * & trxnums, bool indexisvalue);
    
    template <class IU, class NU, class DER, typename _UnaryOperation>
    friend SpParMat<IU, bool, DER> PermMat1 (const FullyDistSpVec<IU,NU> & ri, const IU ncol, _UnaryOperation __unop);
};

#include "FullyDistSpVec.cpp"
#endif

