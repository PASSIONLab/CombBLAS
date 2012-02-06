/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.2 -------------------------------------------------*/
/* date: 10/06/2011 --------------------------------------------*/
/* authors: Aydin Buluc (abuluc@lbl.gov), Adam Lugowski --------*/
/****************************************************************/
/*
 Copyright (c) 2011, Aydin Buluc
 
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

#ifndef _SP_PAR_MAT_H_
#define _SP_PAR_MAT_H_

#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <iterator>

#include "SpMat.h"
#include "SpTuples.h"
#include "SpDCCols.h"
#include "CommGrid.h"
#include "MPIType.h"
#include "LocArr.h"
#include "SpDefs.h"
#include "Deleter.h"
#include "SpHelper.h"
#include "SpParHelper.h"
#include "DenseParMat.h"
#include "FullyDistVec.h"
#include "Friends.h"
#include "Operations.h"
#include "DistEdgeList.h"

using namespace std;
using namespace std::tr1;

/**
  * Fundamental 2D distributed sparse matrix class
  * The index type IT is encapsulated by the class in a way that it is only
  * guarantee that the implementation will ensure the requested semantics. 
  * For instance, if IT=int64 then the implementation can still use 32 bit 
  * local indices but it should return correct 64-bit numbers in its functions. 
  * In other words, DER can be SpDCCols<int32_t, double> while IT=int64_t
  */
template <class IT, class NT, class DER>
class SpParMat
{
public:	
	typedef typename DER::LocalIT LocalIT;
	typedef typename DER::LocalNT LocalNT;
	typedef IT GlobalIT;
	typedef NT GlobalNT;
	
	// Constructors
	SpParMat ();
	SpParMat (DER * myseq, shared_ptr<CommGrid> grid);
		
	SpParMat (ifstream & input, MPI::Intracomm & world);
	SpParMat (DER * myseq, MPI::Intracomm & world);	

	template <class DELIT>
	SpParMat (const DistEdgeList< DELIT > & rhs, bool removeloops = true);	// conversion from distributed edge list

	SpParMat (const SpParMat< IT,NT,DER > & rhs);				// copy constructor
	SpParMat (IT total_m, IT total_n, const FullyDistVec<IT,IT> & , const FullyDistVec<IT,IT> & , const FullyDistVec<IT,NT> & );	// matlab sparse
	SpParMat (IT total_m, IT total_n, const FullyDistVec<IT,IT> & , const FullyDistVec<IT,IT> & , const NT & );	// matlab sparse
	SpParMat< IT,NT,DER > & operator=(const SpParMat< IT,NT,DER > & rhs);	// assignment operator
	SpParMat< IT,NT,DER > & operator+=(const SpParMat< IT,NT,DER > & rhs);
	~SpParMat ();

	template <typename SR>
	void Square (); 

	float LoadImbalance() const;
	void Transpose();
	void FreeMemory();
	void EWiseMult (const SpParMat< IT,NT,DER >  & rhs, bool exclude);
	void EWiseScale (const DenseParMat<IT,NT> & rhs);
	void DimScale (const DenseParVec<IT,NT> & v, Dim dim);
	void Find (FullyDistVec<IT,IT> & , FullyDistVec<IT,IT> & , FullyDistVec<IT,NT> & ) const;
	void Find (FullyDistVec<IT,IT> & , FullyDistVec<IT,IT> & ) const;

	template <typename _BinaryOperation>
	void DimApply(Dim dim, const FullyDistVec<IT, NT>& v, _BinaryOperation __binary_op);

	template <typename _BinaryOperation, typename _UnaryOperation >	
	DenseParVec<IT,NT> Reduce(Dim dim, _BinaryOperation __binary_op, NT id, _UnaryOperation __unary_op) const;

	template <typename _BinaryOperation>	
	DenseParVec<IT,NT> Reduce(Dim dim, _BinaryOperation __binary_op, NT id) const;

	template <typename VT, typename _BinaryOperation, typename _UnaryOperation >	
	void Reduce(DenseParVec<IT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op) const;

	template <typename VT, typename _BinaryOperation>	
	void Reduce(DenseParVec<IT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id) const;

	template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation >	
	void Reduce(FullyDistVec<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op) const;

	template <typename VT, typename GIT, typename _BinaryOperation>	
	void Reduce(FullyDistVec<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id) const;

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{
		spSeq->Apply(__unary_op);	
	}

	IT RemoveLoops();	// returns the number of loops removed
	
	template <typename LIT, typename OT>
	void OptimizeForGraph500(OptBuf<LIT,OT> & optbuf);
	void ActivateThreading(int numsplits);

	template <typename _UnaryOperation>
	SpParMat<IT,NT,DER> Prune(_UnaryOperation __unary_op, bool inPlace = true) //<! Prune any nonzero entries for which the __unary_op evaluates to true	
	{
		if (inPlace)
		{
			spSeq->Prune(__unary_op, inPlace);
			return SpParMat<IT,NT,DER>(); // return blank to match signature
		}
		else
		{
			return SpParMat<IT,NT,DER>(spSeq->Prune(__unary_op, inPlace), commGrid);
		}
	}

	template <typename _BinaryOperation>
	void UpdateDense(DenseParMat<IT, NT> & rhs, _BinaryOperation __binary_op) const;

	void Dump(string filename) const;
	void PrintInfo() const;

	template <typename NNT, typename NDER> operator SpParMat< IT,NNT,NDER > () const;	//!< Type conversion operator
	template <typename NIT, typename NNT, typename NDER> operator SpParMat< NIT,NNT,NDER > () const;	//!< Type conversion operator (for indices as well)

	IT getnrow() const;
	IT getncol() const;
	IT getnnz() const;

	SpParMat<IT,NT,DER> SubsRefCol (const vector<IT> & ci) const;				//!< Column indexing with special parallel semantics

	//! General indexing with serial semantics
	SpParMat<IT,NT,DER> operator() (const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci, bool inplace=false);
	SpParMat<IT,NT,DER> operator() (const SpParVec<IT,IT> & ri, const SpParVec<IT,IT> & ci) const;

	bool operator== (const SpParMat<IT,NT,DER> & rhs) const;

	class ScalarReadSaveHandler
	{
	public:
		NT getNoNum(IT row, IT col) { return static_cast<NT>(1); }
		
		template <typename c, typename t>
		NT read(std::basic_istream<c,t>& is, IT row, IT col)
		{
			NT v;
			is >> v;
			return v;
		}
	
		template <typename c, typename t>
		void save(std::basic_ostream<c,t>& os, const NT& v, IT row, IT col)
		{
			os << v;
		}
	};
	
	template <class HANDLER>
	ifstream& ReadDistribute (ifstream& infile, int master, bool nonum, HANDLER handler, bool transpose = false);
	ifstream& ReadDistribute (ifstream& infile, int master, bool nonum=false) { return ReadDistribute(infile, master, nonum, ScalarReadSaveHandler()); }

	template <class HANDLER>
	void SaveGathered(string filename, HANDLER handler, bool transpose = false) const;
	void SaveGathered(string filename) const { SaveGathered(filename, ScalarReadSaveHandler(), false); }
	
	ofstream& put(ofstream& outfile) const;
	void PrintForPatoh(string filename) const;

	shared_ptr<CommGrid> getcommgrid() const { return commGrid; } 	
	typename DER::LocalIT getlocalrows() const { return spSeq->getnrow(); }
	typename DER::LocalIT getlocalcols() const { return spSeq->getncol();} 
	typename DER::LocalIT getlocalnnz() const { return spSeq->getnnz(); }
	DER & seq() { return (*spSeq); }

	//! Friend declarations
	template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU, NUO, UDERO> 
	Mult_AnXBn_DoubleBuff (SpParMat<IU,NU1,UDER1> & A, SpParMat<IU,NU2,UDER2> & B, bool clearA, bool clearB );

	template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,NUO,UDERO> 
	Mult_AnXBn_Synch (SpParMat<IU,NU1,UDER1> & A, SpParMat<IU,NU2,UDER2> & B, bool clearA, bool clearB );

	template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	Mult_AnXBn_ActiveTarget (const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B );

	template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	Mult_AnXBn_PassiveTarget (const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B );

	template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	Mult_AnXBn_Fence (const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B );

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend DenseParVec<IU,typename promote_trait<NUM,NUV>::T_promote> 
	SpMV (const SpParMat<IU,NUM,UDER> & A, const DenseParVec<IU,NUV> & x );

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend FullyDistSpVec<IU,typename promote_trait<NUM,NUV>::T_promote>  
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,NUV> & x );

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend FullyDistVec<IU,typename promote_trait<NUM,NUV>::T_promote>  
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistVec<IU,NUV> & x );

	template <typename SR, typename IU, typename NUM, typename UDER> 
	friend FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue);

	// output type is part of the signature
	template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
	friend void SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IVT> & x, FullyDistSpVec<IU,OVT> & y, bool indexisvalue);
	
	template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
	friend void SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IVT> & x, FullyDistSpVec<IU,OVT> & y,bool indexisvalue, OptBuf<int32_t, OVT > & optbuf);

	template <typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	EWiseMult (const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B , bool exclude);

	template <typename RETT, typename RETDER, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB, typename _BinaryOperation> 
	friend SpParMat<IU,RETT,RETDER>
	EWiseApply (const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B, _BinaryOperation __binary_op, bool notB, const NU2& defaultBVal);

	template <typename RETT, typename RETDER, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB, typename _BinaryOperation, typename _BinaryPredicate> 
	friend SpParMat<IU,RETT,RETDER>
	EWiseApply (const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B, _BinaryOperation __binary_op, _BinaryPredicate do_op, bool allowANulls, bool allowBNulls, const NU1& ANullVal, const NU2& BNullVal, const bool allowIntersect);

	template<typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
	friend void LocalSpMV(const SpParMat<IU,NUM,UDER> & A, int rowneighs, OptBuf<int32_t, OVT > & optbuf, int32_t * & indacc, IVT * & numacc,
                           int32_t * & sendindbuf, OVT * & sendnumbuf, int * & sdispls, int * sendcnt, int accnz, bool indexisvalue);

private:
	int Owner(IT total_m, IT total_n, IT grow, IT gcol, IT & lrow, IT & lcol) const;
	shared_ptr<CommGrid> commGrid; 
	DER * spSeq;
	
	template <class IU, class NU>
	friend class DenseParMat;

	template <typename IU, typename NU, typename UDER> 	
	friend ofstream& operator<< (ofstream& outfile, const SpParMat<IU,NU,UDER> & s);	
};

template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
void PSpGEMM(SpParMat<IU,NU1,UDER1> & A, SpParMat<IU,NU2,UDER2> & B, SpParMat<IU,NUO,UDERO> & out, bool clearA = false, bool clearB = false)
{
	out = Mult_AnXBn_Synch<SR, NUO, UDERO> (A, B, clearA, clearB );
}


#include "SpParMat.cpp"
#endif
