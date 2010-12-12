#ifndef _FULLY_DIST_SP_VEC_H_
#define _FULLY_DIST_SP_VEC_H_

#include <iostream>
#include <vector>
#include <utility>

#ifdef NOTR1
	#include <boost/tr1/memory.hpp>
#else
	#include <tr1/memory>
#endif

#include "CommGrid.h"
#include "promote.h"
#include "SpParMat.h"

using namespace std;
using namespace std::tr1;

template <class IT, class NT, class DER>
class SpParMat;

template <class IT>
class DistEdgeList;


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
  *
 **/
  
template <class IT, class NT>
class FullyDistSpVec
{
public:
	FullyDistSpVec ( );
	FullyDistSpVec ( IT glen );
	FullyDistSpVec ( shared_ptr<CommGrid> grid);
	FullyDistSpVec ( shared_ptr<CommGrid> grid, IT glen);

	//! like operator=, but instead of making a deep copy it just steals the contents. 
	//! Useful for places where the "victim" will be distroyed immediately after the call.
	void stealFrom(FullyDistSpVec<IT,NT> & victim); 
	FullyDistSpVec<IT,NT> & operator+=(const FullyDistSpVec<IT,NT> & rhs);
	FullyDistSpVec<IT,NT> & operator-=(const FullyDistSpVec<IT,NT> & rhs);
	ifstream& ReadDistribute (ifstream& infile, int master);	

	template <typename NNT> operator FullyDistSpVec< IT,NNT > () const	//!< Type conversion operator
	{
		FullyDistSpVec<IT,NNT> CVT(commGrid);
		CVT.ind = vector<IT>(ind.begin(), ind.end());
		CVT.num = vector<NNT>(num.begin(), num.end());
		CVT.glen = glen;
		CVT.NOT_FOUND = NOT_FOUND;
	}

	void PrintInfo(string vecname) const;
	void iota(IT size, NT first);
	FullyDistSpVec<IT,NT> operator() (const FullyDistSpVec<IT,IT> & ri) const;	//!< SpRef (expects NT of ri to be 0-based)
	void SetElement (IT indx, NT numx);	// element-wise assignment
	NT operator[](IT indx) const;

	// sort the vector itself
	// return the permutation vector (0-based)
	FullyDistSpVec<IT, IT> sort();	

	IT getlocnnz() const 
	{
		return ind.size();
	}
	IT getnnz() const
	{
		IT totnnz = 0;
		IT locnnz = ind.size();
		(commGrid->GetWorld()).Allreduce( &locnnz, & totnnz, 1, MPIType<IT>(), MPI::SUM); 
		return totnnz;
	}
	IT LengthUntil() const;
	IT MyLocLength() const;
	IT TotalLength() const { return glen; }
	int Owner(IT gind, IT & lind) const;

	void setNumToInd()
	{
		MPI::Intracomm World = commGrid->GetWorld();
           	int rank = World.Get_rank();
            	IT n_perproc = getTypicalLocLength();
            	IT offset = static_cast<IT>(rank) * n_perproc;
            	transform(ind.begin(), ind.end(), num.begin(), bind2nd(plus<IT>(), offset));
	}

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{
		transform(num.begin(), num.end(), num.begin(), __unary_op);
	}

	template <typename _BinaryOperation>
	NT Reduce(_BinaryOperation __binary_op, NT init);
	void DebugPrint();
	shared_ptr<CommGrid> getCommGrid() { return commGrid; }

private:
	
	shared_ptr<CommGrid> commGrid;
	vector< IT > ind;	// ind.size() give the number of nonzeros
	vector< NT > num;
	IT glen;		// global length (actual "length" including zeros)
	const static IT zero = static_cast<IT>(0);
	NT NOT_FOUND; 

	template <class IU, class NU>
	friend class FullyDistSpVec;

	template <class IU, class NU>
	friend class FullyDistVec;
	
	template <class IU, class NU, class UDER>
	friend class SpParMat;

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend FullyDistSpVec<IU,typename promote_trait<NUM,NUV>::T_promote> 
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,NUV> & x );

	template <typename IU, typename NU1, typename NU2>
	friend FullyDistSpVec<IU,typename promote_trait<NU1,NU2>::T_promote> 
	EWiseMult (const FullyDistSpVec<IU,NU1> & V, const FullyDistVec<IU,NU2> & W , bool exclude, NU2 zero);

	template <typename IU>
	friend void RandPerm(FullyDistSpVec<IU,IU> & V); 	// called on an existing object, randomly permutes it
	
	template <typename IU>
	friend void RenameVertices(DistEdgeList<IU> & DEL);
};

#include "FullyDistSpVec.cpp"
#endif

