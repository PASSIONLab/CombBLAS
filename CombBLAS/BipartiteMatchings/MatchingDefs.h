//
//  MatchingDefs.h
//  
//
//  Created by Ariful Azad on 8/22/17.
//
//

#ifndef MatchingDefs_h
#define MatchingDefs_h

#include "CombBLAS/CombBLAS.h"
#include <iostream>

namespace combblas {

// Vertex data structure for maximal cardinality matching
template <typename T1, typename T2>
struct VertexTypeML
{
public:
	VertexTypeML(T1 p=-1, T2 com=0){parent=p; comp = com; };
	
	friend bool operator<(const VertexTypeML & vtx1, const VertexTypeML & vtx2 )
	{
        if(vtx1.comp==vtx2.comp) return vtx1.parent<vtx2.parent;
        else return vtx1.comp<vtx2.comp;
	};
	friend bool operator==(const VertexTypeML & vtx1, const VertexTypeML & vtx2 ){return (vtx1.parent==vtx2.parent) & (vtx1.comp==vtx2.comp);};
	friend std::ostream& operator<<(std::ostream& os, const VertexTypeML & vertex ){os << "(" << vertex.parent << "," << vertex.comp << ")"; return os;};
	T1 parent;
	T2 comp; // can be index, probability, degree or an adjacent edge weight
};



// Vertex data structure for maximum cardinality matching
template <typename IT>
struct VertexTypeMM
{
public:
	VertexTypeMM(IT p=-1, IT r=-1, double w=0){parent=p; root = r; comp = w;};
	friend bool operator<(const VertexTypeMM & vtx1, const VertexTypeMM & vtx2 )
	{
		if(vtx1.comp==vtx2.comp)
        {
            if(vtx1.parent==vtx2.parent)
                return vtx1.root<vtx2.root;
            else return vtx1.parent<vtx2.parent;
        }
		else return vtx1.comp<vtx2.comp;
        
	};
	friend bool operator==(const VertexTypeMM & vtx1, const VertexTypeMM & vtx2 ){return vtx1.parent==vtx2.parent;};
	friend std::ostream& operator<<(std::ostream& os, const VertexTypeMM & vertex ){os << "(" << vertex.parent << "," << vertex.root << ","<< vertex.comp << ")"; return os;};
	IT parent;
	IT root;
	double comp; // probability of selecting an edge or weight of an adjacent edge
	//making it double so that we can always use edge weights or probability
	// TODO: this is an overkill for Boolean matrices. Think a better way to cover the Boolean case
};

// Semiring needed to compute degrees within a subgraph
template <typename T1, typename T2>
struct SelectPlusSR
{
	static T2 id(){ return 1; };
	static bool returnedSAID() { return false; }
	static MPI_Op mpi_op() { return MPI_SUM; };
	
	static T2 add(const T2 & arg1, const T2 & arg2)
	{
		return std::plus<T2>()(arg1, arg2);
	}
	
	static T2 multiply(const T1 & arg1, const T2 & arg2)
	{
		return static_cast<T2> (1); // note: it is not called on a Boolean matrix
	}
	
	static void axpy(const T1 a, const T2 & x, T2 & y)
	{
		y = add(y, multiply(a, x));
	}
};


// Usual semiring used in maximal and maximum matching
template <typename T1, typename T2>
struct Select2ndMinSR
{
	static T2 id(){ return T2(); };
	static bool returnedSAID() { return false; }
	static MPI_Op mpi_op() { return MPI_MIN; };
	
	static T2 add(const T2 & arg1, const T2 & arg2)
	{
		return std::min(arg1, arg2);
	}
	
	static T2 multiply(const T1 & arg1, const T2 & arg2)
	{
		return arg2;
	}
	
	static void axpy(const T1 a, const T2 & x, T2 & y)
	{
		y = add(y, multiply(a, x));
	}
};

// Designed to pseudo maximize weights on a maximal matching
template <typename T1, typename T2>
struct WeightMaxMLSR
{
	static T2 id(){ return T2(-1, std::numeric_limits<T1>::lowest()); };
	static bool returnedSAID() { return false; }
	static MPI_Op mpi_op() { return MPI_MAX; };
	
	static T2 add(const T2 & arg1, const T2 & arg2)
	{
		return std::max(arg1, arg2);
	}
	
	static T2 multiply(const T1 & arg1, const T2 & arg2)
	{
		return T2(arg2.parent, arg1);
	}
	
	static void axpy(const T1 a, const T2 & x, T2 & y)
	{
		y = add(y, multiply(a, x));
	}
};



// Designed to pseudo maximize weights on the augmenting paths
// for boolean matrix (T1 <=> bool), this semiring converts to Select2ndMax semiring
template <typename T1, typename T2>
struct WeightMaxMMSR
{
	static T2 id(){ return T2(-1, -1, std::numeric_limits<T1>::lowest()); };
	static bool returnedSAID() { return false; }
	static MPI_Op mpi_op() { return MPI_MAX; };
	
	static T2 add(const T2 & arg1, const T2 & arg2)
	{
		return std::max(arg1, arg2);
	}
	
	static T2 multiply(const T1 & arg1, const T2 & arg2)
	{
		return T2(arg2.parent, arg2.root, arg1);
	}
	
    
	static void axpy(const T1 a, const T2 & x, T2 & y)
	{
		y = add(y, multiply(a, x));
	}
};

}

#endif /* MatchingDefs_h */
