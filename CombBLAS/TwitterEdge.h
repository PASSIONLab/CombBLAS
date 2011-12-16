#ifndef _TWITTER_EDGE_
#define _TWITTER_EDGE_

#include <iostream>
#include <ctype>
#include "CombBLAS.h"

using namespace std;

/**
 * This is not part of the Combinatorial BLAS library, it's just an example.
 * A nonzero A(i,j) is present if at least one edge (of any type)
 * from vertex i to vertex j exists. If multiple types of edges are supported,
 * then the presence bitset encodes the existence of such edges for a given nonzero
 **/
template <class I>	// I: number of types
class TwitterEdge
{
public:
	TwitterEdge():count(0) {};
	bool isFollower() {	return presence[0]; };
	bool isRetwitter() {	return presence[1]; };
	bool TweetWithinInterval(time_t begin, time_t end)	{	return ((count > 0) && (begin <= latest && latest <= end));  };
private:
	bitset<I> presence;	// default constructor sets all to zero
	time_t latest;		// not assigned if no retweets happened
	short count;		
	
	template <typename IT>
	friend class TwitterReadSaveHandler;
};


template <class IT>
class TwitterReadSaveHandler
{
	public:
		TwitterEdge getNoNum(IT row, IT col) { return TwitterEdge<2>(); }

		template <typename c, typename t>
		NT read(std::basic_istream<c,t>& is, IT row, IT col)
		{
			TwitterEdge<2> tw;
			bool following;
			is >> following;
			tw.precence[0] = following;
			while(is >> tw.latest)	++tw.count;	// update the count of retweets, while keeping the date of the latest retweet
			if(tw.count > 0) tw.precence[1] = true;
			return tw;
		}
	
		template <typename c, typename t>
		void save(std::basic_ostream<c,t>& os, const TwitterEdge<2> & tw, IT row, IT col)	// save is NOT compatible with read
		{
			os << (bool) tw.precense[0] << "\t";
			os << tw.count << "\t";
			os << tw.latest << endl;
		}
};

//! Filters officially don't exist in Combinatorial BLAS
//! KDT generates them by embedding their filter stack and pushing those
//! predicates to the SR::multiply() function, conceptually like 
//! if(predicate(maxrix_val)) { bin_op(xxx) }
//! Here we emulate this filtered traversal approach.
template <class T1, class T2, class OUT>
struct LatestRetwitterBFS
{
	static OUT id() { return OUT(); }
	static MPI_Op mpi_op() { return MPI_MAX; };
	static OUT add(const OUT & arg1, const OUT & arg2)
	{
		return arg2;
	}
	static OUT multiply(const T1 & arg1, const T2 & arg2)
	{
		time_t now = time(0);
		struct tm * timeinfo = localtime( &now);
		timeinfo->tm_mon = timeinfo->tm_mon-1;
		time_t monthago = mktime(timeinfo);
		if(T1.isRetwitter() && T1.TweetWithinInterval(monthago, now))	// T1 is of type edges for BFS
		{
			return static_cast<OUT>(arg2);
		}
		else
		{
			return OUT();	// null-type parent id
		}
	}
	static void axpy(T1 a, const T2 & x, OUT & y)
	{
		y = multiply(a, x);
	}
};
#endif
