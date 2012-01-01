#ifndef _TWITTER_EDGE_
#define _TWITTER_EDGE_

#include <iostream>
#include <ctime>
#include "CombBLAS.h"

using namespace std;

/**
 * This is not part of the Combinatorial BLAS library, it's just an example.
 * A nonzero A(i,j) is present if at least one edge (of any type)
 * from vertex i to vertex j exists. Multiple types of edges are supported: follower/retweeter (or both)
 **/
class TwitterEdge
{
public:
	TwitterEdge():count(0) {};
	bool isFollower() const {	return follower; };
	bool isRetwitter() const {	return (count > 0); };
	bool TweetWithinInterval (time_t begin, time_t end) const	{	return ((count > 0) && (begin <= latest && latest <= end));  };
private:
	bool follower;		// default constructor sets all to zero
	time_t latest;		// not assigned if no retweets happened
	short count;		
	
	template <typename IT>
	friend class TwitterReadSaveHandler;
};


template <class IT>
class TwitterReadSaveHandler
{
	public:
		TwitterEdge getNoNum(IT row, IT col) { return TwitterEdge(); }

		template <typename c, typename t>
		TwitterEdge read(std::basic_istream<c,t>& is, IT row, IT col)
		{
			TwitterEdge tw;
			is >> tw.follower;
			is >> tw.count;
			if(tw.count > 0)
			{
				string date;
				string time;
				is >> date;
				is >> time;

				struct tm timeinfo;
                        	int year, month, day, hour, min, sec;
				sscanf (date.c_str(),"%d-%d-%d",&year, &month, &day);
				sscanf (time.c_str(),"%d:%d:%d",&hour, &min, &sec);

				memset(&timeinfo, 0, sizeof(struct tm));
				timeinfo.tm_year = year - 1900; // year is "years since 1900"
				timeinfo.tm_mon = month - 1 ;   // month is in range 0...11
				timeinfo.tm_mday = day;         // range 1...31
				timeinfo.tm_hour = hour;        // range 0...23
				timeinfo.tm_min = min;          // range 0...59
				timeinfo.tm_sec = sec;          // range 0.
				tw.latest = mktime(&timeinfo);
				if(tw.latest == -1) { cout << "Can not parse time date" << endl; exit(-1);}
			}
			else
			{
				tw.latest = 0;	// initialized to dummy
			}
			return tw;
		}
		
	
		template <typename c, typename t>
		void save(std::basic_ostream<c,t>& os, const TwitterEdge & tw, IT row, IT col)	// save is NOT compatible with read
		{
			os << row << "\t" << col << "\t";
			os << tw.follower << "\t";
			os << tw.count << "\t";
			os << tw.latest << endl;
		}
};


struct ParentType
{
	ParentType():id(-1) { };
	ParentType(int64_t myid):id(myid) { };
	int64_t id;
	bool operator==(const ParentType & rhs) const
	{
		return (id == rhs.id);
	}
};

ParentType NumSetter(ParentType & num, int64_t index) 
{
	return ParentType(index);
}


void select2nd(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype);	// forward declaration


//! Filters officially don't exist in Combinatorial BLAS
//! KDT generates them by embedding their filter stack and pushing those
//! predicates to the SR::multiply() function, conceptually like 
//! if(predicate(maxrix_val)) { bin_op(xxx) }
//! Here we emulate this filtered traversal approach.
struct LatestRetwitterBFS
{
	static MPI_Op MPI_BFSADD;
	static ParentType id() { return ParentType(); }	// additive identity
	static ParentType add(const ParentType & arg1, const ParentType & arg2)
	{
		return arg2;
	}

	static MPI_Op mpi_op() 
	{ 
		MPI_Op_create(select2nd, false, &MPI_BFSADD);	// todo: do this once only !
		return MPI_BFSADD;
	}
	static ParentType multiply(const TwitterEdge & arg1, const ParentType & arg2)
	{
		time_t now = time(0);
		struct tm * timeinfo = localtime( &now);
		timeinfo->tm_mon = timeinfo->tm_mon-1;
		time_t monthago = mktime(timeinfo);
		if(arg1.isRetwitter() && arg1.TweetWithinInterval(monthago, now))	// T1 is of type edges for BFS
		{
			return arg2;
		}
		else
		{
			return ParentType();	// null-type parent id
		}
	}
	static void axpy(TwitterEdge a, const ParentType & x, ParentType & y)
	{
		y = multiply(a, x);
	}
};


void select2nd(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype)
{
	ParentType * pinvec = static_cast<ParentType*>(invec);
	ParentType * pinoutvec = static_cast<ParentType*>(inoutvec);
        for (int i = 0; i < *len; i++)
        {
                pinoutvec[i] = LatestRetwitterBFS::add(pinvec[i], pinoutvec[i]);
        }
}

MPI_Op LatestRetwitterBFS::MPI_BFSADD;
#endif
