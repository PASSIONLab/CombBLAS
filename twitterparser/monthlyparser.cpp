#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <stdio.h>
#include <map>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <sys/time.h>
//#include "omp.h"

using namespace std;

#define MILLION 1000000
#define THOUSAND 1000
#define FILENUM 7	// total number of files

template <class T>
inline void hash_combine(std::size_t & seed, const T & v)
{
  hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std {
	template <>
	struct hash<pair<int, int> > 
	{
		public:
        	size_t operator()(pair<int, int> x) const throw() 
		{
             		size_t h = 0;
			hash_combine(h, x.first);
			hash_combine(h, x.second);
             		return h;
        	}
	};
}	// close namespaces

struct TwitterInteraction
{
	int32_t from;
	int32_t to;
	bool follow;
	int16_t retweets;
	time_t twtime;
};

int main(int argc, char *argv[] )
{

	typedef pair<int, int> EDGEKEY;		// (from, to)
	typedef tuple<bool, int, time_t> EDGEVAL;	// (follow, count, last_tweet)
	typedef unordered_map< EDGEKEY, EDGEVAL > EDGEMAP;
	typedef unordered_set<int> VERTEXSET;
	vector< EDGEMAP > edgemaps;
	vector< VERTEXSET > vertexsets;
	cout << "Size of time_t is " << sizeof(time_t) << " bytes" << endl;
	   
	//#pragma omp parallel for
	for(int i=6; i< 6+FILENUM; i++)
	{ 
		ostringstream outs;
		outs << "tweets2009-";
		if (i< 10) outs << "0";
		outs << i << ".txt.triples.num";
		
		int from, to, year, month, day, hour, min, sec;	
		time_t date;
		EDGEMAP rtedges;
		VERTEXSET rtvertices;
		string fname = outs.str();
       		FILE * rFile = fopen (fname.c_str(),"r");
		cout << "Reading " <<fname << endl;
		struct timeval startTime;
    		struct timeval endTime;
    		// get the current time
    		// -NULL because we don't care about time zone
  		gettimeofday(&startTime, NULL);
		if(rFile != NULL)
		{
			int readlines = 0;
			int run =1;
			struct tm timeinfo;
			while(!feof(rFile))
			{
				// example time format: 2009-06-08T21:49:36
				if(fscanf (rFile, "%d %d %d-%d-%dT%d:%d:%d\n",&from,&to,&year,&month,&day,&hour,&min,&sec) ==  0) break;
				memset(&timeinfo, 0, sizeof(struct tm));
				timeinfo.tm_year = year - 1900;	// year is "years since 1900"
				timeinfo.tm_mon = month - 1 ;	// month is in range 0...11
				timeinfo.tm_mday = day;		// range 1...31
				timeinfo.tm_hour = hour;	// range 0...23
				timeinfo.tm_min = min;		// range 0...59
				timeinfo.tm_sec = sec;		// range 0...59
				
				date = mktime(&timeinfo);
				if(date == -1) { cout << "Can not parse time date" << endl; break;}
				rtvertices.insert(from);
				rtvertices.insert(to);
				EDGEKEY ekey = make_pair(from, to);

				// Iterators to elements of map containers access to both the key and the mapped value. 
				// For this, the class defines what is called its value_type, which is a pair class with its 
				// first value corresponding to the const version of the key type (template parameter Key) and 
				// its second value corresponding to the mapped value (template parameter T)

				pair< EDGEMAP::iterator,bool> ret;
				ret = rtedges.insert( EDGEMAP::value_type( ekey, make_tuple(false, 1, date)));
				if(ret.second == false)
				{
					// update with incremented count and updated date (if newer)
					// ABAB: Tested that max(time_t,time_t) and map::operator[] workds as intended
					EDGEVAL eval = rtedges[ekey];
					rtedges[ekey] = make_tuple(false, get<1>(eval)+1, max(get<2>(eval), date));
				}
				readlines++;
				if(readlines == run * MILLION)
				{
					cout << run << " million retweets read, ";
					cout << "Map contains " << (int) rtedges.size() << " unique pairs, ";
					cout << "Subgraph contains " << (int) rtvertices.size() << " vertices." << endl;
					run++;
					gettimeofday(&endTime, NULL);
    					// calculate time in microseconds
    					double tS = startTime.tv_sec + (startTime.tv_usec) / MILLION;
    					double tE = endTime.tv_sec  + (endTime.tv_usec) / MILLION;
   					cout << "Total Time Taken: " << tE - tS << " seconds" << endl;
				}
			}
			edgemaps.push_back(rtedges);
			vertexsets.push_back(rtvertices);
		}
		else
		{
			cout << "Couldn't open " << fname << endl; 
		}
		fclose(rFile);
	}

	cout << "s0 has " << vertexsets[0].size() << " vertices" << endl; 
	cout << "Merging sets... " << endl;
	for(int i=0; i<FILENUM-1; ++i)
	{
		vertexsets[i+1].insert(vertexsets[i].begin(), vertexsets[i].end());	
		cout << "s" << 0 << "..." << i+1 <<" has " << vertexsets[i+1].size() << " vertices" << endl; 
	}
	cout << "Merging edges... " << endl;
	for(int i=0; i<FILENUM-1; ++i)
	{
		for (EDGEMAP::iterator it = edgemaps[i].begin(); it != edgemaps[i].end(); ++it)
		{
			pair< EDGEMAP::iterator,bool> ret = edgemaps[i+1].insert( *it );
			if(ret.second == false)	// already exists
			{
				// update with new count and date
				EDGEVAL eval = edgemaps[i+1][it->first];
				edgemaps[i+1][it->first] = make_tuple(false, get<1>(eval)+get<1>(it->second), max(get<2>(eval), get<2>(it->second)));
			}
		}
		cout << i << " and " << i+1 << " merged." << endl;
	}


	cout << "Reading the follower data..." << endl;
	FILE * pFile = fopen ("twitter_rv.net","r");

	int from, to;
	int count = 0;
	int run =1;
	if(pFile != NULL)
	{
		while(!feof(pFile))
		{
			// from and to reserved because the data is of the form "USER FOLLOWER"
			// it is natural to have an arc (edge) from the follower to the followee
			// just like the edge we put from retweeter to the tweetee 
			if(fscanf (pFile, "%d %d", &to, &from) ==  0)  break;	
			for(int i=0; i< FILENUM; ++i)
			{
				VERTEXSET::iterator fr_it = vertexsets[i].find(from);
				VERTEXSET::iterator to_it = vertexsets[i].find(to);	
				if( fr_it != vertexsets[i].end() && to_it != vertexsets[i].end()) // both ends exist
				{
					// is there a tweet between these guys?
					EDGEKEY ekey = make_pair(from, to);
					EDGEMAP::iterator it = edgemaps[i].find( ekey );
					if(it != edgemaps[i].end())	
					{
						get<0>(it->second) = true;	// mark it "following"
					}
					else
					{
						time_t dummy = 0;
						edgemaps[i].insert( EDGEMAP::value_type( ekey, make_tuple(true, 0, dummy)));	
					}
				}
			}

			if(count == 10*MILLION)
			{
				cout << run << " ten million edges processed." << endl;
				run++;
				count = 0;
			}
			else	count++;
		}
		for(int i=0; i< FILENUM; ++i)
		{	
			
			ostringstream outs;
			outs << "twitter_from06_to-";
			if (i+6< 10) outs << "0";
			outs << (i+6) << ".induced";
		
			string fname = outs.str();
			outs << ".bin";
			string bname = outs.str();
       			FILE * wFile = fopen (fname.c_str(),"w+");
       			FILE * bFile = fopen (bname.c_str(),"w+");	// binary file
			cout << "Writing " <<fname << endl;
			// format: row_id col_id {0/1 for follow} {0-n for retweets} {last retweet date}
			for (EDGEMAP::iterator it = edgemaps[i].begin(); it != edgemaps[i].end(); ++it)
			{
				char buff[20];
				time_t ttime = get<2>(it->second);
				if(ttime != 0)
				{
					struct tm * stime = localtime(&ttime);
					strftime(buff, 20, "%Y-%m-%d %H:%M:%S", stime);
				}
				else
				{
					buff[0] = 0;
				}
				fprintf(wFile, "%d\t%d\t%d\t%d\t%s\n", it->first.first, it->first.second, get<0>(it->second), get<1>(it->second), buff);

				TwitterInteraction twi;
				twi.from = it->first.first;
				twi.to = it->first.second;
				twi.follow = get<0>(it->second);
				twi.retweets =  get<1>(it->second);
				twi.twtime = ttime;

				fwrite(&twi,sizeof(TwitterInteraction),1,bFile);	// write binary file	
			}
			fclose(wFile);
			fclose(bFile);
		}
	}
	return 0;
}
