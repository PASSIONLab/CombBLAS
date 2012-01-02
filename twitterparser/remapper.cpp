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

using namespace std;

#define MILLION 1000000
#define THOUSAND 1000


struct TwitterInteraction
{
	int32_t from;
	int32_t to;
	bool follow;
	int16_t retweets;
	time_t twtime;
};

typedef unordered_map< int32_t, int32_t > VERMAP;	// vertex map

int64_t retweeted, following;

bool ProcessEntry(TwitterInteraction & twi, VERMAP & vertexmap, int & vid, int & count, int & run, FILE * wFile, FILE * bFile, int64_t & edges)
{
	VERMAP::iterator it = vertexmap.find( twi.from );
	if(it != vertexmap.end())	
	{
		twi.from = it->second;	// remap it
	}
	else
	{
		vertexmap.insert( VERMAP::value_type( twi.from, ++vid) );	// create a new vertex id, and insert it
		twi.from = vid;
	}

	// now do the same for the other end
	it = vertexmap.find( twi.to );
	if(it != vertexmap.end())	
	{
		twi.to = it->second;	// remap it
	}
	else
	{
		vertexmap.insert( VERMAP::value_type( twi.to, ++vid) );	// create a new vertex id, and insert it
		twi.to = vid;
	}

	if(count == 10*MILLION)
	{
		cout << run << " ten million edges processed." << endl;
		run++;
		count = 0;
	}
	else	count++;

	char buff[20];
	if(twi.twtime != 0)	// memset was zero'ed initially
	{
		// Localtime: The return value points to a statically allocated struct which might be 
		// overwritten by subsequent calls to any of the date and time functions
		// What happens is the library will allocate memory on the first call, then keep a pointer 
		// to the allocated memory and not allocate it a second time. This is why you should NOT free 
		// the pointer returned to you by localtime(). If you do, then you'll end up with all sorts of 
		// randomness on successive calls to localtime(), and if you try to free the pointer returned by 
		// localtime() a second time you'll get a debug run time error about a bad pointer (notice every 
		// time you call localtime() you're returned a pointer to the same address)

		struct tm * stime = localtime(&(twi.twtime));
		strftime(buff, 20, "%Y-%m-%d %H:%M:%S", stime);
	}
	else
	{
		buff[0] = 0;
	}

	fprintf(wFile, "%d\t%d\t%d\t%d\t%s\n", twi.from, twi.to, twi.follow, twi.retweets, buff);	// write text file
	fwrite(&twi,sizeof(TwitterInteraction),1,bFile);	// write binary file	
	++edges;
	return true;
}

int main(int argc, char *argv[] )
{
	if(argc < 3)
	{
		cout << "Usage: " << argv[0] << " <filename> <bin/text>" << endl;
		return 0; 
	}

	stringstream outs;
	outs << argv[1] << ".remapped";
       	FILE * wFile = fopen (outs.str().c_str(),"w+");
	outs << ".bin";
       	FILE * bFile = fopen (outs.str().c_str(),"wb+");


	VERMAP vertexmap;
	int count = 0;
	int run = 1;
	int32_t vid = 0;	// default I/O is 1 based, but it will be incremented before first insert
	int64_t	edges = 0;

	retweeted = static_cast<int64_t>(10000) * static_cast<int64_t>(MILLION);
	char buffer[40];
	sprintf (buffer, "%% Edges with retweets: %lld\n", retweeted);	// test if %lld works
	cout << buffer;
	following = 0;
	sprintf (buffer, "%% Edges with follows: %lld\n", following);
	cout << buffer;
	sprintf (buffer, "%d\t%d\t%lld\n", vid, vid, edges);
	cout << buffer;
	fseek ( wFile , 120 , SEEK_SET );	// leave space to come back
	retweeted = 0;		// complete the test

	fwrite(&vid,sizeof(int32_t),1,bFile);
	fwrite(&edges,sizeof(int64_t),1,bFile);	// total number of edges

	if(string(argv[2]) == string("text"))
	{
       		FILE * rFile = fopen (argv[1],"r");
		if(rFile != NULL)
		{
			cout << "Reading text file" << endl;
			struct tm timeinfo;
			int year, month, day, hour, min, sec;	
			while(!feof(rFile))
			{
				TwitterInteraction twi;

				// example time format (for text): 2009-06-08 21:49:30
				if(fscanf (rFile, "%d %d %d %d",&(twi.from),&(twi.to),&(twi.follow),&(twi.retweets)) ==  0) break;
				if(twi.retweets > 0)
				{
					++retweeted;
					if(fscanf (rFile, "%d-%d-%d %d:%d:%d\n", &year, &month, &day, &hour, &min, &sec) == 0) 
					{
						cout << "Expected retweet data non-existent\n";
						break;
					}
					memset(&timeinfo, 0, sizeof(struct tm));
					timeinfo.tm_year = year - 1900;	// year is "years since 1900"
					timeinfo.tm_mon = month - 1 ;	// month is in range 0...11
					timeinfo.tm_mday = day;		// range 1...31
					timeinfo.tm_hour = hour;	// range 0...23
					timeinfo.tm_min = min;		// range 0...59
					timeinfo.tm_sec = sec;		// range 0...59
				
					twi.twtime = mktime(&timeinfo);
					if(twi.twtime == -1) { cout << "Can not parse time date" << endl; break;}
				}
				else
				{
					twi.twtime = 0;
					fscanf (rFile, "\n");	// read newline	
				}
				if(twi.follow)	++following;

				if( ! ProcessEntry(twi, vertexmap, vid, count, run, wFile, bFile, edges) )	break;
			}
		}
	}
	else if(string(argv[2]) == string("bin"))
	{
       		FILE * rFile = fopen (argv[1],"rb");
		if(rFile != NULL)
		{
			cout << "Reading binary" << endl;
			while(!feof(rFile))
			{
				TwitterInteraction twi;
				fread (&twi,sizeof(TwitterInteraction),1,rFile);	
			
				if(twi.retweets > 0)	++retweeted;
				if(twi.follow)	++following;
				if( ! ProcessEntry(twi, vertexmap, vid, count, run, wFile, bFile, edges) ) 	break;
			}
		}
	}
	else
	{
		cout << "What's your file format, dude?" << endl;
	}

	cout << "Writing total number of vertices and edges: " << vid << " and " << edges << endl;
	fseek ( bFile , 0 , SEEK_SET );
	fwrite(&vid,sizeof(int32_t),1,bFile);
	fwrite(&edges,sizeof(int64_t),1,bFile);	// total number of edges

	fseek ( wFile , 0 , SEEK_SET );

	fprintf (wFile, "%% Edges with retweets: %lld\n", retweeted);
	fprintf (wFile, "%% Edges with follows: %lld\n", following);
	fprintf (wFile, "%d\t%d\t%lld\n", vid, vid, edges);
	cout << "Ratio of edges with retweets: " << static_cast<double>(retweeted) / static_cast<double>(edges) << endl;
	cout << "Ratio of edges with followers: " << static_cast<double>(following) / static_cast<double>(edges) << endl;

	fclose(wFile);
	fclose(bFile);

	// test written file
	cout << "Reading first line of binary file..." << endl;
	bFile = fopen (outs.str().c_str(),"r");
	int32_t num_vertices;
	int64_t num_edges;
	fread(&num_vertices, sizeof(num_vertices), 1, bFile);
	fread(&num_edges, sizeof(num_edges), 1, bFile);
	printf ("Vertices: %d, Edges: %lld\n", num_vertices, num_edges);
	fclose(bFile);
	
	return 0;
}
