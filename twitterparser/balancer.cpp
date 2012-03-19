#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>

#include <stdio.h>
#include <map>
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include <numeric>
using namespace std;


struct TwitterInteraction
{
	int32_t from;
	int32_t to;
	bool follow;
	int16_t retweets;
	time_t twtime;
};

struct HeaderInfo
{
	HeaderInfo():fileexists(false), headerexists(false) {};
	bool fileexists;
	bool headerexists;
	uint64_t version;
	uint64_t objsize;
	uint64_t format;	// 0: binary, 1: ascii
	
	uint64_t m;
	uint64_t n;
	uint64_t nnz;
};
	
inline HeaderInfo ParseHeader(const string & inputname, FILE * & f, int & seeklength)
{
	f = fopen(inputname.c_str(), "rb");
	HeaderInfo hinfo;
	memset(&hinfo, 0, sizeof(hinfo));
	if(!f)
	{
		cerr << "Problem reading binary input file\n";
		f = NULL;
		return hinfo;
	}
	char fourletters[5];
	size_t result = fread(fourletters, sizeof(char), 4, f);
	fourletters[4] = '\0';
	if (result != 4) { cout << "Error in fread of header, only " << result << " entries read" << endl; return hinfo;}

	if(strcmp(fourletters,"HKDT") != 0)
	{
		rewind(f);
		fclose(f);
		hinfo.fileexists = true;
		return hinfo;
	}
	else 
	{
		hinfo.fileexists = true;
		hinfo.headerexists = true;
	}

	size_t results[6];
	results[0] = fread(&(hinfo.version), sizeof(hinfo.version), 1, f);
	results[1] = fread(&(hinfo.objsize), sizeof(hinfo.objsize), 1, f);
	results[2] = fread(&(hinfo.format), sizeof(hinfo.format), 1, f);

	results[3] = fread(&(hinfo.m), sizeof(hinfo.m), 1, f);
	results[4] = fread(&(hinfo.n), sizeof(hinfo.n), 1, f);
	results[5] = fread(&(hinfo.nnz), sizeof(hinfo.nnz), 1, f);
	if(accumulate(results,results+6,0) != 6)
	{
		cout << "The required 6 fields (version, objsize, format, m,n,nnz) are not read" << endl;
		cout << "Only " << accumulate(results,results+6,0) << " fields are read" << endl;
	} 
	else
	{
	#ifdef DEBUG
		cout << "Version " << hinfo.version << ", object size " << hinfo.objsize << endl;
		cout << "Rows " << hinfo.m << ", columns " << hinfo.m << ", nonzeros " << hinfo.nnz << endl;
	#endif
	}

	seeklength = 4 + 6 * sizeof(uint64_t);
	return hinfo;
}
				  

template<typename _ForwardIter, typename T>
void my_iota(_ForwardIter __first, _ForwardIter __last, T __val)
{
	while (__first != __last)
     		*__first++ = __val++;
}

int main(int argc, char *argv[] )
{
	if(argc < 3)
	{
		cout << "Usage: " << argv[0] << " <filename> <bin/text>" << endl;
		return 0; 
	}

	stringstream outs;
	outs << argv[1] << ".balanced";
       	FILE * bFile = fopen (outs.str().c_str(),"wb");
	int32_t vid;
	int64_t edges;


	FILE * rFile;	// points to "past header" if the file is binary
	int seeklength = 0;
	HeaderInfo hdr = ParseHeader(string(argv[1]), rFile, seeklength);

	cout << "Size of Header: " << sizeof(hdr) << endl;
	cout << hdr.m << " " << hdr.n << " " << hdr.nnz << endl;
	char start[5] = "HKDT";
	fwrite(start, 4, 1, bFile); 
	fwrite(&hdr.version, sizeof(uint64_t), 1,bFile);
	fwrite(&hdr.objsize, sizeof(uint64_t), 1,bFile);
	fwrite(&hdr.format, sizeof(uint64_t), 1,bFile);
	fwrite(&hdr.m, sizeof(uint64_t), 1,bFile);
	fwrite(&hdr.n, sizeof(uint64_t), 1,bFile);
	fwrite(&hdr.nnz, sizeof(uint64_t), 1,bFile);

	vector<uint64_t> permutation(hdr.m, 0);
	my_iota(permutation.begin(), permutation.end(), 0);
	random_shuffle (permutation.begin(), permutation.end()); 

	if(rFile != NULL)
	{
		cout << "Reading binary" << endl;
		while(!feof(rFile))
		{
			TwitterInteraction twi;
			fread (&twi,sizeof(TwitterInteraction),1,rFile);	
			twi.from = permutation[twi.from];
			twi.to = permutation[twi.to];
			fwrite(&twi,sizeof(TwitterInteraction),1,bFile);	// write binary file	
		}
	}

	fclose(bFile);

	// test written file
	cout << "Reading first line of binary file..." << endl;
	bFile = fopen (outs.str().c_str(),"r");
	char begin[5];
	fread(begin, 4, 1, bFile);
	begin[4] = '\0';
	printf ("Header of : %s\n", begin);
	fclose(bFile);
	
	return 0;
}
