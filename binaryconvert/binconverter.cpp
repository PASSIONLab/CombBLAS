#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>

#include <stdio.h>
#include <map>
#include <vector>
#include <sys/time.h>

using namespace std;


struct TwitterInteraction
{
	int32_t from;
	int32_t to;
	bool follow;
	int16_t retweets;
	time_t twtime;
};

struct Header
{
	uint64_t version;
	uint64_t objsize;
	uint64_t format;	// 0: binary, 1: ascii
	
	uint64_t m;
	uint64_t n;
	uint64_t nnz;
};


int main(int argc, char *argv[] )
{
	if(argc < 3)
	{
		cout << "Usage: " << argv[0] << " <filename> <bin/text>" << endl;
		return 0; 
	}

	stringstream outs;
	outs << argv[1] << ".converted";
       	FILE * bFile = fopen (outs.str().c_str(),"wb");
	int32_t vid;
	int64_t edges;
	char start[5] = "HKDT";
	Header hdr;
	hdr.version = 2;	// 2 means 0.2
	hdr.objsize = sizeof(TwitterInteraction);
	hdr.format = 0;	// binary
	hdr.m = vid;
	hdr.n = vid;
	hdr.nnz = edges;

	if(string(argv[2]) == string("text"))
	{
       		FILE * rFile = fopen (argv[1],"r");
		if(rFile != NULL)
		{
			cout << "Reading text file" << endl;

			size_t n=256;
			char * comment = (char*) malloc(n);
			int bytes_read = getline(&comment, &n, rFile);
			while(comment[0] == '%')
			{
				bytes_read = getline(&comment, &n, rFile);
			}
			stringstream ss;
			ss << string(comment);
			ss >> hdr.m >> hdr.n >> hdr.nnz;

			cout << "Size of Header: " << sizeof(hdr) << endl;
			cout << hdr.m << " " << hdr.n << " " << hdr.nnz << endl;
			fwrite(start, 4, 1, bFile); 
			fwrite(&hdr.version, sizeof(uint64_t), 1,bFile);
			fwrite(&hdr.objsize, sizeof(uint64_t), 1,bFile);
			fwrite(&hdr.format, sizeof(uint64_t), 1,bFile);
			fwrite(&hdr.m, sizeof(uint64_t), 1,bFile);
			fwrite(&hdr.n, sizeof(uint64_t), 1,bFile);
			fwrite(&hdr.nnz, sizeof(uint64_t), 1,bFile);

			struct tm timeinfo;
			int year, month, day, hour, min, sec;	
			while(!feof(rFile))
			{
				TwitterInteraction twi;
				int from, to, follow, retweets; 

				// example time format (for text): 2009-06-08 21:49:30
				bytes_read = getline(&comment, &n, rFile);
				cout << comment ;
				
				bytes_read = getline(&comment, &n, rFile);
				cout << comment ;

				bytes_read = getline(&comment, &n, rFile);
				cout << comment ;
				
				return 1;
				if(fscanf (rFile, "%d %d %d %d",&from,&to,&follow,&retweets) ==  0) 
				{
					cout << "breaking... from " << from << " to " << to << " follows? " << follow << ", retweets? " << retweets << endl;
					break;
				}
				else
				{
					cout << "all good" << endl;
				}
				
				if(twi.retweets > 0)
				{
					if(fscanf (rFile, " %d-%d-%d %d:%d:%d\n", &year, &month, &day, &hour, &min, &sec) == 0) 
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
				fwrite(&twi,sizeof(TwitterInteraction),1,bFile);	// write binary file	
			}
		}
	}
	else if(string(argv[2]) == string("bin"))
	{
       		FILE * rFile = fopen (argv[1],"rb");

		fread(&vid,sizeof(int32_t),1,rFile);
		fread(&edges,sizeof(int64_t),1,rFile);	

		hdr.m = vid;
		hdr.n = vid;
		hdr.nnz = edges;

		cout << "Size of Header: " << sizeof(hdr) << endl;
		fwrite(start, 4, 1, bFile); 
		fwrite(&hdr.version, sizeof(uint64_t), 1,bFile);
		fwrite(&hdr.objsize, sizeof(uint64_t), 1,bFile);
		fwrite(&hdr.format, sizeof(uint64_t), 1,bFile);
		fwrite(&hdr.m, sizeof(uint64_t), 1,bFile);
		fwrite(&hdr.n, sizeof(uint64_t), 1,bFile);
		fwrite(&hdr.nnz, sizeof(uint64_t), 1,bFile);

		if(rFile != NULL)
		{
			cout << "Reading binary" << endl;
			while(!feof(rFile))
			{
				TwitterInteraction twi;
				fread (&twi,sizeof(TwitterInteraction),1,rFile);	
				fwrite(&twi,sizeof(TwitterInteraction),1,bFile);	// write binary file	
			}
		}
	}
	else
	{
		cout << "What's your file format, dude?" << endl;
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
