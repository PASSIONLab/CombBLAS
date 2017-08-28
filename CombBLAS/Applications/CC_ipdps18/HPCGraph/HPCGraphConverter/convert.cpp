/*
Written by Ariful Azad, Lawrence Berkeley National Laboratory 
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <alloca.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iterator>


using namespace std;



int main(int argc, char** argv)
{


	if(argc<3)
	{
		cout << "Input and/or output filenames are missing!\n";
		cout << "Usage: convert inputFileName outputFileName \n";
		exit(1);
	}
	char * ifname = argv[1];
	char * ofname = argv[2];
	
	long count=0,i,j;
	long inp, m, sym;
	long numRow;
	double f;
	string s;
	ifstream inf;
	inf.open(ifname, ios::in);
	FILE* outf = fopen(ofname, "wb");
	char space[]= " ";
	
	if(inf.is_open())
	{
		size_t found1, found2, found3;
		getline(inf,s);
		found1 = s.find("pattern");
		if (found1 != string::npos)
			m = 2;
		else
			m = 3;
		found1 = s.find("symmetric");
		found2 = s.find("hermitian");
		found3 = s.find("skew-symmetric");
		if (found1 != string::npos || found2 != string::npos || found3 != string::npos)
			sym = 1;
		else
			sym = 0;
		while(inf.peek()=='%')
			getline(inf,s);
		
		inf>>inp;
		numRow=inp;
		inf>>inp;
		//numCol=inp;
		inf>>inp;
				
		count=inp;
		cout << count << endl;

		while(count>0)
		{
			inf>>i;
			inf>>j;
			i--;
			j--;
			fwrite(&i, sizeof(long), 1, outf);
			fwrite(space, sizeof(space), 1, outf);
			fwrite(&j, sizeof(long), 1, outf);
			fwrite(space, sizeof(space), 1, outf);

			if(m==3)
			{
				inf>>f;
			}
			
			if (sym && i != j)
			{
				fprintf(outf, "%ld %ld ", j-1 , numRow+i-1);
				long i1 = j;
				long j1 = numRow+i;
				fwrite(&i1, sizeof(long), 1, outf);
				fwrite(space, sizeof(space), 1, outf);
				fwrite(&j1, sizeof(long), 1, outf);
				fwrite(space, sizeof(space), 1, outf);
			}
			count--;
		}
		inf.close();
		fclose(outf);
		
	}
	else
    {
        printf("file can not be opened \n");
        exit(-1);
    }
		
}


