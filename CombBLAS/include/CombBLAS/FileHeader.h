/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
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


#ifndef _COMBBLAS_FILE_HEADER_
#define _COMBBLAS_FILE_HEADER_

#include "CombBLAS.h"

namespace combblas {

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
	
// cout's are OK because ParseHeader is run by a single processor only
inline HeaderInfo ParseHeader(const std::string & inputname, FILE * & f, int & seeklength)
{
	f = fopen(inputname.c_str(), "rb");
	HeaderInfo hinfo;
	memset(&hinfo, 0, sizeof(hinfo));
	if(!f)
	{
		std::cerr << "Problem reading binary input file\n";
		f = NULL;
		return hinfo;
	}
	char fourletters[5];
	size_t result = fread(fourletters, sizeof(char), 4, f);
	fourletters[4] = '\0';
	if (result != 4) { std::cout << "Error in fread of header, only " << result << " entries read" << std::endl; return hinfo;}

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
	if(std::accumulate(results,results+6,0) != 6)
	{
		std::cout << "The required 6 fields (version, objsize, format, m,n,nnz) are not read" << std::endl;
		std::cout << "Only " << std::accumulate(results,results+6,0) << " fields are read" << std::endl;
	} 
	else
	{
	#ifdef DEBUG
    std::cout << "Version " << hinfo.version << ", object size " << hinfo.objsize << std::endl;
    std::cout << "Rows " << hinfo.m << ", columns " << hinfo.m << ", nonzeros " << hinfo.nnz << std::endl;
	#endif
	}

	seeklength = 4 + 6 * sizeof(uint64_t);
	return hinfo;
}
				  

}
#endif 

