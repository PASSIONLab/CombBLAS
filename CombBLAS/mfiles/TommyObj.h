#ifndef _TOMMY_OBJ_H_
#define _TOMMY_OBJ_H_

#include "compress_string.h"
#include "Tommy/tommyhashdyn.h"
using namespace std;

struct tommy_object {
    tommy_node node;
    uint32_t vid;
    string vname;
    
    tommy_object(uint32_t val, string name):vid(val)
    {
#ifdef COMPRESS_STRING
	vname = compress_string(name);
#else
	vname = name;
#endif
    }; // constructor

    string getIndex() const
    {
#ifdef COMPRESS_STRING
	return decompress_string(vname);
#else
	return vname;
#endif	
    } 
};


int compare(const void* arg, const void* obj)
{
    return *(const string*)arg != ((const tommy_object *)obj)->getIndex();
}

#endif
