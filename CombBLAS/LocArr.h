#ifndef _LOC_ARR_H_
#define _LOC_ARR_H_


template<class V, class C>
struct LocArr
{
	LocArr(V * myaddr, C mycount): addr(myaddr ), count(mycount){}
	
	V * addr;
	C count;
}

template<class IT, class NT>
struct Arr
{
	vector< LocArr<IT,IT> > indarrs;
	vector< LocArr<NT,IT> > numarrs;	
}

#endif

