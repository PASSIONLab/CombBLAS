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


#ifndef _HEAP_ENTRY_H
#define _HEAP_ENTRY_H

namespace combblas {

template <class IT, class NT>
class HeapEntry
{
public:
	HeapEntry() {};
	HeapEntry(IT mykey, IT myrunr, NT mynum): key(mykey), runr(myrunr),num(mynum){}
	IT key;	
	IT runr;	 
	NT num;	

	// Operators are swapped for performance
	// If you want/need to convert them back to their normal definitions, don't forget to add
	// "greater< HeapEntry<T> >()" optional parameter to all the heap operations operating on HeapEntry<T> objects.
	// For example: push_heap(heap, heap + kisect, greater< HeapEntry<T> >());

	bool operator > (const HeapEntry & rhs) const
	{ return (key < rhs.key); }
	bool operator < (const HeapEntry & rhs) const
	{ return (key > rhs.key); }
	bool operator == (const HeapEntry & rhs) const
	{ return (key == rhs.key); }
};

}

#endif

