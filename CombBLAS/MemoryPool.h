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


#ifndef _MEMORY_POOL_H
#define _MEMORY_POOL_H

#include <iostream>
#include <list>
#include <new>			// For "placement new" (classes using this memory pool may need it)
#include <fstream>

namespace combblas {

//! EqualityComparable memory chunk object
//! Two memory chunks are considered equal if either their beginaddr or the endaddr are the same (just to facilitate deallocation)
class Memory
{
	public:
	Memory(char * m_beg, size_t m_size): begin(m_beg),size(m_size)
	{};
	
	char * begaddr() { return begin; }
	char * endaddr() { return begin + size; }

	bool operator < (const Memory & rhs) const
	{ return (begin < rhs.begin); }
	bool operator == (const Memory & rhs) const
	{ return (begin == rhs.begin); }
	
	char * begin;
	size_t size;
};


/**
  * \invariant Available memory addresses will be sorted w.r.t. their starting positions
  * \invariant At least one element exists in the freelist at any time.
  * \invariant Defragment on the fly: at any time, NO two consecutive chunks with chunk1.endaddr equals chunk2.begaddr exist
  */ 
class MemoryPool
{
public:
	MemoryPool(void * m_beg, size_t m_size);

	void * alloc(size_t size);
	void dealloc (void * base, size_t size);

	friend std::ofstream& operator<< (std::ofstream& outfile, const MemoryPool & mpool);	

private:
	// std::list is a doubly linked list (i.e., a Sequence that supports both forward and backward traversal)
	std::list<Memory> freelist;
	char * initbeg;
	char * initend; 
};


}

#endif
