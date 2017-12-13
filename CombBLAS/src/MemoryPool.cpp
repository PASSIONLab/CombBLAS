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


#include "CombBLAS/MemoryPool.h"

using namespace std;

namespace combblas {

MemoryPool::MemoryPool(void * m_beg, size_t m_size):initbeg((char*)m_beg), initend(((char*)m_beg)+m_size)
{
	Memory m((char*) m_beg, m_size);
	freelist.push_back(m);
}

void * MemoryPool::alloc(size_t size)
{
	for(list<Memory>::iterator iter = freelist.begin(); iter != freelist.end(); ++iter)
	{
    	if ((*iter).size > size)	// return the first 'big enough' chunk of memory
		{
			char * free = (*iter).begin;
			(*iter).begin += size;		// modify the beginning of the remaining chunk
			(*iter).size -= size;		// modify the size of the remaning chunk

			return (void *) free;		// return the memory 
		}
	}
	cout << "No pinned memory available" << endl;
	return NULL;
}


void MemoryPool::dealloc(void * base, size_t size)
{
	if( ((char*) base) >= initbeg && (((char*)base) + size) < initend)
	{	
		list<Memory>::iterator titr = freelist.begin();	// trailing iterator
		list<Memory>::iterator litr = freelist.begin();	// leading iterator
		++litr;
		
		if( (char*)base < titr->begaddr()) 	// if we're inserting to the front of the list
		{
			if(titr->begaddr() == ((char*)base) + size)
			{
				titr->begin = (char*)base;
				titr->size += size;
			}
			else
			{
				Memory m((char*) base, size);
				freelist.insert(titr, m);
			}
		}
		else if( litr == freelist.end() )	// if we're inserting to the end of a 'single element list'
		{
			if(titr->endaddr() == (char*)base)
			{
				titr->size += size;
			}
			else
			{
				Memory m((char*) base, size);
				freelist.insert(litr, m);
			}	
		}
		else		// not a single element list, nor we're inserting to the front
		{
			// loop until you find the right spot
			while( litr != freelist.end() && litr->begaddr() < (char*)base)
			{  ++litr; 	++titr;	}

			//! defragment on the fly 
			//! by attaching to the previous chunk if prevchunk.endaddr equals newitem.beginaddr, or
			//! by attaching to the next available chunk if newitem.endaddr equals nextchunk.begaddr 	
			if(titr->endaddr() == (char*)base)
			{
				//! check the next chunk to see if we perfectly fill the hole 
				if( litr == freelist.end() || litr->begaddr() != ((char*)base) + size)
				{
					titr->size += size;
				}
				else
				{
					titr->size  += (size + litr->size);	// modify the chunk pointed by trailing iterator
					freelist.erase(litr); 			// delete the chunk pointed by the leading iterator
				}
			}
			else
			{
				if( litr == freelist.end() || litr->begaddr() != ((char*)base) + size)
				{
					Memory m((char*) base, size);
				
					//! Insert x before pos: 'iterator insert(iterator pos, const T& x)'
					freelist.insert(litr, m);					
				}
				else
				{
					litr->begin = (char*)base;
					litr->size += size;
				}					
			}
		}
	}
	else
	{
		cerr << "Memory starting at " << base << " and ending at " 
		<< (void*) ((char*) base + size) << " is out of pool bounds, cannot dealloc()" << endl;
	}
}

//! Dump the contents of the pinned memory
ofstream& operator<< (ofstream& outfile, const MemoryPool & mpool)
{
	int i = 0;
	for(list<Memory>::const_iterator iter = mpool.freelist.begin(); iter != mpool.freelist.end(); ++iter, ++i)
	{
		outfile << "Chunk " << i << " of size: " << (*iter).size << " starts:" <<  (void*)(*iter).begin 
			<< " and ends: " << (void*) ((*iter).begin + (*iter).size) << endl ; 
	}
	return outfile;
}

}
