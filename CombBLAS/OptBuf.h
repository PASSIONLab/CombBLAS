#ifndef _OPT_BUF_H
#define _OPT_BUF_H

/**
  * This special data structure is used for optimizing BFS iterations
  * by providing a fixed sized buffer for communication
  * the contents of the buffer are irrelevant until SpImpl:SpMXSpV starts
  * hence the copy constructor that doesn't copy contents
  */
template <class IT, class NT>
class OptBuf
{
public:
	OptBuf(): p_c(0), totmax(0) {};
	void Set(const vector<int> & maxsizes) 
	{
		p_c =  maxsizes.size(); 
		totmax = accumulate(maxsizes.begin(), maxsizes.end(), 0);
		inds = new IT[totmax];
		nums = new NT[totmax];
		dspls = new int[p_c]();
		partial_sum(maxsizes.begin(), maxsizes.end()-1, dspls+1);			
	};
	~OptBuf()
	{
		if(totmax > 0)
		{
			delete [] inds;
			delete [] nums;
		}
		if(p_c > 0)
			delete [] dspls;
	}
	OptBuf(const OptBuf<IT,NT> & rhs)
	{
		p_c = rhs.p_c;
		totmax = rhs.totmax;
		inds = new IT[totmax];
		nums = new NT[totmax];
		dspls = new int[p_c]();	
	}
	OptBuf<IT,NT> & operator=(const OptBuf<IT,NT> & rhs)
	{
		if(this != &rhs)
		{
			if(totmax > 0)
			{
				delete [] inds;
				delete [] nums;
			}
			if(p_c > 0)
				delete [] dspls;
	
			p_c = rhs.p_c;
			totmax = rhs.totmax;
			inds = new IT[totmax];
			nums = new NT[totmax];
			dspls = new int[p_c]();	
		}
		return *this;
	}
	
	IT * inds;	
	NT * nums;	
	int * dspls;
	int p_c;
	int totmax;
};

#endif

