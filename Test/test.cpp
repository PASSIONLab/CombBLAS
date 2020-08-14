#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <sys/time.h> // for gettimeofday
#include <tr1/tuple>
#include "promote.h"
#include "Semirings.h"
#include "Deleter.h"
#include <ext/numeric>

using namespace std;
#define TESTSIZE 100000000

// accumulate with a function object or pointer
template<typename _InputIterator, typename _Tp, typename _BinaryOperation>
_Tp fpaccumulate(_InputIterator __first, _InputIterator __last, _Tp __init,
               _BinaryOperation __binary_op)
{
      for (; __first != __last; ++__first)
        __init = __binary_op(__init, *__first);
      return __init;    
}

// accumulate through a semiring
template<typename SR, typename _InputIterator, typename _Tp>
_Tp sraccumulate(_InputIterator __first, _InputIterator __last, _Tp __init)
{
      for (; __first != __last; ++__first)
        __init = SR::add(__init, *__first);
      return __init;    
}


// This is if you want something akin to covariant return types with CRTP
template <class D, class B> 
class clonable: B 
{ 
public: 
	D * clone () const 
	{ 
		return static_cast <D*> (this-> do_clone ()); 
	} 
private: 
	virtual clonable * do_clone () const 
	{ 
		return new D (static_cast <const D&> (* this)); 
	} 
}; 

class B 
{
public: 
	B * clone () const
	{
		return do_clone ();
	} 
private: 
	virtual B * do_clone () const = 0; 
}; 


template <class T, class DER>
class Base
{
public:
	~Base() 
	{
		cout << "Destructing base" << endl;
	};
	Base Square()
	{
		return static_cast<DER*>(this)->Square();
	}
	void Create(T obj)
	{
		cout << "Creating base with element " << obj << endl;
		static_cast<DER*>(this)->CreateImpl(obj);
	}	
	void Print()
	{
		static_cast<DER*>(this)->Print();
	}		
};

template <class T>
class Derived : public Base < T, Derived<T> >
{
public:
	~Derived() 
	{
		cout << "Destructing derived" << endl;
	};

	void CreateImpl(T obj)
	{
		cout << "Creating derived with element " << obj << endl;
		ele = obj;
	}

	Base< T, Derived<T> > Square()
	{
		Base< T, Derived<T> > mybase;
		mybase.Create(ele*ele);
		return mybase;
	}
	void Print()
	{
		cout << "Element is " << ele << endl;
	}  
private:
	T ele;
};


template <class T, int N>
class Dummy
{
public:
	Dummy()
	{
		cout << "Constructing" << endl;
		vec = new vector<T>(N);
		__gnu_cxx::iota(vec->begin(), vec->end(), 1);
	}
	Dummy(const Dummy<T,N> & rhs)
	{
		cout << "Copy constructing" << endl;
		vec = new vector<T>(*rhs.vec);
	}
	Dummy<T,N> & operator=(const Dummy<T,N> & rhs)
	{
		cout << "Assigning" << endl;
		if(this != &rhs)
		{
			if(vec != NULL)		// unnecessary check, delete on NULL doesn't fail
				delete vec;
			if(rhs.vec != NULL)
				vec =  new vector<T>(*rhs.vec);
		}
		cout << "Assignment returning" << endl;
		return *this;
	}
	
	template <typename SR>
	void EleAdd(const Dummy<T,N> & rhs)
	{
		// No need to check size as it won't compile without N being the same	
		for(int i=0; i < vec->size(); i++)
		{
			(*vec)[i] = SR::add((*rhs.vec)[i], (*vec)[i]);
		}
	}

	void Double()
	{
		typedef PlusTimesSRing<T,T> PT;

		Dummy<T,N> duplicate(*this);
		EleAdd< PT > (duplicate);
	}

	~Dummy()
	{
		cout << "Destructing object with first element " << (*vec)[0] << endl;
		delete vec;
	}
	void Reset()
	{
		*this = Dummy<T,N> ();
	}
	void Print()
	{
		for(int i=0; i<N; ++i)
			cout << (*vec)[i] << " ";
		cout << endl;
	}

	vector<T> * vec;
};


// A templated assignment operator
class AClass {
public:
    template <typename T>
    AClass& operator=(T val) {
        ostringstream oss;
        oss << val;
        m_value = oss.str();
        return *this;
    }
    string const& str() const { return m_value; }
private:
    string m_value;
};

ostream& operator<<(ostream& os, AClass const& obj) {
    os << obj.str();
    return os;
}


template <typename T>
T add_func (const T & a, const T & b)
{
	return a+b;
}


struct mystruct 
{
	double first;
	double second;
	double third;
};

struct StrOfArrays
{
	StrOfArrays(int size)
	{
		firsts = new double[size];
		seconds = new double[size];
		thirds = new double[size];
	}
	~StrOfArrays()
	{
		DeleteAll(firsts, seconds, thirds);
	}
	double * firsts;
	double * seconds;
	double * thirds;
};
	


int main()
{
	cout << sizeof(tr1::tuple<int, int, double>) << endl;

	int * arrint = new int[5];
	double * arrdouble = new double[10];
	float * arrfloat = new float[5];

	__gnu_cxx::iota(arrfloat, arrfloat+5,1);
	transform(arrfloat, arrfloat+5,arrfloat, bind1st(divides<float>(), 1)); 
	for(int i=0; i<5; ++i)
                cout << arrfloat[i] << " ";
        cout << endl;

	DeleteAll(arrint, arrdouble, arrfloat);
	
	
	int * A = new int[5];
	__gnu_cxx::iota(A, A+5, 1);
	
	transform(A, A+5, A, bind2nd(minus<int>(), 10));
	for(int i=0; i<5; ++i)
		cout << A[i] << " ";
	cout << endl;

	delete [] A;

	Dummy<int, 7> BushJr;	
	Dummy<int, 7> BushSr = BushJr;
	(*(BushJr.vec))[0] = 100;

	BushSr.Print();
	BushJr.Print();
	
	BushJr.EleAdd< PlusTimesSRing<int, int> > (BushSr);
	BushJr.Print();

	BushJr.Double();
	BushJr.Print();

	BushJr.Reset();
	BushJr.Print();
	
	vector<int> v = *(BushJr.vec);
	cout << v.size() << " elements, occupying "<< sizeof(v) << " bytes"<< endl;
	
	Base< int, Derived<int> > * mybase = new Base< int, Derived<int> >();
	mybase->Create(5);
	Base< int, Derived<int> > squared = mybase->Square();
	squared.Print();

	Derived<int> & casted = static_cast< Derived<int> & > (squared);
	
	int a = 5; 
	float b = 6.7;
	typedef promote_trait<int,float>::T_promote T_promote;
	T_promote result = a + b;
	cout << result << endl; 

	vector<int>(3).swap(v);
	cout << "size of v: " << v.size() <<" , capacity of v: " << v.capacity() << endl;

	cout << "******* vector performance test *******" << endl;
	int * source = new int[TESTSIZE];
	__gnu_cxx::iota(source, source+TESTSIZE, 1);
	vector<int> destination(TESTSIZE);
	vector<int> inserted;

	timeval tim;		
 	gettimeofday(&tim, NULL);
 	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	std::copy(source, source+TESTSIZE, destination.begin());

	gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed for std::copy\n", t2-t1);
	

	gettimeofday(&tim, NULL);
        t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	inserted.insert(inserted.begin(), source, source+TESTSIZE);

	gettimeofday(&tim, NULL);
        t2=tim.tv_sec+(tim.tv_usec/1000000.0);	
        printf("%.6lf seconds elapsed with insert\n", t2-t1);


	gettimeofday(&tim, NULL);
        t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	memcpy(&destination[0], source, TESTSIZE*sizeof(int));

	gettimeofday(&tim, NULL);
        t2=tim.tv_sec+(tim.tv_usec/1000000.0);	
        printf("%.6lf seconds elapsed with memcpy\n", t2-t1);

	
	gettimeofday(&tim, NULL);
        t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	for (int i=0; i< TESTSIZE; ++i)
		destination[i] = source[i];

	gettimeofday(&tim, NULL);
        t2=tim.tv_sec+(tim.tv_usec/1000000.0);	
        printf("%.6lf seconds elapsed with loops\n", t2-t1);

	delete [] source;

	cout << "******* accumulate performance test *******" << endl;

 	gettimeofday(&tim, NULL);
 	t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	int acc = fpaccumulate(destination.begin(), destination.end(), 0, plus<int>());

	gettimeofday(&tim, NULL);
	t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed with plus<int> function object, gave result: %d\n", t2-t1, acc);

	gettimeofday(&tim, NULL);
 	t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	typedef PlusTimesSRing<int, int> PT;
	acc = sraccumulate< PT > (destination.begin(), destination.end(), 0);

	gettimeofday(&tim, NULL);
	t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed with SR::add static method, gave result: %d\n", t2-t1, acc);


	gettimeofday(&tim, NULL);
 	t1=tim.tv_sec+(tim.tv_usec/1000000.0);

	acc = fpaccumulate(destination.begin(), destination.end(), 0, &add_func<int>);

	gettimeofday(&tim, NULL);
	t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed with function pointer, gave result: %d\n", t2-t1, acc);


	cout << "******* array of structs test *******" << endl;
	
	int * linear = new int[TESTSIZE];
	int * random = new int[TESTSIZE];
	__gnu_cxx::iota(linear, linear+TESTSIZE, 0);
	__gnu_cxx::iota(random, random+TESTSIZE, 0);
	random_shuffle(random, random+TESTSIZE);
	
	
	StrOfArrays strofarrays (TESTSIZE);
	mystruct * arrayofstrs = new mystruct[TESTSIZE];

	for(int i=0; i<TESTSIZE; ++i)
	{
		arrayofstrs[i].first = 1;
		arrayofstrs[i].second = 2;
		arrayofstrs[i].third = 3;

		strofarrays.firsts[i] = 1;
		strofarrays.seconds[i] = 2;
		strofarrays.thirds[i] = 3;
	}
	
	gettimeofday(&tim, NULL);
 	t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	for(int i=0; i<TESTSIZE; ++i)
	{
		arrayofstrs[linear[i]].first += 1;
		arrayofstrs[linear[i]].second += 2;
		arrayofstrs[linear[i]].third += 3;
	}
	gettimeofday(&tim, NULL);
 	t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed for streaming array of structs\n", t2-t1);

	gettimeofday(&tim, NULL);
 	t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	for(int i=0; i<TESTSIZE; ++i)
	{
		strofarrays.firsts[linear[i]] += 1;
		strofarrays.seconds[linear[i]] += 2;
		strofarrays.thirds[linear[i]] += 3;
	}
	gettimeofday(&tim, NULL);
 	t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("%.6lf seconds elapsed for streaming struct of arrays\n", t2-t1);

	gettimeofday(&tim, NULL);
        t1=tim.tv_sec+(tim.tv_usec/1000000.0);
        for(int i=0; i<TESTSIZE; ++i)
        {
                arrayofstrs[random[i]].first += 1;
                arrayofstrs[random[i]].second += 2;
                arrayofstrs[random[i]].third += 3;
        }
        gettimeofday(&tim, NULL);
        t2=tim.tv_sec+(tim.tv_usec/1000000.0);
        printf("%.6lf seconds elapsed for randomly accessing array of structs\n", t2-t1);

        gettimeofday(&tim, NULL);
        t1=tim.tv_sec+(tim.tv_usec/1000000.0);
        for(int i=0; i<TESTSIZE; ++i)
        {
                strofarrays.firsts[random[i]] += 1;
                strofarrays.seconds[random[i]] += 2;
                strofarrays.thirds[random[i]] += 3;
        }
        gettimeofday(&tim, NULL);
        t2=tim.tv_sec+(tim.tv_usec/1000000.0);
        printf("%.6lf seconds elapsed for randomly accessing struct of arrays\n", t2-t1);

	delete mybase;
	
	return 0;
}
