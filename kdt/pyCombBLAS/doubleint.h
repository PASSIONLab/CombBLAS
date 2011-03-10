#ifndef DOUBLEINT_H
#define DOUBLEINT_H

#include <iostream>
#include <math.h>
#include <limits>

using namespace std;

struct doubleint 
{
	double d;
	
	doubleint():					d(0) {}
	doubleint(double v):			d(v) {}
	doubleint(const doubleint & v):	d(v.d) {}
	
	bool is_nan() const							{ return isnan(d); }
	static doubleint nan()						{ return doubleint(numeric_limits<double>::quiet_NaN()); } 
	
	//operator unsigned long() const				{ double ret = (d < 0 ? std::ceil(d-0.5) : std::floor(d+0.5)); return static_cast<unsigned long>(ret); }
	//operator long() const						{ double ret = (d < 0 ? std::ceil(d-0.5) : std::floor(d+0.5)); return static_cast<long>(ret); }
	operator int64_t() const					{ double ret = (d < 0 ? std::ceil(d-0.5) : std::floor(d+0.5)); return static_cast<int64_t>(ret); }
	operator uint64_t() const					{ double ret = (d < 0 ? std::ceil(d-0.5) : std::floor(d+0.5)); return static_cast<uint64_t>(ret); }
	operator int() const						{ double ret = (d < 0 ? std::ceil(d-0.5) : std::floor(d+0.5)); return static_cast<int>(ret); }
	operator unsigned int() const				{ double ret = (d < 0 ? std::ceil(d-0.5) : std::floor(d+0.5)); return static_cast<unsigned int>(ret); }
	operator bool() const						{ double ret = (d < 0 ? std::ceil(d-0.5) : std::floor(d+0.5)); return static_cast<bool>(ret); }
	operator float() const						{ return static_cast<float>(d); }
	operator double() const						{ return d; }
	
	double operator=(const double v) 			{ d = v; return d; }
	int operator=(const int v) 					{ d = static_cast<int>(v); return static_cast<int>(d); }
	int64_t operator=(const int64_t v) 			{ d = static_cast<double>(v); return static_cast<int64_t>(d); }
	doubleint& operator=(const doubleint& v) 	{ d = v.d; return *this; }

	// unary operations: ++, --, ~, !
	doubleint& operator++()						{ d += 1; return *this; }
	doubleint& operator++(int)					{ d += 1; return *this; }
	doubleint& operator--()						{ d -= 1; return *this; }
	doubleint& operator--(int)					{ d -= 1; return *this; }

	doubleint operator-()						{ return doubleint(-d); }
	doubleint operator~() const					{ return doubleint(~static_cast<int64_t>(d)); }
	bool operator!() const						{ return !d; }
};

// Binary operators. The preprocessor can help save a lot of monotony.

#define BINARY_OPERATOR_INTONLY(op, returntype, doubleinttype) \
inline returntype operator op(const doubleint& lhs, const doubleint& rhs)	{ return returntype((doubleinttype)lhs.d op (doubleinttype)rhs.d ); } \
inline returntype operator op(const int lhs, const doubleint& rhs)			{ return returntype(lhs op (doubleinttype)rhs.d ); } \
inline returntype operator op(const int64_t lhs, const doubleint& rhs)		{ return returntype(lhs op (doubleinttype)rhs.d ); } \
inline returntype operator op(const uint64_t lhs, const doubleint& rhs)		{ return returntype(lhs op (doubleinttype)rhs.d ); } \
inline returntype operator op(const doubleint& lhs, const int rhs)			{ return returntype((doubleinttype)lhs.d op rhs ); } \
inline returntype operator op(const doubleint& lhs, const int64_t rhs)		{ return returntype((doubleinttype)lhs.d op rhs ); } \
inline returntype operator op(const doubleint& lhs, const uint64_t rhs)		{ return returntype((doubleinttype)lhs.d op rhs ); } \


#define BINARY_OPERATOR(op, returntype, doubleinttype) \
inline returntype operator op(const double lhs, const doubleint& rhs)		{ return returntype(lhs op (doubleinttype)rhs.d ); } \
inline returntype operator op(const doubleint& lhs, const double rhs)		{ return returntype((doubleinttype)lhs.d op rhs ); } \
BINARY_OPERATOR_INTONLY(op, returntype, doubleinttype)



#define BINARY_OPERATOR_EQ_INTONLY(op, opeq, returntype, doubleinttype) \
inline doubleint& operator opeq(doubleint& lhs, const doubleint& rhs)		{ lhs.d = static_cast<double>(((doubleinttype)lhs.d) op ((doubleinttype)rhs.d)); return lhs; } \
inline doubleint& operator opeq(doubleint& lhs, const int rhs)				{ lhs.d = static_cast<double>(((doubleinttype)lhs.d) op rhs); return lhs; } \
inline doubleint& operator opeq(doubleint& lhs, const int64_t rhs)			{ lhs.d = static_cast<double>(((doubleinttype)lhs.d) op rhs); return lhs; } \
inline doubleint& operator opeq(doubleint& lhs, const uint64_t rhs)			{ lhs.d = static_cast<double>(((doubleinttype)lhs.d) op rhs); return lhs; } \

#define BINARY_OPERATOR_EQ(op, opeq, returntype, doubleinttype) \
inline doubleint& operator opeq(doubleint& lhs, const double rhs)			{ lhs.d opeq rhs; return lhs; } \
BINARY_OPERATOR_EQ_INTONLY(op, opeq, returntype, doubleinttype)


#define BINARY_OPERATOR_LOGICAL(op) \
inline bool operator op(const bool lhs, const doubleint& rhs)				{ return (lhs op (int64_t)rhs.d ); } \
inline bool operator op(const doubleint& lhs, const bool rhs)				{ return ((int64_t)lhs.d op rhs ); } \
inline bool operator op(const doubleint& lhs, const doubleint& rhs)			{ return ((int64_t)lhs.d op (int64_t)rhs.d ); } \


BINARY_OPERATOR(+, doubleint, double)
BINARY_OPERATOR(-, doubleint, double)
BINARY_OPERATOR(*, doubleint, double)
BINARY_OPERATOR(/, doubleint, double) //////////////////////////////// make sure this works as expected with integer division. probably requires floor() of result
BINARY_OPERATOR_INTONLY(%, doubleint, int64_t)

BINARY_OPERATOR_EQ(+, +=, doubleint, double)
BINARY_OPERATOR_EQ(-, -=, doubleint, double)
BINARY_OPERATOR_EQ(*, *=, doubleint, double)
BINARY_OPERATOR_EQ(/, /=, doubleint, double)
BINARY_OPERATOR_EQ_INTONLY(%, %=, doubleint, int64_t)


BINARY_OPERATOR_INTONLY(|, doubleint, int64_t)
BINARY_OPERATOR_INTONLY(&, doubleint, int64_t)
BINARY_OPERATOR_INTONLY(^, doubleint, int64_t)
BINARY_OPERATOR_INTONLY(<<, doubleint, int64_t)
BINARY_OPERATOR_INTONLY(>>, doubleint, int64_t)

BINARY_OPERATOR_EQ_INTONLY(|, |=, doubleint, int64_t)
BINARY_OPERATOR_EQ_INTONLY(&, &=, doubleint, int64_t)
BINARY_OPERATOR_EQ_INTONLY(^, ^=, doubleint, int64_t)
BINARY_OPERATOR_EQ_INTONLY(<<, <<=, doubleint, int64_t)
BINARY_OPERATOR_EQ_INTONLY(>>, >>=, doubleint, int64_t)

BINARY_OPERATOR(==, bool, double)
BINARY_OPERATOR(!=, bool, double)
BINARY_OPERATOR(<, bool, double)
BINARY_OPERATOR(<=, bool, double)
BINARY_OPERATOR(>, bool, double)
BINARY_OPERATOR(>=, bool, double)

BINARY_OPERATOR_LOGICAL(&&)
BINARY_OPERATOR_LOGICAL(||)

template <typename c, typename t>
inline std::basic_ostream<c,t>& operator<<(std::basic_ostream<c,t>& lhs, const doubleint& rhs) { return lhs << (rhs.d); }

template <typename c, typename t>
inline std::basic_istream<c,t>& operator>>(std::basic_istream<c,t>& lhs, const doubleint& rhs) { return lhs >> rhs.d; }



// From CombBLAS/promote.h:
/*
template <class T1, class T2>
struct promote_trait  { };

#define DECLARE_PROMOTE(A,B,C)                  \
    template <> struct promote_trait<A,B>       \
    {                                           \
        typedef C T_promote;                    \
    };
*/
DECLARE_PROMOTE(doubleint, doubleint, doubleint)
DECLARE_PROMOTE(double, doubleint, doubleint)
DECLARE_PROMOTE(doubleint, double, doubleint)
DECLARE_PROMOTE(int, doubleint, doubleint)
DECLARE_PROMOTE(doubleint, int, doubleint)
DECLARE_PROMOTE(bool, doubleint, doubleint)
DECLARE_PROMOTE(doubleint, bool, doubleint)
DECLARE_PROMOTE(int64_t, doubleint, doubleint)
DECLARE_PROMOTE(doubleint, int64_t, doubleint)


// From CombBLAS/MPIType.h
extern MPI::Datatype doubleint_MPI_datatype; // Defined in pyCombBLAS.cpp
template<> MPI::Datatype MPIType< doubleint >( void );

/*
inline void doubleint_plus(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype)
{
	doubleint* in = (doubleint*)invec;
	doubleint* inout = (doubleint*)inoutvec;
	
	for (int i = 0; i < *len; i++)
	{
		inout[i] = in[i] + inout[i];
	}
}

template<> struct MPIOp< std::plus<doubleint>, doubleint >
{
	static MPI_Op op()
	{
		MPI_Op o;
		MPI_Op_create(doubleint_plus, true, &o);
		return o;
	}
};
*/

#endif
