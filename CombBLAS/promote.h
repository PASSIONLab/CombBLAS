#ifndef _PROMOTE_H_
#define _PROMOTE_H_

template <class T1, class T2>
struct promote_trait  { };

#define DECLARE_PROMOTE(A,B,C)                  \
    template <> struct promote_trait<A,B>       \
    {                                           \
        typedef C T_promote;                    \
    };
DECLARE_PROMOTE(int64_t, bool, int64_t);
DECLARE_PROMOTE(int64_t, int, int64_t);
DECLARE_PROMOTE(bool, int64_t, int64_t);
DECLARE_PROMOTE(int, int64_t, int64_t);
DECLARE_PROMOTE(int64_t, int64_t, int64_t);
DECLARE_PROMOTE(int, bool,int);
DECLARE_PROMOTE(short, bool,short);
DECLARE_PROMOTE(unsigned, bool, unsigned);
DECLARE_PROMOTE(float, bool, float);
DECLARE_PROMOTE(double, bool, double);
DECLARE_PROMOTE(unsigned long long, bool, unsigned long long);
DECLARE_PROMOTE(bool, int, int);
DECLARE_PROMOTE(bool, short, short);
DECLARE_PROMOTE(bool, unsigned, unsigned);
DECLARE_PROMOTE(bool, float, float);
DECLARE_PROMOTE(bool, double, double);
DECLARE_PROMOTE(bool, unsigned long long, unsigned long long);
DECLARE_PROMOTE(bool, bool, bool);
DECLARE_PROMOTE(float, int, float);
DECLARE_PROMOTE(double, int, double);
DECLARE_PROMOTE(int, float, float);
DECLARE_PROMOTE(int, double, double);
DECLARE_PROMOTE(float, float, float);
DECLARE_PROMOTE(double, double, double);
DECLARE_PROMOTE(int, int, int);
DECLARE_PROMOTE(unsigned, unsigned, unsigned);
DECLARE_PROMOTE(unsigned long long, unsigned long long, unsigned long long);

#endif
