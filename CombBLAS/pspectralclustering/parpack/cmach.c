/* cmach.f -- translated by f2c (version 20050501).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"

/* Table of constant values */

static complex c_b4 = {1.f,0.f};

doublereal cmach_(integer *job)
{
    /* System generated locals */
    real ret_val;
    complex q__1, q__2;

    /* Builtin functions */
    void c_div(complex *, complex *, complex *);
    double sqrt(doublereal);

    /* Local variables */
    static real s, eps, huge__, tiny;


/*     smach computes machine parameters of floating point */
/*     arithmetic for use in testing only.  not required by */
/*     linpack proper. */

/*     if trouble with automatic computation of these quantities, */
/*     they can be set by direct assignment statements. */
/*     assume the computer has */

/*        b = base of arithmetic */
/*        t = number of base  b  digits */
/*        l = smallest possible exponent */
/*        u = largest possible exponent */

/*     then */

/*        eps = b**(1-t) */
/*        tiny = 100.0*b**(-l+t) */
/*        huge = 0.01*b**(u-t) */

/*     dmach same as smach except t, l, u apply to */
/*     double precision. */

/*     cmach same as smach except if complex division */
/*     is done by */

/*        1/(x+i*y) = (x-i*y)/(x**2+y**2) */

/*     then */

/*        tiny = sqrt(tiny) */
/*        huge = sqrt(huge) */


/*     job is 1, 2 or 3 for epsilon, tiny and huge, respectively. */



    eps = 1.f;
L10:
    eps /= 2.f;
    s = eps + 1.f;
    if (s > 1.f) {
	goto L10;
    }
    eps *= 2.f;
    ret_val = eps;
    if (*job == 1) {
	return ret_val;
    }

    s = 1.f;
L20:
    tiny = s;
    s /= 16.f;
    if (s * 1.f != 0.f) {
	goto L20;
    }
    tiny = tiny / eps * 100.f;
    q__2.r = tiny, q__2.i = 0.f;
    c_div(&q__1, &c_b4, &q__2);
    s = q__1.r;
    if (s != 1.f / tiny) {
	tiny = sqrt(tiny);
    }
    huge__ = 1.f / tiny;
    if (*job == 1) {
	ret_val = eps;
    }
    if (*job == 2) {
	ret_val = tiny;
    }
    if (*job == 3) {
	ret_val = huge__;
    }
    return ret_val;
} /* cmach_ */

