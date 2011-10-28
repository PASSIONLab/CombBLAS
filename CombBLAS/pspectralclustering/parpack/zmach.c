/* zmach.f -- translated by f2c (version 20050501).
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

static doublecomplex c_b4 = {1.,0.};

doublereal zmach_(integer *job)
{
    /* System generated locals */
    doublereal ret_val;
    doublecomplex z__1, z__2;

    /* Builtin functions */
    void z_div(doublecomplex *, doublecomplex *, doublecomplex *);
    double sqrt(doublereal);

    /* Local variables */
    static doublereal s, eps, huge__, tiny;


/*     double complex floating point arithmetic constants. */
/*     for the linpack test drivers only. */
/*     not used by actual linpack subroutines. */

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


    eps = 1.;
L10:
    eps /= 2.;
    s = eps + 1.;
    if (s > 1.) {
	goto L10;
    }
    eps *= 2.;

    s = 1.;
L20:
    tiny = s;
    s /= 16.;
    if (s * 1. != 0.) {
	goto L20;
    }
    tiny /= eps;
    z__2.r = tiny, z__2.i = 0.;
    z_div(&z__1, &c_b4, &z__2);
    s = z__1.r;
    if (s != 1. / tiny) {
	tiny = sqrt(tiny);
    }
    huge__ = 1. / tiny;

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
} /* zmach_ */

