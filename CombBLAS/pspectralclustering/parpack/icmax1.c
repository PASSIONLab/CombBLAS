/* icmax1.f -- translated by f2c (version 20050501).
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

integer icmax1_(integer *n, complex *cx, integer *incx)
{
    /* System generated locals */
    integer ret_val, i__1, i__2;
    real r__1;

    /* Local variables */
    static integer i__, ix;
    static real smax;


/*  -- LAPACK auxiliary routine (version 2.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd., */
/*     Courant Institute, Argonne National Lab, and Rice University */
/*     September 30, 1994 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ICMAX1 finds the index of the element whose real part has maximum */
/*  absolute value. */

/*  Based on ICAMAX from Level 1 BLAS. */
/*  The change is to use the 'genuine' absolute value. */

/*  Contributed by Nick Higham for use with CLACON. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The number of elements in the vector CX. */

/*  CX      (input) COMPLEX array, dimension (N) */
/*          The vector whose elements will be summed. */

/*  INCX    (input) INTEGER */
/*          The spacing between successive values of CX.  INCX >= 1. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function definitions .. */

/*     NEXT LINE IS THE ONLY MODIFICATION. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --cx;

    /* Function Body */
    ret_val = 0;
    if (*n < 1) {
	return ret_val;
    }
    ret_val = 1;
    if (*n == 1) {
	return ret_val;
    }
    if (*incx == 1) {
	goto L30;
    }

/*     CODE FOR INCREMENT NOT EQUAL TO 1 */

    ix = 1;
    smax = (r__1 = cx[1].r, dabs(r__1));
    ix += *incx;
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	i__2 = ix;
	if ((r__1 = cx[i__2].r, dabs(r__1)) <= smax) {
	    goto L10;
	}
	ret_val = i__;
	i__2 = ix;
	smax = (r__1 = cx[i__2].r, dabs(r__1));
L10:
	ix += *incx;
/* L20: */
    }
    return ret_val;

/*     CODE FOR INCREMENT EQUAL TO 1 */

L30:
    smax = (r__1 = cx[1].r, dabs(r__1));
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	i__2 = i__;
	if ((r__1 = cx[i__2].r, dabs(r__1)) <= smax) {
	    goto L40;
	}
	ret_val = i__;
	i__2 = i__;
	smax = (r__1 = cx[i__2].r, dabs(r__1));
L40:
	;
    }
    return ret_val;

/*     End of ICMAX1 */

} /* icmax1_ */

