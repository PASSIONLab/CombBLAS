/* clartg.f -- translated by f2c (version 20050501).
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

/* Subroutine */ int clartg_(complex *f, complex *g, real *cs, complex *sn, 
	complex *r__)
{
    /* System generated locals */
    real r__1, r__2;
    complex q__1, q__2, q__3;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);
    double c_abs(complex *), r_imag(complex *), sqrt(doublereal);

    /* Local variables */
    static real d__, f1, f2, g1, g2, fa, ga, di;
    static complex fs, gs, ss;


/*  -- LAPACK auxiliary routine (version 2.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd., */
/*     Courant Institute, Argonne National Lab, and Rice University */
/*     September 30, 1994 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CLARTG generates a plane rotation so that */

/*     [  CS  SN  ]     [ F ]     [ R ] */
/*     [  __      ]  .  [   ]  =  [   ]   where CS**2 + |SN|**2 = 1. */
/*     [ -SN  CS  ]     [ G ]     [ 0 ] */

/*  This is a faster version of the BLAS1 routine CROTG, except for */
/*  the following differences: */
/*     F and G are unchanged on return. */
/*     If G=0, then CS=1 and SN=0. */
/*     If F=0 and (G .ne. 0), then CS=0 and SN=1 without doing any */
/*        floating point operations. */

/*  Arguments */
/*  ========= */

/*  F       (input) COMPLEX */
/*          The first component of vector to be rotated. */

/*  G       (input) COMPLEX */
/*          The second component of vector to be rotated. */

/*  CS      (output) REAL */
/*          The cosine of the rotation. */

/*  SN      (output) COMPLEX */
/*          The sine of the rotation. */

/*  R       (output) COMPLEX */
/*          The nonzero component of the rotated vector. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function definitions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     [ 25 or 38 ops for main paths ] */

    if (g->r == 0.f && g->i == 0.f) {
	*cs = 1.f;
	sn->r = 0.f, sn->i = 0.f;
	r__->r = f->r, r__->i = f->i;
    } else if (f->r == 0.f && f->i == 0.f) {
	*cs = 0.f;

	r_cnjg(&q__2, g);
	r__1 = c_abs(g);
	q__1.r = q__2.r / r__1, q__1.i = q__2.i / r__1;
	sn->r = q__1.r, sn->i = q__1.i;
	r__1 = c_abs(g);
	r__->r = r__1, r__->i = 0.f;

/*         SN = ONE */
/*         R = G */

    } else {
	f1 = (r__1 = f->r, dabs(r__1)) + (r__2 = r_imag(f), dabs(r__2));
	g1 = (r__1 = g->r, dabs(r__1)) + (r__2 = r_imag(g), dabs(r__2));
	if (f1 >= g1) {
	    q__1.r = g->r / f1, q__1.i = g->i / f1;
	    gs.r = q__1.r, gs.i = q__1.i;
/* Computing 2nd power */
	    r__1 = gs.r;
/* Computing 2nd power */
	    r__2 = r_imag(&gs);
	    g2 = r__1 * r__1 + r__2 * r__2;
	    q__1.r = f->r / f1, q__1.i = f->i / f1;
	    fs.r = q__1.r, fs.i = q__1.i;
/* Computing 2nd power */
	    r__1 = fs.r;
/* Computing 2nd power */
	    r__2 = r_imag(&fs);
	    f2 = r__1 * r__1 + r__2 * r__2;
	    d__ = sqrt(g2 / f2 + 1.f);
	    *cs = 1.f / d__;
	    r_cnjg(&q__3, &gs);
	    q__2.r = q__3.r * fs.r - q__3.i * fs.i, q__2.i = q__3.r * fs.i + 
		    q__3.i * fs.r;
	    r__1 = *cs / f2;
	    q__1.r = r__1 * q__2.r, q__1.i = r__1 * q__2.i;
	    sn->r = q__1.r, sn->i = q__1.i;
	    q__1.r = d__ * f->r, q__1.i = d__ * f->i;
	    r__->r = q__1.r, r__->i = q__1.i;
	} else {
	    q__1.r = f->r / g1, q__1.i = f->i / g1;
	    fs.r = q__1.r, fs.i = q__1.i;
/* Computing 2nd power */
	    r__1 = fs.r;
/* Computing 2nd power */
	    r__2 = r_imag(&fs);
	    f2 = r__1 * r__1 + r__2 * r__2;
	    fa = sqrt(f2);
	    q__1.r = g->r / g1, q__1.i = g->i / g1;
	    gs.r = q__1.r, gs.i = q__1.i;
/* Computing 2nd power */
	    r__1 = gs.r;
/* Computing 2nd power */
	    r__2 = r_imag(&gs);
	    g2 = r__1 * r__1 + r__2 * r__2;
	    ga = sqrt(g2);
	    d__ = sqrt(f2 / g2 + 1.f);
	    di = 1.f / d__;
	    *cs = fa / ga * di;
	    r_cnjg(&q__3, &gs);
	    q__2.r = q__3.r * fs.r - q__3.i * fs.i, q__2.i = q__3.r * fs.i + 
		    q__3.i * fs.r;
	    r__1 = fa * ga;
	    q__1.r = q__2.r / r__1, q__1.i = q__2.i / r__1;
	    ss.r = q__1.r, ss.i = q__1.i;
	    q__1.r = di * ss.r, q__1.i = di * ss.i;
	    sn->r = q__1.r, sn->i = q__1.i;
	    q__2.r = g->r * ss.r - g->i * ss.i, q__2.i = g->r * ss.i + g->i * 
		    ss.r;
	    q__1.r = d__ * q__2.r, q__1.i = d__ * q__2.i;
	    r__->r = q__1.r, r__->i = q__1.i;
	}
    }
    return 0;

/*     End of CLARTG */

} /* clartg_ */

