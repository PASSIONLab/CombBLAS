/* zlartg.f -- translated by f2c (version 20050501).
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

/* Subroutine */ int zlartg_(doublecomplex *f, doublecomplex *g, doublereal *
	cs, doublecomplex *sn, doublecomplex *r__)
{
    /* System generated locals */
    doublereal d__1, d__2;
    doublecomplex z__1, z__2, z__3;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);
    double z_abs(doublecomplex *), d_imag(doublecomplex *), sqrt(doublereal);

    /* Local variables */
    static doublereal d__, f1, f2, g1, g2, fa, ga, di;
    static doublecomplex fs, gs, ss;


/*  -- LAPACK auxiliary routine (version 2.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd., */
/*     Courant Institute, Argonne National Lab, and Rice University */
/*     September 30, 1994 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLARTG generates a plane rotation so that */

/*     [  CS  SN  ]     [ F ]     [ R ] */
/*     [  __      ]  .  [   ]  =  [   ]   where CS**2 + |SN|**2 = 1. */
/*     [ -SN  CS  ]     [ G ]     [ 0 ] */

/*  This is a faster version of the BLAS1 routine ZROTG, except for */
/*  the following differences: */
/*     F and G are unchanged on return. */
/*     If G=0, then CS=1 and SN=0. */
/*     If F=0 and (G .ne. 0), then CS=0 and SN=1 without doing any */
/*        floating point operations. */

/*  Arguments */
/*  ========= */

/*  F       (input) COMPLEX*16 */
/*          The first component of vector to be rotated. */

/*  G       (input) COMPLEX*16 */
/*          The second component of vector to be rotated. */

/*  CS      (output) DOUBLE PRECISION */
/*          The cosine of the rotation. */

/*  SN      (output) COMPLEX*16 */
/*          The sine of the rotation. */

/*  R       (output) COMPLEX*16 */
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

    if (g->r == 0. && g->i == 0.) {
	*cs = 1.;
	sn->r = 0., sn->i = 0.;
	r__->r = f->r, r__->i = f->i;
    } else if (f->r == 0. && f->i == 0.) {
	*cs = 0.;

	d_cnjg(&z__2, g);
	d__1 = z_abs(g);
	z__1.r = z__2.r / d__1, z__1.i = z__2.i / d__1;
	sn->r = z__1.r, sn->i = z__1.i;
	d__1 = z_abs(g);
	r__->r = d__1, r__->i = 0.;

/*         SN = ONE */
/*         R = G */

    } else {
	f1 = (d__1 = f->r, abs(d__1)) + (d__2 = d_imag(f), abs(d__2));
	g1 = (d__1 = g->r, abs(d__1)) + (d__2 = d_imag(g), abs(d__2));
	if (f1 >= g1) {
	    z__1.r = g->r / f1, z__1.i = g->i / f1;
	    gs.r = z__1.r, gs.i = z__1.i;
/* Computing 2nd power */
	    d__1 = gs.r;
/* Computing 2nd power */
	    d__2 = d_imag(&gs);
	    g2 = d__1 * d__1 + d__2 * d__2;
	    z__1.r = f->r / f1, z__1.i = f->i / f1;
	    fs.r = z__1.r, fs.i = z__1.i;
/* Computing 2nd power */
	    d__1 = fs.r;
/* Computing 2nd power */
	    d__2 = d_imag(&fs);
	    f2 = d__1 * d__1 + d__2 * d__2;
	    d__ = sqrt(g2 / f2 + 1.);
	    *cs = 1. / d__;
	    d_cnjg(&z__3, &gs);
	    z__2.r = z__3.r * fs.r - z__3.i * fs.i, z__2.i = z__3.r * fs.i + 
		    z__3.i * fs.r;
	    d__1 = *cs / f2;
	    z__1.r = d__1 * z__2.r, z__1.i = d__1 * z__2.i;
	    sn->r = z__1.r, sn->i = z__1.i;
	    z__1.r = d__ * f->r, z__1.i = d__ * f->i;
	    r__->r = z__1.r, r__->i = z__1.i;
	} else {
	    z__1.r = f->r / g1, z__1.i = f->i / g1;
	    fs.r = z__1.r, fs.i = z__1.i;
/* Computing 2nd power */
	    d__1 = fs.r;
/* Computing 2nd power */
	    d__2 = d_imag(&fs);
	    f2 = d__1 * d__1 + d__2 * d__2;
	    fa = sqrt(f2);
	    z__1.r = g->r / g1, z__1.i = g->i / g1;
	    gs.r = z__1.r, gs.i = z__1.i;
/* Computing 2nd power */
	    d__1 = gs.r;
/* Computing 2nd power */
	    d__2 = d_imag(&gs);
	    g2 = d__1 * d__1 + d__2 * d__2;
	    ga = sqrt(g2);
	    d__ = sqrt(f2 / g2 + 1.);
	    di = 1. / d__;
	    *cs = fa / ga * di;
	    d_cnjg(&z__3, &gs);
	    z__2.r = z__3.r * fs.r - z__3.i * fs.i, z__2.i = z__3.r * fs.i + 
		    z__3.i * fs.r;
	    d__1 = fa * ga;
	    z__1.r = z__2.r / d__1, z__1.i = z__2.i / d__1;
	    ss.r = z__1.r, ss.i = z__1.i;
	    z__1.r = di * ss.r, z__1.i = di * ss.i;
	    sn->r = z__1.r, sn->i = z__1.i;
	    z__2.r = g->r * ss.r - g->i * ss.i, z__2.i = g->r * ss.i + g->i * 
		    ss.r;
	    z__1.r = d__ * z__2.r, z__1.i = d__ * z__2.i;
	    r__->r = z__1.r, r__->i = z__1.i;
	}
    }
    return 0;

/*     End of ZLARTG */

} /* zlartg_ */

