/* pzlarnv.f -- translated by f2c (version 20050501).
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

/* \BeginDoc */

/* \Name: pzlarnv */

/* Message Passing Layer: MPI */

/* \Description: */

/*  Parallel Version of ARPACK utility routine zlarnv */

/*  PZLARNV returns a vector of n (nloc) random Complex*16 numbers from a uniform or */
/*  normal distribution. It is assumed that X is distributed across a 1-D array */
/*  of processors ( nprocs < 1000 ) */

/* \Arguments */
/*  COMM    MPI Communicator for the processor grid */

/*  IDIST   (input) INTEGER */
/*          Specifies the distribution of the random numbers: */
/*          = 1:  uniform (0,1) */
/*          = 2:  uniform (-1,1) */
/*          = 3:  normal (0,1) */

/*  ISEED   (input/output) INTEGER array, dimension (4) */
/*          On entry, the seed of the random number generator; the array */
/*          elements must be between 0 and 4095, and ISEED(4) must be */
/*          odd. */
/*          On exit, the seed is updated. */

/*  N       (input) INTEGER */
/*          The number of random numbers to be generated. */

/*  X       (output) Complex*16 array, dimension (N) */
/*          The generated random numbers. */

/* \Author: Kristi Maschhoff */

/* \Details */

/*  Simple parallel version of LAPACK auxiliary routine zlarnv */
/*  for X distributed across a 1-D array of processors. */
/*  This routine calls the auxiliary routine CLARNV to generate random */
/*  Complex*16 numbers from a uniform or normal distribution. Output is consistent */
/*  with serial version. */

/* \SCCS Information: */
/* FILE: larnv.F   SID: 1.3   DATE OF SID: 04/17/99 */

/* ----------------------------------------------------------------------- */

/* Subroutine */ int pzlarnv_(integer *comm, integer *idist, integer *iseed, 
	integer *n, doublecomplex *x)
{
    extern /* Subroutine */ int zlarnv_(integer *, integer *, integer *, 
	    doublecomplex *);


/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --x;
    --iseed;

    /* Function Body */
    zlarnv_(idist, &iseed[1], n, &x[1]);

    return 0;
} /* pzlarnv_ */

