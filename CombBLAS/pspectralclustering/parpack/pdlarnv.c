/* pdlarnv.f -- translated by f2c (version 20050501).
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

/* Common Block Declarations */

struct {
    integer mpi_bottom__, mpi_integer__, mpi_real__, mpi_double_precision__, 
	    mpi_complex__, mpi_double_complex__, mpi_logical__, 
	    mpi_character__, mpi_byte__, mpi_2integer__, mpi_2real__, 
	    mpi_2double_precision__, mpi_2complex__, mpi_2double_complex__, 
	    mpi_integer1__, mpi_integer2__, mpi_integer4__, mpi_real2__, 
	    mpi_real4__, mpi_real8__, mpi_ub__, mpi_lb__, mpi_comm_world__, 
	    mpi_comm_self__, mpi_group_empty__, mpi_sum__, mpi_max__, 
	    mpi_min__, mpi_prod__, mpi_land__, mpi_band__, mpi_lor__, 
	    mpi_bor__, mpi_lxor__, mpi_bxor__, mpi_minloc__, mpi_maxloc__, 
	    mpi_op_null__, mpi_tag_ub__, mpi_host__, mpi_io__, 
	    mpi_errors_are_fatal__, mpi_errors_return__, mpi_packed__;
} mpipriv_;

#define mpipriv_1 mpipriv_

/* \BeginDoc */

/* \Name: pdlarnv */

/* Message Passing Layer: MPI */

/* \Description: */

/*  Parallel Version of ARPACK utility routine dlarnv */

/*  PSLARNV returns a vector of n (nloc) random real numbers from a uniform or */
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

/*  X       (output) Double precision array, dimension (N) */
/*          The generated random numbers. */

/* \Author: Kristi Maschhoff */

/* \Details */

/*  Simple parallel version of LAPACK auxiliary routine dlarnv */
/*  for X distributed across a 1-D array of processors. */
/*  This routine calls the auxiliary routine SLARNV to generate random */
/*  real numbers from a uniform (0,1) distribution. Output is consistent */
/*  with serial version. */

/* \SCCS Information: */
/* FILE: larnv.F   SID: 1.4   DATE OF SID: 04/16/99 */

/* ----------------------------------------------------------------------- */

/* Subroutine */ int pdlarnv_(integer *comm, integer *idist, integer *iseed, 
	integer *n, doublereal *x)
{
    extern /* Subroutine */ int dlarnv_(integer *, integer *, integer *, 
	    doublereal *);



/*     .. MPI VARIABLES AND FUNCTIONS .. */
/* /+ */
/* * */
/* *  (C) 1993 by Argonne National Laboratory and Mississipi State University. */
/* *      All rights reserved.  See COPYRIGHT in top-level directory. */
/* +/ */

/* /+ user include file for MPI programs, with no dependencies +/ */

/* /+ return codes +/ */







/*     We handle datatypes by putting the variables that hold them into */
/*     common.  This way, a Fortran program can directly use the various */
/*     datatypes and can even give them to C programs. */

/*     MPI_BOTTOM needs to be a known address; here we put it at the */
/*     beginning of the common block.  The point-to-point and collective */
/*     routines know about MPI_BOTTOM, but MPI_TYPE_STRUCT as yet does not. */

/*     The types MPI_INTEGER1,2,4 and MPI_REAL4,8 are OPTIONAL. */
/*     Their values are zero if they are not available.  Note that */
/*     using these reduces the portability of code (though may enhance */
/*     portability between Crays and other systems) */



/*     All other MPI routines are subroutines */

/*     The attribute copy/delete functions are symbols that can be passed */
/*     to MPI routines */
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
    dlarnv_(idist, &iseed[1], n, &x[1]);

    return 0;
} /* pdlarnv_ */

