/* 
 * Graph500 BFS
 * Uses the generator and test harness provided in the Graph 500 reference implementation v1.2
 *
 * Kamesh Madduri, Penn State University
 * last updated: December 2011
 */

/* Copyright (C) 2010 The Trustees of Indiana University.                  */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/stat.h>
#if USE_MPI
#include <mpi.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include "generator/make_graph.h"
#include "graph.h"
#ifdef __cplusplus
}
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

int rank, nprocs;

enum {s_minimum, s_firstquartile, s_median, s_thirdquartile, s_maximum, s_mean, s_std, s_LAST};

static int compare_doubles(const void* a, const void* b) {
    double aa = *(const double*)a;
    double bb = *(const double*)b;
    return (aa < bb) ? -1 : (aa == bb) ? 0 : 1;
}

void get_statistics(const double x[], int n, double r[s_LAST]) {

    double temp;
    int i;

    /* Compute mean. */
    temp = 0;
    for (i = 0; i < n; ++i) 
        temp += x[i];
    temp /= n;
    r[s_mean] = temp;
    
    /* Compute std. dev. */
    temp = 0;
    for (i = 0; i < n; ++i) 
        temp += (x[i] - r[s_mean]) * (x[i] - r[s_mean]);
    temp /= n - 1;
    r[s_std] = sqrt(temp);
  
    /* Sort x. */
    double* xx = (double*)malloc(n * sizeof(double));
    memcpy(xx, x, n * sizeof(double));
    qsort(xx, n, sizeof(double), compare_doubles);
  
    /* Get order statistics. */
    r[s_minimum] = xx[0];
    r[s_firstquartile] = (xx[(n - 1) / 4] + xx[n / 4]) * .5;
    r[s_median] = (xx[(n - 1) / 2] + xx[n / 2]) * .5;
    r[s_thirdquartile] = (xx[n - 1 - (n - 1) / 4] + xx[n - 1 - n / 4]) * .5;
    r[s_maximum] = xx[n - 1];
  
    /* Clean up. */
    free(xx);
}


int main(int argc, char** argv) {

    int SCALE;
    double edgefactor;
    //uint64_t nedges;
    //uint32_t *edges;
    char *input_filename;
    int num_procrows, num_proccols, num_replicas;
    graph_gen_data_t ggi;
    //graph_gen_aux_data_t ggaux;
    int create_2D_graph = 0;
    int read_graph_from_file = 0;

#if USE_MPI
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#else
    rank = 0;
    nprocs = 1;
#endif

    /* Parse arguments. */
    SCALE = 16;
    edgefactor = 16.0;
    if (argc >= 2) SCALE = atoi(argv[1]);
    if (argc >= 3) edgefactor = atof(argv[2]);
    create_2D_graph = 0;
    if (argc <= 7) { 
        num_replicas = atoi(argv[3]);
        num_procrows = atoi(argv[4]);
        num_proccols = atoi(argv[5]);
        if ((num_replicas * num_proccols * num_procrows) != nprocs) {
            if (rank > 0) {
                fprintf(stderr, "Invalid input!\n");
#if USE_MPI
                MPI_Abort(MPI_COMM_WORLD, 1);
#else
                exit(1);
#endif
            }
        }
        create_2D_graph = 1;
    }
    /* Read graph from file */
    if (argc == 7) { 
        read_graph_from_file = 1;
        input_filename = argv[6];
    }


    if (argc <= 1 || argc >= 8) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s SCALE edgefactor\n  SCALE = log_2(# vertices) [integer, required]\n  edgefactor = (# edges) / (# vertices) = .5 * (average vertex degree) [float, defaults to 16]\n(Random number seed and Kronecker initiator are in main.c)\n", argv[0]);
        }

#if USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(1);
#endif
    }

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    /* Make the raw graph edges. */

#pragma omp parallel
{
#ifdef _OPENMP
    int nthreads = omp_get_num_threads();
#else
    int nthreads = 1;
#endif
#pragma omp single
    if (rank == 0) {
        fprintf(stderr, "SCALE: %d, %d MPI tasks, %d OMP threads\n"
                "nreplicas %d, nproc_rows %d, nproc_cols %d\n",
             SCALE, nprocs, nthreads, num_replicas, 
             num_procrows, num_proccols);
#if REPLICATE_D
        fprintf(stderr, "Vertex status array replicated\n");
#endif
    }

}

    int64_t nedges;
    packed_edge* gen_edges;
    double make_graph_time;

    if (read_graph_from_file == 0) {
    
        double make_graph_start = get_seconds();
        ggi.SCALE = SCALE;
        ggi.n = 1UL<<SCALE;
        ggi.m = edgefactor*ggi.n;
    
        // gen_graph_edges(&ggi, &ggaux);
        make_graph(SCALE, edgefactor*(1L<<SCALE), 1, 2, &nedges, &gen_edges);
     
        double make_graph_stop = get_seconds();
        make_graph_time = make_graph_stop - make_graph_start;
    
    } else {

        double make_graph_start = get_seconds();
        //ggi.SCALE = SCALE;
        //ggi.n = 1UL<<SCALE;
        //ggi.m = edgefactor*ggi.n;
        // read_graph(input_filename, &nedges, &gen_edges); 
        FILE *infp = fopen(input_filename, "rb");
        assert(infp != NULL);

        struct stat st;
        stat(input_filename, &st);
        long nedges_global = st.st_size/8;

        ggi.m = nedges_global;
        uint64_t read_offset_start = rank * 8* (nedges_global/nprocs);
        uint64_t read_offset_end   = (rank+1) * 8 *
            (nedges_global/nprocs);

        if (rank == nprocs - 1)
           read_offset_end = 8*nedges_global;

        nedges = (read_offset_end - read_offset_start)/8;

        /* gen_edges is an array of unsigned ints of size 2*m_local */
        fseek(infp, read_offset_start, SEEK_SET);

        uint32_t *gen_edges_fp = (uint32_t *)
            malloc(2*nedges*sizeof(uint32_t));
        assert(gen_edges_fp != NULL);

        if (rank == 0) {
            fprintf(stderr, "nedges is %ld\n", nedges);
        }

        fread(gen_edges_fp, 2*nedges, sizeof(uint32_t), infp);
        fclose(infp);

        gen_edges = (packed_edge *) malloc(nedges*sizeof(packed_edge));
        assert(gen_edges != NULL);
        uint32_t *gedges = (uint32_t *) gen_edges;
        for (long i=0; i<nedges; i++) {
            gedges[3*i] = gen_edges_fp[2*i];
            gedges[3*i+1] = gen_edges_fp[2*i+1];
            gedges[3*i+2] = 0;
        }
        free(gen_edges_fp);

        // gen_graph_edges(&ggi, &ggaux);
        // make_graph(SCALE, edgefactor*(1L<<SCALE), 1, 2, &nedges, &gen_edges);
     
        double make_graph_stop = get_seconds();
        make_graph_time = make_graph_stop - make_graph_start;

    }


#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

        
    /* the distributed graph data structure */
    dist_graph_t g;
    //nedges = ggi.m_local;
    //edges  = ggi.gen_edges;

    /* Time graph creation */
    double data_struct_start = get_seconds();

    uint32_t *edges = (uint32_t *) gen_edges;

    if (create_2D_graph) {
            
        g.nproc_rows = num_procrows;
        g.nproc_cols = num_proccols;
        g.nreplicas  = num_replicas;

        create_2Ddist_graph(nedges, edges, &g);

    } else {

        create_dist_graph(nedges, edges, &g);

    }

    double data_struct_stop = get_seconds();
    double data_struct_time = data_struct_stop - data_struct_start;
    
    /* Get roots for BFS runs. */
    int num_bfs_roots = 32;
    uint64_t* bfs_roots = (uint64_t *) malloc(num_bfs_roots * sizeof(uint64_t));
    find_bfs_start_vertices(num_bfs_roots, &g, bfs_roots);

    /* Number of edges visited in each BFS; a double so get_statistics can be
     * used directly. */
    double* edge_counts = (double *) malloc(num_bfs_roots * sizeof(double));

    /* Run BFS. */
    int validation_passed = 1;
    double* bfs_times = (double *) malloc(num_bfs_roots * sizeof(double));
    double* validate_times = (double *) malloc(num_bfs_roots * sizeof(double));

    uint64_t* pred = (uint64_t *) _mm_malloc((3 * g.n_local) * sizeof(uint64_t), 16);
    assert(pred != NULL);

    uint8_t* d_trans = (uint8_t *) malloc(g.n_local * sizeof(uint8_t));
    assert(d_trans != NULL);

    uint8_t* d_trans_full = (uint8_t *) malloc(g.n_local_row * sizeof(uint8_t));
    assert(d_trans_full != NULL);


    int bfs_root_idx;
#if 0
    for (bfs_root_idx = 0; bfs_root_idx < num_bfs_roots; ++bfs_root_idx) {

        uint64_t root = bfs_roots[bfs_root_idx];

        if (rank == 0) 
            fprintf(stderr, "Running BFS %d (%lu)\n", bfs_root_idx, root);

        /* Clear the pred array. */
        memset(pred, 0, 3 * g.n_local * sizeof(uint64_t));
        assert(pred != NULL);

        /* Do the actual BFS. */
        double bfs_start = get_seconds();
        
        uint64_t nvisited = 0;
        uint64_t pred_array_size = 0;
        if (create_2D_graph)
            run_bfs_2Dgraph(&g, root, pred, &pred_array_size,
                    &nvisited);
        double bfs_stop = get_seconds();
        bfs_times[bfs_root_idx] = bfs_stop - bfs_start;

       
        /* Calculate number of input edges visited. */
        uint64_t edge_visit_count = 0;
#if REPLICATE_D
        for (uint64_t i = 0; i < g.n_local_row; i++) {

            for (uint64_t j=g.num_edges[i]; j<g.num_edges[i+1]; j++) {

                uint64_t v = g.adj[j];
                uint32_t dv = g.d[v/2];

                if ((v & 1U) == 0) { 
                    dv = (dv & 0xF0)>>4;
                } else {
                    dv = (dv & 0x0F);
                }
                if (dv != 0)
                    edge_visit_count++;

            }
        }

        int irow = (g.comm_data).irow;
        int jcol = (g.comm_data).jcol;
        int nproc_rows = g.nproc_rows;
        int nproc_cols = g.nproc_cols;

        int recv_proc =  ((irow*nproc_cols+jcol)/nproc_rows) + 
            nproc_cols * ((irow*nproc_cols+jcol)%nproc_rows);
        MPI_Status status1;
        uint8_t *d_send_offset = g.d + (irow*g.n_local)/2;
        if (rank == 0) {
            if (g.n_local % 2 != 0) 
                fprintf(stderr, "Warning! Visited edge count will be incorrect\n");
        }

        MPI_Sendrecv(d_send_offset, g.n_local, MPI_UNSIGNED_CHAR, 
            jcol*nproc_rows+irow, rank % (nproc_cols * nproc_rows), 
            d_trans, g.n_local, MPI_UNSIGNED_CHAR,
            recv_proc, recv_proc,
            (g.comm_data).replicas_comm, &status1);
        MPI_Allgather(d_trans, g.n_local, MPI_UNSIGNED_CHAR, 
                d_trans_full, g.n_local, MPI_UNSIGNED_CHAR,
                (g.comm_data).row_comm);
        g.d_trans = d_trans;
        g.d_trans_full = d_trans_full;

#else
        int irow = (g.comm_data).irow;
        int jcol = (g.comm_data).jcol;
        int nproc_rows = g.nproc_rows;
        int nproc_cols = g.nproc_cols;

        int recv_proc =  ((irow*nproc_cols+jcol)/nproc_rows) + 
            nproc_cols * ((irow*nproc_cols+jcol)%nproc_rows);
        MPI_Status status1;
         
        MPI_Sendrecv(g.d, g.n_local, MPI_UNSIGNED_CHAR, 
            jcol*nproc_rows+irow, rank % (nproc_cols * nproc_rows), 
            d_trans, g.n_local, MPI_UNSIGNED_CHAR,
            recv_proc, recv_proc,
            (g.comm_data).replicas_comm, &status1);
        MPI_Allgather(d_trans, g.n_local, MPI_UNSIGNED_CHAR, 
                d_trans_full, g.n_local, MPI_UNSIGNED_CHAR,
                (g.comm_data).row_comm);
        g.d_trans = d_trans;
        g.d_trans_full = d_trans_full;
        
        for (uint64_t i = 0; i < g.n_local_row; i++) {

            uint64_t u_off = i % g.n_local;
            
            uint32_t du = d_trans_full[(i/g.n_local)*g.n_local + u_off/2];
            if ((u_off & 1U) == 0) { 
                du = (du & 0xF0)>>4;
            } else {
                du = (du & 0x0F);
            }
            if (du != 0) {
                edge_visit_count += (g.num_edges[i+1] - g.num_edges[i]);
            }
        }
#endif
        
#if USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &edge_visit_count, 1, MPI_UNSIGNED_LONG, 
                        MPI_SUM, MPI_COMM_WORLD);
#endif
        edge_counts[bfs_root_idx] = (double) edge_visit_count/2;

         
        if (rank==0)
            fprintf(stderr, "edge visit count: %ld\n", edge_visit_count/2);

        
        double validate_start = get_seconds();
        int validation_passed_one 
            = validate_bfs_result(&g, root, pred, pred_array_size);
        
        double validate_stop = get_seconds();
        validate_times[bfs_root_idx] = validate_stop - validate_start;
        
        if (!validation_passed_one) {
            validation_passed = 0;
            if (rank == 0) fprintf(stderr, "Validation failed for this BFS root; skipping rest.\n");
            break;
        }
        MPI_Barrier(MPI_COMM_WORLD); 
        
    }



#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    /* Print results. */
    if (rank == 0) {
        fflush(stderr);
        fprintf(stderr, "\n\n");
        if (!validation_passed) {
            fprintf(stderr, "No results printed for invalid run.\n");
        } else {
            int i;
            fprintf(stderr, "SCALE:                          %d\n", SCALE);
            fprintf(stderr, "edgefactor:                     %.2g\n", edgefactor);
            fprintf(stderr, "NBFS:                           %d\n", num_bfs_roots);
            fprintf(stderr, "num_mpi_processes:              %d\n", nprocs);
            fprintf(stderr, 
                            "graph_generation:               %g s\n", make_graph_time);
            fprintf(stderr, "construction_time:              %g s\n", data_struct_time);
            double stats[s_LAST];
            get_statistics(bfs_times, num_bfs_roots, stats);
            
            fprintf(stderr, "min_time:                       %g s\n", stats[s_minimum]);
            fprintf(stderr, "firstquartile_time:             %g s\n", stats[s_firstquartile]);
            fprintf(stderr, "median_time:                    %g s\n", stats[s_median]);
            fprintf(stderr, "thirdquartile_time:             %g s\n", stats[s_thirdquartile]);
            fprintf(stderr, "max_time:                       %g s\n", stats[s_maximum]);
            fprintf(stderr, "mean_time:                      %g s\n", stats[s_mean]);
            fprintf(stderr, "stddev_time:                    %g\n", stats[s_std]);
            get_statistics(edge_counts, num_bfs_roots, stats);
            fprintf(stderr, "min_nedge:                      %.11g\n", stats[s_minimum]);
            fprintf(stderr, "firstquartile_nedge:            %.11g\n", stats[s_firstquartile]);
            fprintf(stderr, "median_nedge:                   %.11g\n", stats[s_median]);
            fprintf(stderr, "thirdquartile_nedge:            %.11g\n", stats[s_thirdquartile]);
            fprintf(stderr, "max_nedge:                      %.11g\n", stats[s_maximum]);
            fprintf(stderr, "mean_nedge:                     %.11g\n", stats[s_mean]);
            fprintf(stderr, "stddev_nedge:                   %.11g\n", stats[s_std]);
            double* secs_per_edge = (double*)malloc(num_bfs_roots * sizeof(double));
            for (i = 0; i < num_bfs_roots; ++i) secs_per_edge[i] = bfs_times[i] / edge_counts[i];
            get_statistics(secs_per_edge, num_bfs_roots, stats);
            fprintf(stderr, "min_TEPS:                       %g TEPS\n", 1. / stats[s_maximum]);
            fprintf(stderr, "firstquartile_TEPS:             %g TEPS\n", 1. / stats[s_thirdquartile]);
            fprintf(stderr, "median_TEPS:                    %g TEPS\n", 1. / stats[s_median]);
            fprintf(stderr, "thirdquartile_TEPS:             %g TEPS\n", 1. / stats[s_firstquartile]);
            fprintf(stderr, "max_TEPS:                       %g TEPS\n", 1. / stats[s_minimum]);
            fprintf(stderr, "harmonic_mean_TEPS:             %g TEPS\n", 1. / stats[s_mean]);
            /* Formula from:
             * Title: The Standard Errors of the Geometric and Harmonic Means and
             *        Their Application to Index Numbers
             * Author(s): Nilan Norris
             * Source: The Annals of Mathematical Statistics, Vol. 11, No. 4 (Dec., 1940), pp. 445-448
             * Publisher(s): Institute of Mathematical Statistics
             * Stable URL: http://www.jstor.org/stable/2235723
             * (same source as in specification). */
            fprintf(stderr, "harmonic_stddev_TEPS:           %g\n", 
                    stats[s_std] / (stats[s_mean] * stats[s_mean] * sqrt(num_bfs_roots - 1)));
            free(secs_per_edge); secs_per_edge = NULL;
            get_statistics(validate_times, num_bfs_roots, stats);
            fprintf(stderr, "min_validate:                   %g s\n", stats[s_minimum]);
            fprintf(stderr, "firstquartile_validate:         %g s\n", stats[s_firstquartile]);
            fprintf(stderr, "median_validate:                %g s\n", stats[s_median]);
            fprintf(stderr, "thirdquartile_validate:         %g s\n", stats[s_thirdquartile]);
            fprintf(stderr, "max_validate:                   %g s\n", stats[s_maximum]);
            fprintf(stderr, "mean_validate:                  %g s\n", stats[s_mean]);
            fprintf(stderr, "stddev_validate:                %g\n",
                    stats[s_std]);
#if 0
            for (i = 0; i < num_bfs_roots; ++i) {
                fprintf(stderr, "Run %3d:                        %g s, validation %g s\n", i + 1, bfs_times[i], validate_times[i]);
      }
#endif
        }
        fflush(stderr);
    }
#endif
#if 1
    for (bfs_root_idx = 0; bfs_root_idx < num_bfs_roots; ++bfs_root_idx) {

        uint64_t root = bfs_roots[bfs_root_idx];

        if (rank == 0) 
            fprintf(stderr, "Running threaded BFS %d (%lu)\n", bfs_root_idx, root);

        /* Clear the pred array. */
        memset(pred, 0, 3 * g.n_local * sizeof(uint64_t));
        assert(pred != NULL);

        /* Do the actual BFS. */
        double bfs_start = get_seconds();
        
        uint64_t nvisited = 0;
        uint64_t pred_array_size = 0;
       
        if (create_2D_graph) 
            run_bfs_2Dgraph_threaded(&g, root, pred, &pred_array_size, &nvisited);
        double bfs_stop = get_seconds();
        bfs_times[bfs_root_idx] = bfs_stop - bfs_start;

        /* Calculate number of input edges visited. */
        uint64_t edge_visit_count = 0;
#if REPLICATE_D
        for (uint64_t i = 0; i < g.n_local_row; i++) {

            for (uint64_t j=g.num_edges[i]; j<g.num_edges[i+1]; j++) {

                uint64_t v = g.adj[j];
                uint32_t dv = g.d[v];

                if (dv != 0)
                    edge_visit_count++;
            }
        }

        int irow = (g.comm_data).irow;
        int jcol = (g.comm_data).jcol;
        int nproc_rows = g.nproc_rows;
        int nproc_cols = g.nproc_cols;

        int recv_proc =  ((irow*nproc_cols+jcol)/nproc_rows) + 
            nproc_cols * ((irow*nproc_cols+jcol)%nproc_rows);
        MPI_Status status1;
        uint8_t *d_send_offset = g.d + (irow*g.n_local);

        MPI_Sendrecv(d_send_offset, g.n_local, MPI_UNSIGNED_CHAR, 
            jcol*nproc_rows+irow, rank % (nproc_cols * nproc_rows), 
            d_trans, g.n_local, MPI_UNSIGNED_CHAR,
            recv_proc, recv_proc,
            (g.comm_data).replicas_comm, &status1);
        MPI_Allgather(d_trans, g.n_local, MPI_UNSIGNED_CHAR, 
                d_trans_full, g.n_local, MPI_UNSIGNED_CHAR,
                (g.comm_data).row_comm);
        g.d_trans = d_trans;
        g.d_trans_full = d_trans_full;

#else
        int irow = (g.comm_data).irow;
        int jcol = (g.comm_data).jcol;
        int nproc_rows = g.nproc_rows;
        int nproc_cols = g.nproc_cols;

        int recv_proc =  ((irow*nproc_cols+jcol)/nproc_rows) + 
            nproc_cols * ((irow*nproc_cols+jcol)%nproc_rows);
        MPI_Status status1;
         
        MPI_Sendrecv(g.d, g.n_local, MPI_UNSIGNED_CHAR, 
            jcol*nproc_rows+irow, rank % (nproc_cols * nproc_rows), 
            d_trans, g.n_local, MPI_UNSIGNED_CHAR,
            recv_proc, recv_proc,
            (g.comm_data).replicas_comm, &status1);
        MPI_Allgather(d_trans, g.n_local, MPI_UNSIGNED_CHAR, 
                d_trans_full, g.n_local, MPI_UNSIGNED_CHAR,
                (g.comm_data).row_comm);
        g.d_trans = d_trans;
        g.d_trans_full = d_trans_full;
 
        for (uint64_t i = 0; i < g.n_local_row; i++) {

            uint64_t u_off = i % g.n_local;
            
            uint32_t du = d_trans_full[(i/g.n_local)*g.n_local + u_off];
            if (du != 0) {
                edge_visit_count += (g.num_edges[i+1] - g.num_edges[i]);
            }
        }
#endif
        
#if USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &edge_visit_count, 1, MPI_UNSIGNED_LONG, 
                        MPI_SUM, MPI_COMM_WORLD);
#endif
        edge_counts[bfs_root_idx] = (double) edge_visit_count/2;

         
        double validate_start = get_seconds();
        int validation_passed_one 
            = validate_bfs_result_threaded(&g, root, pred, pred_array_size);
        
        double validate_stop = get_seconds();
        validate_times[bfs_root_idx] = validate_stop - validate_start;
        
        if (!validation_passed_one) {
            validation_passed = 0;
            if (rank == 0) fprintf(stderr, "Validation failed for this BFS root; skipping rest.\n");
            break;
        }
        MPI_Barrier(MPI_COMM_WORLD); 
 

    }


    /* Print results. */
    if (rank == 0) {
        fflush(stderr);
        fprintf(stderr, "\n\n");
        if (!validation_passed) {
            fprintf(stderr, "No results printed for invalid run.\n");
        } else {
            int i;
            fprintf(stderr, "SCALE:                          %d\n", SCALE);
            fprintf(stderr, "edgefactor:                     %.2g\n", edgefactor);
            fprintf(stderr, "NBFS:                           %d\n", num_bfs_roots);
            fprintf(stderr, "num_mpi_processes:              %d\n", nprocs);
            fprintf(stderr, 
                            "graph_generation:               %g s\n", make_graph_time);
            fprintf(stderr, "construction_time:              %g s\n", data_struct_time);
            double stats[s_LAST];
            get_statistics(bfs_times, num_bfs_roots, stats);
            fprintf(stderr, "min_time:                       %g s\n", stats[s_minimum]);
            fprintf(stderr, "firstquartile_time:             %g s\n", stats[s_firstquartile]);
            fprintf(stderr, "median_time:                    %g s\n", stats[s_median]);
            fprintf(stderr, "thirdquartile_time:             %g s\n", stats[s_thirdquartile]);
            fprintf(stderr, "max_time:                       %g s\n", stats[s_maximum]);
            fprintf(stderr, "mean_time:                      %g s\n", stats[s_mean]);
            fprintf(stderr, "stddev_time:                    %g\n", stats[s_std]);
            get_statistics(edge_counts, num_bfs_roots, stats);
            fprintf(stderr, "min_nedge:                      %.11g\n", stats[s_minimum]);
            fprintf(stderr, "firstquartile_nedge:            %.11g\n", stats[s_firstquartile]);
            fprintf(stderr, "median_nedge:                   %.11g\n", stats[s_median]);
            fprintf(stderr, "thirdquartile_nedge:            %.11g\n", stats[s_thirdquartile]);
            fprintf(stderr, "max_nedge:                      %.11g\n", stats[s_maximum]);
            fprintf(stderr, "mean_nedge:                     %.11g\n", stats[s_mean]);
            fprintf(stderr, "stddev_nedge:                   %.11g\n", stats[s_std]);
            double* secs_per_edge = (double*)malloc(num_bfs_roots * sizeof(double));
            for (i = 0; i < num_bfs_roots; ++i) secs_per_edge[i] = bfs_times[i] / edge_counts[i];
                get_statistics(secs_per_edge, num_bfs_roots, stats);
            fprintf(stderr, "min_TEPS:                       %g TEPS\n", 1. / stats[s_maximum]);
            fprintf(stderr, "firstquartile_TEPS:             %g TEPS\n", 1. / stats[s_thirdquartile]);
            fprintf(stderr, "median_TEPS:                    %g TEPS\n", 1. / stats[s_median]);
            fprintf(stderr, "thirdquartile_TEPS:             %g TEPS\n", 1. / stats[s_firstquartile]);
            fprintf(stderr, "max_TEPS:                       %g TEPS\n", 1. / stats[s_minimum]);
            fprintf(stderr, "harmonic_mean_TEPS:             %g TEPS\n", 1. / stats[s_mean]);
            /* Formula from:
             * Title: The Standard Errors of the Geometric and Harmonic Means and
             *        Their Application to Index Numbers
             * Author(s): Nilan Norris
             * Source: The Annals of Mathematical Statistics, Vol. 11, No. 4 (Dec., 1940), pp. 445-448
             * Publisher(s): Institute of Mathematical Statistics
             * Stable URL: http://www.jstor.org/stable/2235723
             * (same source as in specification). */
            
            fprintf(stderr, "harmonic_stddev_TEPS:           %g\n", 
                    stats[s_std] / (stats[s_mean] * stats[s_mean] * sqrt(num_bfs_roots - 1)));
            free(secs_per_edge); secs_per_edge = NULL;
            get_statistics(validate_times, num_bfs_roots, stats);
            fprintf(stderr, "min_validate:                   %g s\n", stats[s_minimum]);
            fprintf(stderr, "firstquartile_validate:         %g s\n", stats[s_firstquartile]);
            fprintf(stderr, "median_validate:                %g s\n", stats[s_median]);
            fprintf(stderr, "thirdquartile_validate:         %g s\n", stats[s_thirdquartile]);
            fprintf(stderr, "max_validate:                   %g s\n", stats[s_maximum]);
            fprintf(stderr, "mean_validate:                  %g s\n", stats[s_mean]);
            fprintf(stderr, "stddev_validate:                %g\n", stats[s_std]);
#if 0
            for (i = 0; i < num_bfs_roots; ++i) {
                fprintf(stderr, "Run %3d:                        %g s, validation %g s\n", i + 1, bfs_times[i], validate_times[i]);
      }
#endif
        }
        fflush(stderr);
    }

#endif


    free(edge_counts); edge_counts = NULL;

    _mm_free(pred);
    free(bfs_roots);
    free_graph(&g);
    free(d_trans);
    free(d_trans_full);

    free(bfs_times);
    free(validate_times);

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif

    return 0;

}

