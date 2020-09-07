#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#if USE_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "graph.h"

#if PRED_CACHE_BYPASS
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

int run_bfs_2Dgraph(dist_graph_t* g, uint64_t root, uint64_t* pred, 
        uint64_t *pred_array_size_ptr, uint64_t* nvisited) {
    
    const uint32_t* restrict adj;
    const uint32_t* restrict num_edges;
    uint64_t n_local, n_local_col, n_local_row;
    uint8_t* d;

    //int32_t* restrict sendbuf_displs;
    int32_t* restrict recvbuf_displs;
    uint32_t *adj_sendbuf, *adj_recvbuf;
    int32_t *adj_sendbuf_counts, *adj_sendbuf_displs;
    int32_t *adj_recvbuf_counts, *adj_recvbuf_displs;
    
    int irow, jcol, rank_rep;
    int nproc_rows, nproc_cols;

    int32_t *sendbuf_counts, *recvbuf_counts;
    
    uint32_t *queue_current, *queue_next;
    
    uint64_t current_vert_count, next_vert_count, global_queue_vert_count;

    uint64_t pred_vert_pair_count;
    uint64_t i, j;
    uint64_t bitmask63;
    uint64_t vis_level;
    uint8_t shift_tab[2];
    uint8_t mask_tab[2];

#if TIME_BFS_SUBROUTINES
    double elt, full_elt, init_elt, mpi_elt, 
           sift_elt, sift_elt_part, load_elt, trans_elt, trans_elt_part, load_elt_part, mpi_elt_part;
    elt = init_elt = mpi_elt = load_elt = sift_elt = 
        full_elt = load_elt_part = sift_elt_part = trans_elt = trans_elt_part = mpi_elt_part = 0.0;
#endif

    shift_tab[0] = 16;
    shift_tab[1] = 1;
#if __x86_64__
    bitmask63 = (1UL<<63)-1; 
#else
    bitmask63 = (1ULL<<63)-1;
#endif

    mask_tab[0] = 0xF0;
    mask_tab[1] = 0x0F;

#if TIME_BFS_SUBROUTINES
    if (rank == 0) {
        full_elt = elt = get_seconds();
    }
#endif

    n_local = g->n_local;
    n_local_col = g->n_local_col;
    n_local_row = g->n_local_row;
    nproc_rows = g->nproc_rows;
    nproc_cols = g->nproc_cols;
    d = g->d;

    sendbuf_counts = (g->comm_data).sendbuf_counts;
    recvbuf_counts = (g->comm_data).recvbuf_counts;
    //sendbuf_displs = (g->comm_data).sendbuf_displs;
    recvbuf_displs = (g->comm_data).recvbuf_displs;
    adj_sendbuf    = (g->comm_data).adj_sendbuf;
    adj_recvbuf    = (g->comm_data).adj_recvbuf;
    adj_sendbuf_counts = (g->comm_data).adj_sendbuf_counts;
    adj_sendbuf_displs = (g->comm_data).adj_sendbuf_displs;
    adj_recvbuf_counts = (g->comm_data).adj_recvbuf_counts;
    adj_recvbuf_displs = (g->comm_data).adj_recvbuf_displs;
    
    adj = g->adj;
    num_edges = g->num_edges;
    queue_current = g->queue_current;
    queue_next = g->queue_next;
    pred_vert_pair_count = 0;
    irow = (g->comm_data).irow; jcol = (g->comm_data).jcol; 
    rank_rep = (rank % (nproc_rows * nproc_cols));
    *nvisited = 0;
    int num_a2a_iterations = 0;

    /*
    if (rank == 0)
        fprintf(stderr, "source vertex: %ld\n", root);

    for (i=0; i<nprocs; i++) {
        sendbuf_counts[i] = 0;
        recvbuf_counts[i] = 0;
    }
    */

#if REPLICATE_D
    for (i=0; i<n_local_col/2 + 1; i++) {
        d[i] = 0;
    }
    //memset(d, 0, (n_local_col/2 + 1));
#else
    memset(d, 0, (n_local/2 + 1));
#endif

    uint32_t src_owner_proc_row = root/n_local_row;
    
    if (src_owner_proc_row == irow) {

        queue_current[0] = root;

        current_vert_count = 1;

    } else {

        current_vert_count = 0;
    
    }

    uint32_t src_owner_proc = root/n_local;
    if (src_owner_proc == rank_rep) {
        pred[pred_vert_pair_count]   = root % n_local_col;
        pred[pred_vert_pair_count+1] = bitmask63;
        pred_vert_pair_count = 2;
    }

#if REPLICATE_D
    uint32_t src_owner_proc_col = root/n_local_col;
    if (src_owner_proc_col == jcol) {
        uint64_t local_src_val = root % n_local_col;
        d[local_src_val>>1] = shift_tab[local_src_val & 1U];
    }
#else
    if (src_owner_proc == rank_rep) {
        uint64_t local_src_val = root % n_local;
        d[local_src_val>>1] = shift_tab[local_src_val & 1U];
    }
#endif

    next_vert_count = 0;
    global_queue_vert_count = 1;
    vis_level = 2;

#if TIME_BFS_SUBROUTINES
    if (rank == 0) {
        elt = get_seconds() - elt;
        init_elt = elt;
    }
#endif

    while (global_queue_vert_count > 0) {

       
        for (i=0; i<nproc_rows; i++) {
            adj_sendbuf_counts[i] = 0;
            adj_recvbuf_counts[i] = 0;
        }

#if TIME_BFS_SUBROUTINES
        if (rank == 0) {
            load_elt_part = get_seconds();
        }
#endif
 
        for (i=0; i<current_vert_count; i++) {

            uint64_t u = (queue_current[i] % n_local_row);
            uint64_t num_edges_u_start = num_edges[u];
            uint64_t num_edges_u_end   = num_edges[u+1];
                
            for (j=num_edges_u_start; j<num_edges_u_end; j++) {

                uint64_t v = adj[j];

                uint64_t v_owner_proc_row = v/n_local;

#if SAMEPROC_NOREAD
#if USE_MPI
                if (v_owner_proc_row == irow) {
#endif
                    uint64_t u_global = n_local_row*irow + u;
#if REPLICATE_D
                    uint32_t dv_pos  = v>>1;
                    //assert(dv_pos < (n_local_col/2 + 1));
                    
                    uint32_t dv_even_odd  = v & 1U;
                    uint32_t d_val = (d[dv_pos] & mask_tab[dv_even_odd]);

                    if (d_val == 0) {
                        d[dv_pos] += vis_level * shift_tab[dv_even_odd];
                        queue_next[next_vert_count++] = v+jcol*n_local_col;
#if PRED_CACHE_BYPASS
                        __m128i pv = 
                            _mm_set_epi64(*(__m64 *) &u_global, 
                                *(__m64 *) &v);
                        _mm_stream_si128((__m128i *) 
                            &pred[pred_vert_pair_count], pv);
#else
                        pred[pred_vert_pair_count] = v;
                        pred[pred_vert_pair_count+1] = u_global;
#endif
 
                        pred_vert_pair_count += 2;
                    }

#else

                    uint64_t v_local = v % n_local;

                    uint32_t dv_pos  = v_local>>1;
                    uint32_t dv_even_odd  = v_local & 1U;
                    uint32_t d_val = (d[dv_pos] & mask_tab[dv_even_odd]);

                    if (d_val == 0) {
                        d[dv_pos] += vis_level * shift_tab[dv_even_odd];
                        queue_next[next_vert_count++] = v+jcol*n_local_col;
#if PRED_CACHE_BYPASS
                        __m128i pv = 
                            _mm_set_epi64(*(__m64 *) &u_global, 
                                *(__m64 *) &v);
                        _mm_stream_si128((__m128i *) 
                            &pred[pred_vert_pair_count], pv);
#else
                        pred[pred_vert_pair_count] = v;
                        pred[pred_vert_pair_count+1] = u_global;
#endif
 
                        pred_vert_pair_count += 2;
                    }
#endif



#endif 

#if USE_MPI
#if SAMEPROC_NOREAD
                } else {
#endif

#if REPLICATE_D
                    uint32_t dv_pos  = v>>1;
                    /*
                    if (dv_pos >= (n_local_col/2 + 1)) {
                        fprintf(stderr, "v dv_pos n_local_col/2+1 %d %d %d\n",
                                v, dv_pos, n_local_col/2+1);
                        assert(dv_pos < (n_local_col/2 + 1));
                    }
                    */
                    uint32_t dv_even_odd  = v & 1U;
                    uint32_t d_val = (d[dv_pos] & mask_tab[dv_even_odd]);

                    if (d_val == 0) {
                        d[dv_pos] += vis_level * shift_tab[dv_even_odd];

                        int32_t pos = adj_sendbuf_displs[v_owner_proc_row] + 
                            adj_sendbuf_counts[v_owner_proc_row];

                        adj_sendbuf_counts[v_owner_proc_row] += 2;
                        adj_sendbuf[pos] = v;
                        adj_sendbuf[pos+1] = u;
                    }
#else

                    int32_t pos = adj_sendbuf_displs[v_owner_proc_row] + 
                       adj_sendbuf_counts[v_owner_proc_row];

                    adj_sendbuf_counts[v_owner_proc_row] += 2;
                    adj_sendbuf[pos] = v;
                    adj_sendbuf[pos+1] = u;
#endif           
                }
#endif
#if SAMEPROC_NOREAD
            }
#endif

        }
       
#if TIME_BFS_SUBROUTINES
        MPI_Barrier((g->comm_data).col_comm);
        if (rank == 0) {
            load_elt += get_seconds() - load_elt_part;
            mpi_elt_part = get_seconds();
        }
#endif

#if USE_MPI
        MPI_Alltoall(adj_sendbuf_counts, 1, MPI_UNSIGNED, 
                adj_recvbuf_counts, 1, 
                MPI_UNSIGNED, (g->comm_data).col_comm);
        MPI_Alltoallv(adj_sendbuf, adj_sendbuf_counts, 
                adj_sendbuf_displs, MPI_UNSIGNED, 
                adj_recvbuf, adj_recvbuf_counts, adj_recvbuf_displs, 
                MPI_UNSIGNED, (g->comm_data).col_comm);
#endif

#if TIME_BFS_SUBROUTINES
#if 0
        uint64_t datavol_sendsize = 0;
        uint64_t datavol_recvsize = 0;
        for (i=0; i<nproc_rows; i++) {
            datavol_sendsize += adj_sendbuf_counts[i];
            datavol_recvsize += adj_recvbuf_counts[i];
        }
        MPI_Barrier((g->comm_data).col_comm);
#endif
        MPI_Barrier((g->comm_data).col_comm);
        if (rank == 0) {
            mpi_elt_part = get_seconds() - mpi_elt_part;
            /*
            if (mpi_elt_part > 0.1) {
                fprintf(stderr, "level %lu, datavol %lu %lu, time %9.6lf s\n", 
                        vis_level, datavol_sendsize, datavol_recvsize, mpi_elt_part);        
            }
            */
            mpi_elt += mpi_elt_part;
            sift_elt_part = get_seconds();
        }
#endif

#if USE_MPI
        for (i=0; i<nproc_rows; i++) {
        
            uint64_t recvbuf_start = adj_recvbuf_displs[i];
            uint64_t recvbuf_end = recvbuf_start + adj_recvbuf_counts[i];
            
            for (j=recvbuf_start; j<recvbuf_end; j+= 2) {

                uint64_t v        = adj_recvbuf[j];
                uint64_t u_pred   = adj_recvbuf[j+1]; 
                uint64_t u_global = i*n_local_row+u_pred; 

#if REPLICATE_D
                uint32_t dv_pos  = v>>1;
                // assert(dv_pos < (n_local_col/2 + 1));
                
                uint32_t dv_even_odd  = (v & 1U);
                uint32_t d_val = (d[dv_pos] & mask_tab[dv_even_odd]);

                if (d_val == 0) {
                    d[dv_pos] += vis_level * shift_tab[dv_even_odd];
                    queue_next[next_vert_count++] = v + jcol*n_local_col;
 
#if PRED_CACHE_BYPASS
                    __m128i pv = 
                        _mm_set_epi64(*(__m64 *) &u_global, 
                            *(__m64 *) &v);
                    _mm_stream_si128((__m128i *) 
                        &pred[pred_vert_pair_count], pv);
#else
                    pred[pred_vert_pair_count] = v;
                    pred[pred_vert_pair_count+1] = u_global;
#endif
 
                    pred_vert_pair_count += 2;
                }
#else

                uint64_t v_local = v % n_local;
                uint32_t dv_pos  = v_local>>1;
                uint32_t dv_even_odd  = (v_local & 1U);
                uint32_t d_val = (d[dv_pos] & mask_tab[dv_even_odd]);

                if (d_val == 0) {
                    d[dv_pos] += vis_level * shift_tab[dv_even_odd];
                    queue_next[next_vert_count++] = v+jcol*n_local_col;
 
#if PRED_CACHE_BYPASS
                    __m128i pv = 
                        _mm_set_epi64(*(__m64 *) &u_global, 
                            *(__m64 *) &v);
                    _mm_stream_si128((__m128i *) 
                        &pred[pred_vert_pair_count], pv);
#else
                    pred[pred_vert_pair_count] = v;
                    pred[pred_vert_pair_count+1] = u_global;
#endif
 
                    pred_vert_pair_count += 2;
                }


#endif
            }
        }
#endif


#if TIME_BFS_SUBROUTINES
        MPI_Barrier((g->comm_data).col_comm);
        if (rank == 0) {
            sift_elt += get_seconds() - sift_elt_part;
            trans_elt_part = get_seconds();
        }
#endif

        /* Transpose queue_next */
        uint64_t next_vert_count_trans = 0; 
#if USE_MPI
        MPI_Status status1, status2;
#endif

        int recv_proc =  ((irow*nproc_cols+jcol)/nproc_rows) + 
            nproc_cols * ((irow*nproc_cols+jcol)%nproc_rows);
        assert(recv_proc < nproc_rows * nproc_cols);
#if USE_MPI
        MPI_Sendrecv(&next_vert_count, 1, MPI_UNSIGNED_LONG,
                jcol*nproc_rows+irow, rank % (nproc_cols*nproc_rows), 
                &next_vert_count_trans, 1, MPI_UNSIGNED_LONG, 
                recv_proc, recv_proc, (g->comm_data).replicas_comm, 
                &status1);

        MPI_Sendrecv(queue_next, next_vert_count, MPI_UNSIGNED,
                jcol*nproc_rows+irow, rank % (nproc_cols*nproc_rows),
                queue_current, next_vert_count_trans, MPI_UNSIGNED, 
                recv_proc, recv_proc, (g->comm_data).replicas_comm, 
                &status2);
#else
        next_vert_count_trans = next_vert_count;
        memcpy(queue_current, queue_next,
                sizeof(uint32_t)*next_vert_count_trans);
#endif
        /* Allgather */
        sendbuf_counts[0] = next_vert_count_trans;
#if USE_MPI
        MPI_Allgather(sendbuf_counts, 1, MPI_INT, recvbuf_counts, 1, 
                MPI_INT, (g->comm_data).row_comm);
#else
        recvbuf_counts[0] = sendbuf_counts[0];
#endif
        recvbuf_displs[0] = 0; 
        for (i=1; i<nproc_cols; i++) {
            recvbuf_displs[i] = recvbuf_displs[i-1] +
                recvbuf_counts[i-1];
        }
        current_vert_count = recvbuf_displs[nproc_cols-1] + 
                                recvbuf_counts[nproc_cols-1];

        assert(current_vert_count < n_local_row);

#if USE_MPI
        MPI_Allgatherv(queue_current, next_vert_count_trans,
                MPI_UNSIGNED, queue_next, recvbuf_counts,
                recvbuf_displs, MPI_UNSIGNED, (g->comm_data).row_comm);
#else
        memcpy(queue_next, queue_current,
                next_vert_count_trans*sizeof(uint32_t));
#endif
        
        uint32_t *tmp_queue_ptr = queue_current;
        queue_current = queue_next;
        queue_next = tmp_queue_ptr;
        

#if TIME_BFS_SUBROUTINES
        MPI_Barrier((g->comm_data).row_comm);
        if (rank == 0) {
            trans_elt += get_seconds() - trans_elt_part;
        }
#endif

        *nvisited += current_vert_count;
        //fprintf(stderr, "rank %d, current_vert_count %lu\n", 
        //       rank, current_vert_count);
        //uint64_t tmp_vert_count = current_vert_count;
        //current_vert_count = next_vert_count;
        next_vert_count = 0;
        
        //MPI_Barrier(MPI_COMM_WORLD);

#if USE_MPI
        MPI_Allreduce(&current_vert_count, &global_queue_vert_count, 1, 
                MPI_UNSIGNED_LONG, MPI_SUM, (g->comm_data).col_comm);
#else
        global_queue_vert_count = current_vert_count;
#endif
        vis_level++;
        num_a2a_iterations++;

        /*
        if (rank == 0)
            fprintf(stderr, "vis level %lu, global vert count %lu\n", vis_level, 
                global_queue_vert_count);
        */
        /*
        for (i=0; i<nprocs; i++) {
            sendbuf_counts[i] = 0;
            recvbuf_counts[i] = 0;
        }
        */
        assert(vis_level < 15);

    }

    *pred_array_size_ptr = pred_vert_pair_count/2;
#if USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, nvisited, 1, MPI_UNSIGNED_LONG,
            MPI_SUM, (g->comm_data).col_comm);
    MPI_Barrier((g->comm_data).replicas_comm);
#endif

#if TIME_BFS_SUBROUTINES
    full_elt = get_seconds() - full_elt;
    if (rank == 0)
        fprintf(stderr, "time: %9.6lf s.\n"
                "sift %9.6lf s. (%3.3lf), load %9.6lf s. (%3.3lf)\n"
                "a2a  %9.6lf s. (%3.3lf), gather %9.6lf s. (%3.3lf)\n"
                "comm %9.6lf s. (%3.3lf), comp %9.6lf s. (%3.3lf)\n"
                "init %9.6lf s., vis levels %lu, nvisited %lu\n",
                full_elt, 
                sift_elt, (sift_elt/full_elt)*100.0,
                load_elt, (load_elt/full_elt)*100.0,
                mpi_elt,  (mpi_elt/full_elt)*100.0,
                trans_elt, (trans_elt/full_elt)*100.0,
                mpi_elt+trans_elt, ((mpi_elt+trans_elt)/full_elt)*100.0,
                sift_elt+load_elt, ((sift_elt+load_elt)/full_elt)*100.0,
                init_elt, vis_level, *nvisited);
#endif


    return 0;
}

/*
int run_bfs_threaded(dist_graph_t* g, uint64_t root, uint64_t* pred, 
        uint64_t *pred_array_size_ptr, uint64_t* nvisited) {

    return 0;
}
*/
