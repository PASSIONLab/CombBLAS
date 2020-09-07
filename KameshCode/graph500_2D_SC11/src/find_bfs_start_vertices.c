#include <stdlib.h>
#include <math.h>
#if USE_MPI
#include <mpi.h>
#endif
#include "graph.h"

int find_bfs_start_vertices(int num_bfs_roots, dist_graph_t* g, uint64_t* bfs_roots) {

    uint64_t num_roots_selected;

    /* let rank 0 populate the array with non-zero vertices */
    if (rank == 0) {

        srand(ROOT_VERT_RNG_SEED);
        num_roots_selected = 0;
        while (num_roots_selected != num_bfs_roots) {
            
            uint64_t u, i;
            u = ((double) rand()/RAND_MAX) * g->n_local;
            if ((g->num_edges[u+1] - g->num_edges[u]) >= 1) {

                /* check that this vertex hasn't been previously selected */
                int uniq_root = 1;
                for (i = 0; i<num_roots_selected; i++) {
                    if (bfs_roots[i] == u) {
                       uniq_root = 0;
                       break;
                    }
                }
                if (uniq_root == 1) {
                    bfs_roots[num_roots_selected++] = u;
                }
            }
        }
    }
    
#if USE_MPI
    MPI_Bcast(bfs_roots, num_bfs_roots, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    return 0;
}
