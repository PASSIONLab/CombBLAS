top_srcdir = ../..
include $(top_srcdir)/Makefile.inc

# Set these variables appropriately
CFLAGS = -v -std=c99 -O3 -m64 -DGRAPH_GENERATOR_MPI -DGRAPHGEN_DISTRIBUTED_MEMORY \
 -DNDEBUG -I$(MPIDIR)/include64
COPTFLAGS =
LDFLAGS = -O3

GENERATOR_SOURCES = graph_generator.c make_graph.c splittable_mrg.c utils.c
GENERATOR_OBJS = $(GENERATOR_SOURCES:.c=.o)
GENERATOR_HEADERS = graph_generator.h make_graph.h mod_arith_32bit.h mod_arith_64bit.h mod_arith.h mod_arith_xmt.h splittable_mrg.h utils.h user_settings.h mrg_transitions.c

all: $(GENERATOR_OBJS)

clean:
	-rm -f $(GENERATOR_OBJS)
