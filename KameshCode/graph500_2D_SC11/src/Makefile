top_srcdir = ../
include $(top_srcdir)/Makefile.inc

COMMON_HDRS = graph500.h
COMMON_LDFLAGS = $(CLDFLAGS)

#=======================================================================

TARGETS = graph500_bfs$(EXEEXT)
CLEANFILES = $(TARGETS)

all: $(TARGETS)

#=======================================================================

graph500_gen_OBJS = generator/graph_generator.o generator/make_graph.o \
                    generator/splittable_mrg.o generator/utils.o

graph500_bfs_HDRS = $(COMMON_HDRS)
graph500_bfs_SRCS = graph_gen.c bfs.c bfs_threaded.c utils.c find_bfs_start_vertices.c \
    validate.c
graph500_bfs_OBJS = $(graph500_bfs_SRCS:.c=.o) create_graph.o RngStream.o $(cas_lock_OBJS)

graph500_bfs_DEPS = $(graph500_bfs_SRCS:.c=.d)
graph500_bfs_LDFLAGS = $(COMMON_LDFLAGS)
-include $(graph500_bfs_DEPS)

RngStream.o: RngStream.c RngStream.h
	$(CC) $(CFLAGSRNG) -c RngStream.c

graph500_bfs$(EXEEXT): main.c $(graph500_bfs_OBJS)
	$(MPICC) $(CFLAGS_STD) $(CFLAGS) $(CFLAGSOMP) $(COPTFLAGS) \
	  -o $@ main.c $(graph500_bfs_OBJS) $(graph500_gen_OBJS) \
	  $(graph500_bfs_LDFLAGS) $(IPM)

CLEANFILES += main.o $(graph500_bfs_OBJS)
CLEANFILES += $(graph500_bfs_DEPS)

#=======================================================================
