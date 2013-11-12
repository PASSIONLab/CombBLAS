/* Copyright (C) 2010-2013 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#ifndef COROUTINE_HPP
#define COROUTINE_HPP

#include <stdlib.h>
#include "common.hpp"

struct coroutine {
  virtual ~coroutine() {}
  virtual coroutine* step() = 0;
};

struct cor_par: coroutine {
  coroutine* a;
  coroutine* b;
  cor_par(coroutine* a, coroutine* b): a(a), b(b) {}
  coroutine* step() {
    coroutine* a = this->a;
    coroutine* b = this->b;
    if (a) {
      this->a = a = a->step();
    }
    if (b) {
      this->b = b = b->step();
    }
    if (!a) {memfree("cor_par", sizeof(*this)); delete this; return b;}
    if (!b) {memfree("cor_par", sizeof(*this)); delete this; return a;}
    return this;
  }
};

struct cor_seq: coroutine {
  coroutine* a;
  coroutine* b;
  cor_seq(coroutine* a, coroutine* b): a(a), b(b) {}
  coroutine* step() {
    coroutine* a = this->a;
    coroutine* b = this->b;
    if (a) {
      this->a = a = a->step();
    }
    if (!a) {memfree("cor_seq", sizeof(*this)); delete this; return b;}
    return this;
  }
};

inline coroutine* skip() {return NULL;}
inline coroutine* par(coroutine* a, coroutine* b) {memalloc("cor_par", sizeof(cor_par)); return new cor_par(a, b);}
inline coroutine* seq(coroutine* a, coroutine* b) {memalloc("cor_seq", sizeof(cor_seq)); return new cor_seq(a, b);}
inline void run(coroutine* c) {while (c) c = c->step();}

#define COROUTINE_STATE_DECL int coroutine_current_state;
#define COROUTINE_STATE_INIT coroutine_current_state(0)
#define COROUTINE_PROC_START coroutine* step() {switch (this->coroutine_current_state) {default: abort(); case 0: ;
#define COROUTINE_PROC_END }}
#define COROUTINE_YIELD(id) do {this->coroutine_current_state = (id); return this; case (id): ;} while (0)
#define COROUTINE_EXIT do {return NULL;} while (0)

#define COROUTINE_MPI_WAIT(id, req_ptr, status_maybe) \
  do { \
    while (1) { \
      int flag; \
      WITH_MPI_LOCK MPI_Test((req_ptr), &flag, (status_maybe)); \
      if (flag) break; else COROUTINE_YIELD(id); \
    } \
  } while (0)

#endif // COROUTINE_HPP
