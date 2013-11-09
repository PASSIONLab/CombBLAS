#ifndef BITMAP_H
#define BITMAP_H

#include <algorithm>
#include <stdint.h>

#define WORD_OFFSET(n) (n/64)
#define BIT_OFFSET(n) (n & 0x3f)


class BitMap {
 public:
  // actually always rounds up
  BitMap(uint64_t size) {
    uint64_t num_longs = (size + 63) / 64;
    start = new uint64_t[num_longs];
    end = start + num_longs;
  }

  ~BitMap() {
    delete[] start;
  }

  inline
  void reset() {
    for(uint64_t *it=start; it!=end; it++)
      *it = 0;
  }

  inline
  void set_bit(uint64_t pos) {
    start[WORD_OFFSET(pos)] |= ((uint64_t) 1l<<BIT_OFFSET(pos));
  }

  inline
  void set_bit_atomic(long pos) {
    set_bit(pos);
    // uint64_t old_val, new_val;
    // uint64_t *loc = start + WORD_OFFSET(pos);
    // do {
    //   old_val = *loc;
    //   new_val = old_val | ((uint64_t) 1l<<BIT_OFFSET(pos));
    // } while(!__sync_bool_compare_and_swap(loc, old_val, new_val));
  }

  inline
  bool get_bit(uint64_t pos) {
    if (start[WORD_OFFSET(pos)] & (1l<<BIT_OFFSET(pos)))
      return true;
    else
      return false;
  }

  inline
  long get_next_bit(uint64_t pos) {
    uint64_t next = pos;
    int bit_offset = BIT_OFFSET(pos);
    uint64_t *it = start + WORD_OFFSET(pos);
    uint64_t temp = (*it);
    if (bit_offset != 63) {
      temp = temp >> (bit_offset+1);
    } else {
      temp = 0;
    }
    if (!temp) {
      next = (next & 0xffffffc0);
      while (!temp) {
        it++;
        if (it >= end)
          return -1;
        temp = *it;
        next += 64;
      }
    } else {
      next++;
    }
    while(!(temp&1)) {
      temp = temp >> 1;
      next++;
    }
    return next;
  }

  inline
  uint64_t* data() {
    return start;
  }

  void copy_from(const BitMap* other) {
    copy(other->start, other->end, start);
  }

  void print_ones() {
    uint64_t max_size = (end-start)*64;
    for (uint64_t i=0; i<max_size; i++)
      if (get_bit(i))
        cout << " " << i;
    cout << endl;
  }

 private:
  uint64_t *start;
  uint64_t *end;
};

#endif // BITMAP_H
