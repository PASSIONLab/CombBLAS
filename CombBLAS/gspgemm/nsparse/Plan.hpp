#include <nsparse.hpp>

template <class idType>
class Plan
{
public:
    Plan(): isPlan(false), seg_size(1), block_size(1), memory_access(INT_MAX), min_msec(sfFLT_MAX)
    {
    }
    Plan(idType segment, idType block): isPlan(true)
    {
        seg_size = segment;
        if (seg_size > USHORT_MAX) {
            seg_size = USHORT_MAX;
        }
        block_size = block;
        if (block_size < 1 || block_size > MAX_BLOCK_SIZE) {
            block_size = 1;
        }
    }
    ~Plan()
    {
    }
    void set_plan(idType s_size, idType b_size)
    {
        seg_size = s_size;
        block_size = b_size;
        isPlan = true;
    }

    idType thread_grid;
    idType thread_block;
    bool isPlan;
    idType SIGMA;
    idType seg_size;
    idType seg_num;
    idType block_size;
    idType memory_access;
    float min_msec;
};

