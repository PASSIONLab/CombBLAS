#ifndef _STACK_ENTRY_H
#define _STACK_ENTRY_H

#include <utility>

namespace combblas {

template <class T1, class T2>
class StackEntry
{
public:
	T1 value;
	T2 key;
};

}

#endif
