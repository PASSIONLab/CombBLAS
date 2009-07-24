#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <sys/time.h> // for gettimeofday
#include "promote.h"
#include "Semirings.h"
#include "Deleter.h"
#include <ext/numeric>

using namespace std;


int main()
{
	vector<int> tvec;
	tvec.reserve(10);
	
	tvec[5] = 5;
	cout << tvec.size() << " " << tvec.capacity() << endl;
	return 0;
}
