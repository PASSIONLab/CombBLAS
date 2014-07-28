#include <vector>
#include <utility>
#include <parallel/algorithm>
#include <algorithm>
#include <iostream>
#include <iterator>

using namespace std;

int main()
{

   int sequences[10][10];
   for (int i = 0; i < 10; ++i)
     for (int j = 0; j < 10; ++j)
       sequences[i][j] = i*j;
    int out[100];
    std::vector<std::pair<int*, int*> > seqs;
    for (int i = 0; i < 10; ++i)
      { seqs.push_back(std::make_pair(sequences[i], sequences[i] + 10)); }
 
    int * final = __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), out, 100, std::less<int>());
	copy(out, final, ostream_iterator<int>(cout, " "));
	return 0;
}


