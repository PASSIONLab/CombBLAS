#include <vector>
#include <utility>
#include <parallel/algorithm>

int main()
{

   int sequences[10][10];
   for (int i = 0; i < 10; ++i)
     for (int j = 0; i < 10; ++j)
       sequences[i][j] = j;
    int out[33];
    std::vector<std::pair<int*, int*> > seqs;
    for (int i = 0; i < 10; ++i)
      { seqs.push_back(std::make_pair(sequences[i], sequences[i] + 10)); }
 
    __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), out, std::less<int>(), 33);
	return 0;
}


