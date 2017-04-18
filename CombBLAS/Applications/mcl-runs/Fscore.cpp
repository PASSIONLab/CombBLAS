#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string.h>
#include <assert.h>
#include <omp.h>
using namespace std;

template <typename T>
vector<size_t> sortIndices(const vector<T> &v) {
    
    // initialize original index locations
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
    
    return idx;
}

int64_t intersect(vector<int64_t> vec1, vector<int64_t> vec2)
{
    int64_t isect = 0;
    for(int i=0, j=0; i<vec1.size() && j<vec2.size();)
    {
        if(vec1[i] < vec2[j]) i++;
        else if(vec2[j] < vec1[i]) j++;
        else
        {
            isect ++;
            i++; j++;
        }
    }
    return isect;
}

// fname1 is the reference (form MCL)
double Fscore(string fname1, string fname2)
{
    ifstream infile1 (fname1);
    ifstream infile2 (fname2);
    string line;
    int64_t item, clustID = 0;
    int64_t nclust1, nclust2;
    int64_t N1 = 0, N2 = 0; // total items
    
    vector<vector<int64_t>> clusters1;
    vector<vector<int64_t>> clusters2;
    
    if (infile1.is_open())
    {
        while ( getline (infile1,line) )
        {
            istringstream iss(line);
            if(clustID >= clusters1.size())
            {
                clusters1.resize(clustID * 2 + 1);
            }
            while ( iss >> item)
            {
                clusters1[clustID].push_back(item);
            }
            
            N1 += clusters1[clustID].size();
            clustID++;
        }
        infile1.close();
        nclust1 = clustID;
    }
    else
    {
        cout << "Unable to open " << fname1 << endl;
        return -1;
    }
    
    clustID = 0;
    if (infile2.is_open())
    {
        while ( getline (infile2,line) )
        {
            istringstream iss(line);
            if(clustID >= clusters2.size())
            {
                clusters2.resize(clustID * 2 + 1);
            }
            while ( iss >> item)
            {
                clusters2[clustID].push_back(item);
            }
            N2 += clusters2[clustID].size();
            clustID++;
        }
        infile2.close();
        nclust2 = clustID;
    }
    else
    {
        cout << "Unable to open " << fname2 << endl;
        return -1;
    }
    
    assert(N1==N2);
#pragma omp parallel for
    for(int i=0; i<nclust1; i++)
        sort(clusters1[i].begin(), clusters1[i].end());
#pragma omp parallel for
    for(int i=0; i<nclust2; i++)
        sort(clusters2[i].begin(), clusters2[i].end());
        
    vector<double> F(nclust1);
#pragma omp parallel for
    for(int i=0; i<nclust1; i++)
    {
        double Fi = 0;
        for(int j=0; j<nclust2; j++)
        {
            int64_t isect = intersect(clusters1[i], clusters2[j]);
            double precision = (double) isect / clusters2[j].size();
            double recall = (double) isect / clusters1[i].size();
            double Fij = 2 * precision * recall / (precision + recall);
            Fi = max(Fi, Fij);
        }
        Fi = Fi * clusters1[i].size()/ N1;
        F[i] = Fi;
    }
    return accumulate(F.begin(), F.end(), 0.0);
}

int main(int argc, char* argv[])
{
    
    string ifilename1 = "";
    string ifilename2 = "";
  
    
    if(argc < 2)
    {
        cout << "Usage: ./fscore -M1 <FILENAME_Output_ MCL> -M2 <FILENAME_Output_HipMCL>\n";
        cout << "Example: ./fscore -M1 input1.txt -M2 input2.txt " << endl;
        return -1;
    }
    
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i],"-M1")==0)
        {
            ifilename1 = string(argv[i+1]);
            printf("\nfilename1: %s",ifilename1.c_str());
        }
        else if (strcmp(argv[i],"-M2")==0)
        {
            ifilename2 = string(argv[i+1]);
            printf("\nfilename2: %s",ifilename2.c_str());
        }
    }
    printf("\n");
    double F = Fscore(ifilename1, ifilename2);
    cout << "F score: " << F << endl;
    return 0;
}
