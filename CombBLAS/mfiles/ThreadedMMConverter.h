#ifndef _THREADED_MM_CONVERTER_
#define _THREADED_MM_CONVERTER_

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <omp.h>
#include <sys/stat.h>
#include <string.h>
#include "mmio.h"
using namespace std;

#define BATCH 10000000  // 10MB
#define MAXLINELENGTH 128

template <typename IT1, typename NT1, typename IT2, typename NT2>
void push_to_vectors(vector<IT1> & rows, vector<IT1> & cols, vector<NT1> & vals, IT2 ii, IT2 jj, NT2 vv)
{
    rows.push_back(ii);
    cols.push_back(jj);
    vals.push_back(vv);
}

template <typename IT1, typename NT1>
void ProcessLines(vector<IT1> & rows, vector<IT1> & cols, vector<NT1> & vals, vector<string> & lines, const map<string, uint32_t> & vertexmap, const vector<uint32_t> & shuffler)
{
    char fr[64];
    char to[64];
    double vv;
    for (vector<string>::iterator itr=lines.begin(); itr != lines.end(); ++itr)
    {
        // string::c_str() -> Returns a pointer to an array that contains a null-terminated sequence of characters (i.e., a C-string)
        sscanf(itr->c_str(), "%s %s %lg", fr, to, &vv);
        uint32_t fr_id = shuffler[vertexmap.at(string(fr))];
        uint32_t to_id = shuffler[vertexmap.at(string(to))];
        push_to_vectors(rows, cols, vals, fr_id, to_id, vv);
    }
    vector<string>().swap(lines);
}


void check_newline(int *bytes_read, int bytes_requested, char *buf)
{
    if ((*bytes_read) < bytes_requested) {
        // fewer bytes than expected, this means EOF
        if (buf[(*bytes_read) - 1] != '\n') {
            // doesn't terminate with a newline, add one to prevent infinite loop later
            buf[(*bytes_read) - 1] = '\n';
            cout << "Error in Matrix Market format, appending missing newline at end of file" << endl;
            (*bytes_read)++;
        }
    }
}

// updates to curpos are reflected in the caller function
bool FetchBatch(FILE * f_local, long int & curpos, long int end_fpos, bool firstcall, vector<string> & lines)
{
    size_t bytes2fetch = BATCH;    // we might read more than needed but no problem as we won't process them
    bool begfile = (ftell(f_local) == 0);
    if(firstcall && !begfile)
    {
        curpos -= 1;    // first byte is to check whether we started at the beginning of a line
        bytes2fetch += 1;
    }
    char * buf = new char[bytes2fetch]; // needs to happen **after** bytes2fetch is updated
    char * originalbuf = buf;   // so that we can delete it later because "buf" will move
    
    int seekfail = fseek(f_local, curpos, SEEK_SET); // move the file pointer to the beginning of thread data
    if(seekfail != 0)
        cout << "fseek failed to move to " << curpos << endl;
    
    int bytes_read = fread(buf, sizeof(char), bytes2fetch, f_local);  // read byte by byte
    if(!bytes_read)
    {
        delete [] originalbuf;
        return true;    // done
    }
    check_newline(&bytes_read, bytes2fetch, buf);
    if(firstcall && !begfile)
    {
        if(buf[0] == '\n')  // we got super lucky and hit the line break
        {
            buf += 1;
            bytes_read -= 1;
            curpos += 1;
        }
        else    // skip to the next line and let the preceeding thread take care of this partial line
        {
            cout << "skipping" << endl;
            char *c = (char*)memchr(buf, '\n', MAXLINELENGTH); //  return a pointer to the matching byte or NULL if the character does not occur
            if (c == NULL) {
                cout << "Unexpected line without a break" << endl;
            }
            else
            {
                cout << c << endl;
            }
            int n = c - buf + 1;
            bytes_read -= n;
            buf += n;
            curpos += n;
        }
    }
    while(bytes_read > 0 && curpos < end_fpos)  // this will also finish the last line
    {
        char *c = (char*)memchr(buf, '\n', bytes_read); //  return a pointer to the matching byte or NULL if the character does not occur
        if (c == NULL) {
            delete [] originalbuf;
            return false;  // if bytes_read stops in the middle of a line, that line will be re-read next time since curpos has not been moved forward yet
        }
        int n = c - buf + 1;
        
        // string constructor from char * buffer: copies the first n characters from the array of characters pointed by s
        lines.push_back(string(buf, n-1));  // no need to copy the newline character
        bytes_read -= n;   // reduce remaining bytes
        buf += n;   // move forward the buffer
        curpos += n;
    }
    delete [] originalbuf;
    if (curpos >= end_fpos) return true;  // don't call it again, nothing left to read
    else    return false;
}

template <typename IT, typename NT>
void ThreadedMMConverter(const string & filename, vector<IT> & allrows, vector<IT> & allcols, vector<NT> & allvals, IT& nvertices, ofstream & dictout)
{
    FILE *f;
    if ((f = fopen(filename.c_str(), "r")) == NULL)
    {
        printf("file can not be found\n");
        exit(1);
    }
        
    // Use fseek again to go backwards two bytes and check that byte with fgetc
    struct stat st;     // get file size
    if (stat(filename.c_str(), &st) == -1)
    {
        exit(1);
    }
    int64_t file_size = st.st_size;
    cout << "File is " << file_size << " bytes" << endl;
    long int ffirst = ftell(f); // doesn't change
    long int fpos = ffirst;
    long int end_fpos = file_size;
    
    double time_start = omp_get_wtime();
    map<string, uint32_t> vertexmap;
    vector<string> lines;
    bool finished = FetchBatch(f, fpos, end_fpos, true, lines); // fpos will move
    int64_t entriesread = lines.size();
    
    char from[64];
    char to[64];
    double vv;
    uint32_t vertexid = 0;
    for (vector<string>::iterator itr=lines.begin(); itr != lines.end(); ++itr)
    {
        // string::c_str() -> Returns a pointer to an array that contains a null-terminated sequence of characters (i.e., a C-string)
        sscanf(itr->c_str(), "%s %s %lg", from, to, &vv);
        auto ret = vertexmap.insert(make_pair(string(from), vertexid));
        if (ret.second)	++vertexid; // insert successful
        ret = vertexmap.insert(make_pair(string(to), vertexid));
        if (ret.second)	++vertexid; // insert successful
    }
    vector<string>().swap(lines);
    
    while(!finished)
    {
        finished = FetchBatch(f, fpos, end_fpos, false, lines);
        entriesread += lines.size();
        
        // Process files
        char from[64];
        char to[64];
        double vv;
        for (vector<string>::iterator itr=lines.begin(); itr != lines.end(); ++itr)
        {
            // string::c_str() -> Returns a pointer to an array that contains a null-terminated sequence of characters (i.e., a C-string)
            sscanf(itr->c_str(), "%s %s %lg", from, to, &vv);
            auto ret = vertexmap.insert(make_pair(string(from), vertexid));
            if (ret.second)	++vertexid; // insert successful
            ret = vertexmap.insert(make_pair(string(to), vertexid));
            if (ret.second)	++vertexid; // insert successful
        }
        vector<string>().swap(lines);
    }
    cout << "Populated maps " << omp_get_wtime() - time_start << "  seconds"<< endl;
    cout << "There are " << nvertices << " vertices and " << vertexid << " edges" << endl;

    time_start = omp_get_wtime();
    nvertices = vertexid;
    vector< uint32_t > shuffler(nvertices);
    iota(shuffler.begin(), shuffler.end(), static_cast<uint32_t>(0));
    random_shuffle ( shuffler.begin(), shuffler.end() );
    fclose(f);
    
    for (auto it = vertexmap.begin(); it != vertexmap.end(); ++it)
    {
        dictout << shuffler[it->second] << "\t" << it ->first << "\n";
    }
    cout << "Shuffled and wrote dictionary in " << omp_get_wtime() - time_start << "  seconds"<< endl;

    
    vector<IT> localsizes(omp_get_max_threads());
#pragma omp parallel
    {
        long int fpos, end_fpos;
        int this_thread = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        if(this_thread == 0)
            fpos = ffirst;
        else
            fpos = this_thread * file_size / num_threads;
        
        if(this_thread != (num_threads-1)) end_fpos = (this_thread + 1) * file_size / num_threads;
        else end_fpos = file_size;
        
        FILE * f_perthread = fopen(filename.c_str(), "rb");   // reopen
        
        vector<string> lines;
        bool finished = FetchBatch(f_perthread, fpos, end_fpos, true, lines);
        
        vector<IT> rows;
        vector<IT> cols;
        vector<NT> vals;
        
        ProcessLines(rows, cols, vals, lines, vertexmap, shuffler);
        while(!finished)
        {
            finished = FetchBatch(f_perthread, fpos, end_fpos, false, lines);
            ProcessLines(rows, cols, vals, lines, vertexmap, shuffler);
        }
        localsizes[this_thread] = rows.size();
#pragma omp barrier
        
#pragma omp single
        {
            size_t nnz_after_symmetry = std::accumulate(localsizes.begin(), localsizes.begin()+num_threads, IT(0));
            
            allrows.resize(nnz_after_symmetry);
            allcols.resize(nnz_after_symmetry);
            allvals.resize(nnz_after_symmetry);
            
            copy(localsizes.begin(), localsizes.end(), ostream_iterator<IT>(cout, " ")); cout << endl;
        }
        
        IT untilnow = std::accumulate(localsizes.begin(), localsizes.begin()+this_thread, IT(0));
        
        std::copy(rows.begin(), rows.end(), allrows.begin() + untilnow);
        std::copy(cols.begin(), cols.end(), allcols.begin() + untilnow);
        std::copy(vals.begin(), vals.end(), allvals.begin() + untilnow);
    }
    
}

#endif

