/*
Copyleft (C) Michaelangel007, Oct 2022
https://github.com/Michaelangel007/5words5letters
Optimized Jotto / wordle-inspired solver that is significantly faster then the Python one Matt Parker wrote.

Word List downloaded from:
* https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt

Output should have 538 unique 5-cliques sans anagrams.  The following list DOES includes anagrams such as "sprung" (0x0016A000)
* https://gitlab.com/bpaassen/five_clique/-/raw/main/cliques.csv

See:
* https://en.wikipedia.org/wiki/Clique_(graph_theory)
*/

// Includes
    #define _CRT_SECURE_NO_WARNINGS 1 // MSVC warnings
    #include <stdio.h>    // printf(), fopen(), fread(), fclose()
    #include <stdlib.h>   // atoi(), exit()
    #include <sys/stat.h> // stat()
    #include <string.h>   // memset()
    #include <chrono>     // now()
    #include <omp.h>
#ifdef _MSC_VER 
    #include <intrin.h>                 // https://stackoverflow.com/questions/3849337/msvc-equivalent-to-builtin-popcount
    #define __builtin_popcount __popcnt // same as gcc; also known as Hamming Weight, https://en.wikipedia.org/wiki/Hamming_weight
#endif

#if _WIN32                // MS-DOS / Windows
    #define EOL_CHAR '\r' // 0x0D 0x0A
    #define EOL_SIZE 2    // CR LF
    #define ALIGN64(t) __declspec((align(64))t       /* MSVC alignment is pre type*/
#else                     // Un*x
    #define EOL_CHAR '\n' // 0x0A
    #define EOL_SIZE 1    // LF
    #define ALIGN64(t) t __attribute__((aligned(64)) /* gcc/clang alignment is post type */
#endif

// Globals
          size_t gnBufferSize = 0;
          char   gaBufferText[8388608]; // "all_words.txt" is  4,234,917 bytes

    // Total number of permutations for one word with 5 letters:
    //   26 * 26 * 26 * 26 * 26 = 26^5 = 11,881,376  <  ceil( log2 ) = 24
    // In practice we use a dictionary of valid words which have significantly fewer combinations
    const int    NUM_CHARS     =    5;  // letters per word
    const int    NUM_WORDS     =    5;  // total words
    const int    MAX_5_WORDS   = 8192;  // permutation of all letters in one word; in practice we have 5,977 unique words
    const int    MAX_NEIGHBORS = 4096;  // List of neighbors for this hash; in practice we have 2,347 neighbors.
    const int    MAX_THREADS  =   256;  // Threadripper 3990X

          int    gnUniqueWords = 0;                           // number of words with exactly NUM_CHARS letters
          char  *gaWords    [ MAX_5_WORDS ];                  // pointers to first letter of words that have 5 letters
          int    gaHash     [ MAX_5_WORDS ];
          short  gaNeighbors[ MAX_5_WORDS ][ MAX_NEIGHBORS ]; // DAG of valid neighbors
          short  gaSolutions[ MAX_THREADS ];
          short  gaOutput   [ MAX_THREADS ][ MAX_NEIGHBORS ]; // Each thread outputs 5x words, maximum 538*5 = 2690

// ======================================================================
void Init()
{
    memset( gaSolutions, 0, sizeof( gaSolutions ) );  // Scatter
}

// Read raw word file where words are of varying length
// ======================================================================
void Read4( const char *filename )
{
    FILE *file   = fopen( filename, "rb" );
    if ( !file )
        exit( printf( "ERROR: Couldn't open input file: %s\n", filename ) );

    if (gnBufferSize >= sizeof(gaBufferText) - 2)
        exit( printf( "ERROR: Couldn't allocate memory for file. %d KB > %d KB\n", (int)gnBufferSize/1024, (int)sizeof(gaBufferText)/1024 ) );

    gnBufferSize = fread( gaBufferText, 1, sizeof(gaBufferText)-2, file );
    gaBufferText[ gnBufferSize+0 ] = EOL_CHAR;
    gaBufferText[ gnBufferSize+1 ] = 0;

    fclose( file );
}

// Parses dictionary reading all 5 letter words, assumes in already sorted format
// ======================================================================
void Parse()
{
    char *pText = (char*) gaBufferText;
    char *pEnd  = (char*) gaBufferText + gnBufferSize;

    int nTotalWords  = 0;
    int nLengthWords = 0;
    int nUniqueWords = 0;
    int nDuplicates  = 0;

    while (pText < pEnd)
    {
        char *eow = pText;

        while (*eow != EOL_CHAR)
            eow++;

        size_t len = (eow - pText);
        *eow = 0;

        if (len == NUM_CHARS)
        {
            int   nHash = 0;
            for( int iLetter = 0; iLetter < NUM_CHARS; ++iLetter )
                nHash |= 1 << (pText[iLetter] - 'a');  // convert 7-bit ASCII string to 26-bit bit mask

            nLengthWords++;
            gaWords[ nUniqueWords ] = pText;
            gaHash [ nUniqueWords ] = nHash;

            if (__builtin_popcount(gaHash[nUniqueWords]) == NUM_CHARS) // Only accept words with 5 letters, trivial reject words that have duplicate letters
            {
                // if this hash already exists skip anagrams
                bool found = false;
                for (int word = 0; word < nUniqueWords; ++word)
                {
                    if (gaHash[ nUniqueWords ] == gaHash[ word ])
                    {
                        found = true;
                        nDuplicates++;
                    }
                }
                if (!found)
                    nUniqueWords++;
            }
        }
        nTotalWords++;
        pText = eow + EOL_SIZE;
    }
    gnUniqueWords = nUniqueWords;

    printf( "%6d Total words\n"           , nTotalWords             );
    printf( "%6d length %d words\n"       , nLengthWords, NUM_CHARS );
    printf( "%6d duplicate %d words\n"    , nDuplicates , NUM_CHARS );
    printf( "%6d unique %d letter words\n", nUniqueWords, NUM_CHARS );
}

// ======================================================================
void Prepare()
{
#pragma omp parallel for
    for( short word0 = 0; word0 < gnUniqueWords; ++word0 )
    {
        short nNeighbors = 1; // See note below

        // Instead of starting from 0, if our dictionary of words is sorted we can start testing for candidates from the next word
        for( short word1 = word0+1; word1 < gnUniqueWords; ++word1 )
            if ((gaHash[word0] & gaHash[word1]) == 0)         // two words are unique if the bitwise AND of bitmasks is zero!
                gaNeighbors[ word0 ][ nNeighbors++ ] = (short) word1;

        gaNeighbors[ word0 ][0] = nNeighbors; // [1,n] for loop counters since[0] has list size
    }
}

// ======================================================================
void Search3()
{
#pragma omp parallel for
    for (int word0 = 0; word0 < gnUniqueWords; ++word0) // Every word effectively has a neighbor
    {
        int iThread = omp_get_thread_num();

        int nHash0   = 0 | gaHash[ word0 ];       // "previous" hash is zero
        int nOffset1 = gaNeighbors[ word0 ][ 0 ];

        for (int iOffset1 = 1; iOffset1 < nOffset1; ++iOffset1)
        {
            int word1 = gaNeighbors[ word0 ][ iOffset1 ];
            int hash1 = gaHash[ word1 ] & nHash0;
            if( hash1 )
                continue;

            int nHash1   = nHash0 | gaHash[ word1 ];
            int nOffset2 = gaNeighbors[ word1 ][ 0 ];

            for (int iOffset2 = 1; iOffset2 < nOffset2; ++iOffset2)
            {
                int word2 = gaNeighbors[ word1 ][ iOffset2 ];
                int hash2 = nHash1 & gaHash[ word2 ];
                if( hash2 )
                    continue;

                int nHash2   = nHash1 | gaHash[ word2 ];
                int nOffset3 = gaNeighbors[ word2 ][ 0 ];

                for (int iOffset3 = 1; iOffset3 < nOffset3; ++iOffset3)
                {
                    int word3 = gaNeighbors[ word2 ][ iOffset3 ];
                    int hash3 = nHash2 & gaHash[ word3 ];
                    if( hash3 )
                        continue;

                    int nHash3   = nHash2 | gaHash[ word3 ];
                    int nOffset4 = gaNeighbors[ word3 ][ 0 ];

                    for (int iOffset4 = 1; iOffset4 < nOffset4; ++iOffset4)
                    {
                        int word4 = gaNeighbors[ word3 ][ iOffset4 ];
                        int hash4 = nHash3 & gaHash[ word4 ];
                        if( hash4 )
                            continue;

                        short   iSolutions   = gaSolutions[ iThread ];
                        short  *pSolution    = &gaOutput[ iThread ][ iSolutions*NUM_WORDS ];
                                pSolution[0] = (short) word0;
                                pSolution[1] = (short) word1;
                                pSolution[2] = (short) word2;
                                pSolution[3] = (short) word3;
                                pSolution[4] = (short) word4;
                        ++gaSolutions[ iThread ];
                    }
                }
            }
        }
    }
}

// ======================================================================
void Solutions()
{
    // Gather
    int nTotal   = 0;
    int nThreads = 0;

    for (int iThread = 0; iThread < MAX_THREADS; ++iThread)
    {
        nTotal     +=  gaSolutions[ iThread ]     ;
        nThreads   += (gaSolutions[ iThread ] > 0);

        if (gaSolutions[ iThread ] > 0)
            printf( "Thread %d found %d solutions:\n", iThread, gaSolutions[ iThread ] );

        for (int iSolution = 0; iSolution < gaSolutions[ iThread ]; ++iSolution)
        {
            short *pWord = &gaOutput[ iThread ][ iSolution*NUM_WORDS ];
            printf( "    %s, %s, %s, %s, %s,\n", gaWords[ pWord[0] ], gaWords[ pWord[1] ], gaWords[ pWord[2] ], gaWords[ pWord[3] ], gaWords[ pWord[4] ] );
        }
    }

    printf( "Solutions: %d   \n", nTotal   );
    printf( "Threads with solutions: %d\n", nThreads );
}

// ======================================================================
int main( int nArg, char *aArg[] )
{
    auto begin = std::chrono::high_resolution_clock::now();

        int gnCurThreads = 0; // auto-detect, use max threads
        if (nArg > 1)
            gnCurThreads = atoi( aArg[ 1 ] );

        int gnMaxThreads = omp_get_max_threads(); // omp_get_num_procs();
        omp_set_num_threads( gnCurThreads );
        gnCurThreads = gnCurThreads ? gnCurThreads : gnMaxThreads;
        printf( "Using %d / %d threads\n", gnCurThreads, gnMaxThreads );

        Init();
        Read4( (nArg > 2) ? aArg[2] : "words_alpha.txt" ); // NOTE: words_alpha.txt (in MS-DOS format) has varying lengths of non-unique words
        Parse();
        Prepare();
        Search3();
        Solutions();

    auto end    = std::chrono::high_resolution_clock::now();
    int ms      = (int) std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    int minutes = (ms / 1000) / 60;
    int seconds = (ms / 1000) % 60;
    printf( "%d:%02d = %d seconds (%d ms)\n", minutes, seconds, (ms / 1000), ms );
    return 0;
}
