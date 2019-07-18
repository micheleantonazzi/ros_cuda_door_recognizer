//
// Created by michele on 18/07/19.
//

#include <sys/time.h>
#include "utilities.h"

double Utilities::seconds(){
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}