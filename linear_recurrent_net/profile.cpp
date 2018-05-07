#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <string>

#include "linear_recurrence.h"


int main(int argc, char** argv)
{
    if (argc == 3) {
        char *alg = argv[1];
        int n_iters = atoi(argv[2]);
        printf("%d iterations\n", n_iters);
        if (strcmp(alg, "baseline") == 0) {
            profile_base(n_iters);
        } else if (strcmp(alg, "fast") == 0) {
            profile_fast(n_iters);
        } else if (strcmp(alg, "serial") == 0) {
            profile_serial(n_iters);
        }
    } else {
        printf("Usage: ./profile [baseline|fast|serial] <num_iters>\n");
    }
	return 0;
}
