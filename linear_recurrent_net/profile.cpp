#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <string>

#include "linear_recurrence.h"


int main(int argc, char** argv)
{
    if (argc == 2) {
        char *alg = argv[1];
        if (strcmp(alg, "baseline") == 0) {
            profile_base();
        } else if (strcmp(alg, "fast") == 0) {
            profile_fast();
        } else if (strcmp(alg, "serial") == 0) {
            profile_serial();
        }
    } else {
        printf("Usage: ./profile [baseline|fast|serial]\n");
    }
	return 0;
}
