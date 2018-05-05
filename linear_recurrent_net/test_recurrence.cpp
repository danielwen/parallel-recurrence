#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "linear_recurrence.h"





int main(int argc, char** argv)
{
	float* fast_out;
	float* serial_out;

	int n_dims = 20; //100;
  	int n_steps = 20; //1000000;
  	int n_elems = n_dims * n_steps;

	serial_out = test_base(n_dims, n_steps);
	printf("\n\n");
	fast_out = test_fast(n_dims, n_steps);

	bool correct = true;
	for(int i=0; i<n_elems; i++)
	{
		if(serial_out[i] != fast_out[i]) 
		{
			correct = false;
		}
	}

    printf("\n\nSerial ");
    printf("\nFast \n");
    for(int i=0; i<n_elems; i++)
    {
      printf("%f ", serial_out[i]);
    }

    printf("\n");
    for(int i=0; i<n_elems; i++)
    {
      printf("%f ", fast_out[i]);
    }

	printf("\n\nRESULT: %d\n", correct);

	return 0;
}



