#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "linear_recurrence.h"





int main(int argc, char** argv)
{
	float* fast_out;
	float* serial_out;

	int n_dims = 100;
  	int n_steps = 200; //1000000;
  	int n_elems = n_dims * n_steps;

  	printf("Testing with n_dims=%d, n_steps=%d", n_dims, n_steps);

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

    // printf("\n\nSerial ");
    // printf("\nFast \n");
    // for(int i=0; i<n_elems; i++)
    // {
    //   printf("%f ", serial_out[i]);
    // }

    // printf("\n");
    // for(int i=0; i<n_elems; i++)
    // {
    //   printf("%f ", fast_out[i]);
    // }

    std::string result;
    if (correct) 
    	result = "Correct";
    else
    	result = "WRONG";

	printf("\n\nRESULT: %s\n", result.c_str());

	return 0;
}



