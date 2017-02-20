#include <stdlib.h>
#include <time.h>

float randomGen(int n)
{
	//srand(time(NULL));
	int min = int(-20000 / float(n));
	int max = int(20000 / float(n));
	return ((rand() % (max - min + 1) + min) / float(10000));
}