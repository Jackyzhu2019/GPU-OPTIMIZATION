#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>
#include <arm_neon.h>

inline long useconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((long)tp.tv_sec * 1000000 + (long)tp.tv_usec);
}

void initialData(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}


void initialData_u32(uint32_t *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (uint32_t)(rand() & 0xFFFFFFFF);
    }
}

void initialData_u8(uint8_t *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (uint8_t)(rand() & 0xFF);
    }
}


void initialData_s16(int16_t *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (int16_t)(rand() & 0xFFFF);
		//printf("val[%d]: %d \n", i, ip[i]);
    }
}

void initialData_f32(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFFFF);
		//printf("val[%d]: %d \n", i, ip[i]);
    }
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    int match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %f gpu %f at %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");
}


void mimo64_naive_kernel(float *G, float *Y, float *X, int nElem, int nElemPerMatrix)
{
	int numOfloop = nElem / nElemPerMatrix;
	int loopIdx;
	int iRow, iCol, iK;
	float *G0, *Y0, *X0;
	
	for (loopIdx = 0; loopIdx < numOfloop; loopIdx++)
	{
		G0 = G + 64 * 64 * loopIdx;
		Y0 = Y + 64 * nElemPerMatrix * loopIdx;
		X0 = X + 64 * nElemPerMatrix * loopIdx;
		
		// G matrix (64 x 64) x Y matrix (64 x nElemPerMatrix(4))
		for (iRow = 0; iRow < 64; iRow++)
		{
			for (iCol = 0; iCol < nElemPerMatrix; iCol++)
			{
				float accuVal = 0.0f;
				
				for (iK = 0; iK < 64; iK++)
				{
					
					accuVal += G0[iRow * 64 + iK] * Y0[iK * nElemPerMatrix + iCol];
				
				}
				
				X0[iRow * nElemPerMatrix + iCol] = accuVal;
			}
		}
	}


	return;
}

#define BLOCK_SIZE_ROW 8
#define BLOCK_SIZE_COL 4
#define BLOCK_SIZE_K 8

void mimo64_block_kernel(float *G, float *Y, float *X, int nElem, int nElemPerMatrix)
{
	int numOfloop = nElem / nElemPerMatrix;
	int loopIdx;
	int iRow, iCol, iK;
	float *G0, *Y0, *X0;
	int i, j, k, t;
	
	for (loopIdx = 0; loopIdx < numOfloop; loopIdx++)
	{
		G0 = G + 64 * 64 * loopIdx;
		Y0 = Y + 64 * nElemPerMatrix * loopIdx;
		X0 = X + 64 * nElemPerMatrix * loopIdx;
		
		// G matrix (64 x 64) x Y matrix (64 x nElemPerMatrix(4))
		for (iRow = 0; iRow < 64; iRow += BLOCK_SIZE_ROW)
		{
			for (iCol = 0; iCol < nElemPerMatrix; iCol += BLOCK_SIZE_COL)
			{
				float accuVal[BLOCK_SIZE_ROW][BLOCK_SIZE_COL];
				
				for (i = 0; i < BLOCK_SIZE_ROW; i++)
				{
					for (j = 0; j < BLOCK_SIZE_ROW; j++)
					{
						accuVal[i][j] = 0.0f;
					}
				}
				
				
				for (iK = 0; iK < 64; iK += BLOCK_SIZE_K)
				{
					
					for (i = 0; i < BLOCK_SIZE_ROW; i++)
					{	
						for (j = 0; j < BLOCK_SIZE_COL; j++)
						{	
							for (k = 0; k < BLOCK_SIZE_K; k++){
								accuVal[i][j] += G0[(iRow + i) * 64 + iK + k] * Y0[(iK + k) * nElemPerMatrix + iCol + j];		
							}
						}
					}
				}
				
				
				for (i = 0; i < BLOCK_SIZE_ROW; i++)
				{
					for (j = 0; j < BLOCK_SIZE_ROW; j++)
					{
						X0[(iRow + i) * nElemPerMatrix + iCol + j] = accuVal[i][j];
					}
				}		
			}
		}
	}


	return;
}


void mimo64_cache_kernel(float *G, float *Y, float *X, int nElem, int nElemPerMatrix)
{
	int numOfloop = nElem / nElemPerMatrix;
	int loopIdx;
	int iRow, iCol, iK;
	float *G0, *Y0, *X0;
	int i, j, k, t;
	float G_cache[64][64];
	float Y_cache[4][64];
	
	for (loopIdx = 0; loopIdx < numOfloop; loopIdx++)
	{
		G0 = G + 64 * 64 * loopIdx;
		Y0 = Y + 64 * nElemPerMatrix * loopIdx;
		X0 = X + 64 * nElemPerMatrix * loopIdx;
		
		
		// prefetch the data to L1 cahce
		for (int i = 0; i < 64; i++)
		{
			for (int j = 0; j < 64; j++)
			{
				G_cache[i][j] = *G0++;
			}			
		}
		
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 64; j++)
			{
				// transpose
				Y_cache[i][j] = Y0[j * nElemPerMatrix + i];
			}
		}
/*		
		if (loopIdx == 0){
			for (int i = 0; i < 64; i++)
			{
				for (int j = 0; j < 64; j++)
				{
					printf("i: %d j: %d val: %f \n", i, j, G_cache[i][j]);
				}			
			}
			
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 64; j++)
				{
					// transpose
					printf("i: %d j: %d val: %f \n", i, j, Y_cache[i][j]);
				}		
			}
		}
*/	
		
		// G matrix (64 x 64) x Y matrix (64 x nElemPerMatrix(4))
		for (iRow = 0; iRow < 64; iRow += BLOCK_SIZE_ROW)
		{
			for (iCol = 0; iCol < nElemPerMatrix; iCol += BLOCK_SIZE_COL)
			{
				float accuVal[BLOCK_SIZE_ROW][BLOCK_SIZE_COL];
				for (i = 0; i < BLOCK_SIZE_ROW; i++)
				{
					for (j = 0; j < BLOCK_SIZE_ROW; j++)
					{
						accuVal[i][j] = 0.0f;
					}
				}
				
				
				for (iK = 0; iK < 64; iK += BLOCK_SIZE_K)
				{
					for (i = 0; i < BLOCK_SIZE_ROW; i++)
					{	
						for (j = 0; j < BLOCK_SIZE_COL; j++)
						{	
							for (k = 0; k < BLOCK_SIZE_K; k++){
								accuVal[i][j] += G_cache[iRow + i][iK + k] * Y_cache[iCol + j][iK + k];		
							}
						}
					}
				}
				
				
				for (i = 0; i < BLOCK_SIZE_ROW; i++)
				{
					for (j = 0; j < BLOCK_SIZE_ROW; j++)
					{
						X0[(iRow + i) * nElemPerMatrix + iCol + j] = accuVal[i][j];
					}
				}		
			}
		}
	}


	return;
}


int main(int argc, char **argv)
{
    printf("> %s Starting...\n", argv[0]);
	
    // set up data size of vectors
    int nElem = 273 * 12 * 14;
	int nElemPerMatrix = 4;
    printf("> vector size = %d\n", nElem);

    float *G;
	float *Y;
	float *N0;
	float *X_base;
	float *X;
	
	G = (float *)malloc((nElem / nElemPerMatrix) * 64 * 64 * sizeof(float));
	Y = (float *)malloc(nElem * 64 * sizeof(float));
//	N0 = (float *)malloc(nElem * 64 * sizeof(float));
	X = (float *)malloc(nElem * 64 * sizeof(float));
	X_base = (float *)malloc(nElem * 64 * sizeof(float));

	
	initialData_f32(G, (nElem / nElemPerMatrix) * 64 * 64);
	initialData_f32(Y, nElem * 64);

	mimo64_naive_kernel(G, Y, X_base, nElem, nElemPerMatrix);

	float t_val[10];
	long t_start = useconds();

	mimo64_naive_kernel(G, Y, X_base, nElem, nElemPerMatrix);
	
	long t_end = useconds();
	printf("mimo64_naive_kernel() costs %ld us \n", (t_end - t_start) );
	t_val[0] = t_end - t_start;

	t_start = useconds();
	mimo64_naive_kernel(G, Y, X_base, nElem, nElemPerMatrix);	
	t_end = useconds();
	printf("mimo64_naive_kernel() costs %ld us \n", (t_end - t_start) );
	t_val[1] = t_end - t_start;

	t_start = useconds();
	mimo64_naive_kernel(G, Y, X_base, nElem, nElemPerMatrix);	
	t_end = useconds();
	printf("mimo64_naive_kernel() costs %ld us \n", (t_end - t_start) );
	t_val[2] = t_end - t_start;

	long t_val_avg = (t_val[1] + t_val[2]) / 2;
	int num_mads = nElem * 64 * 64; // number of multiply-and-accumulate ops

	double measured_naive_throughput = (num_mads / (t_val_avg * 0.000001)) / 1e9;
	printf("num_mads: %d time: %ld throughput: %f Gflops\n", num_mads, t_val_avg, measured_naive_throughput);

	t_start = useconds();
	mimo64_block_kernel(G, Y, X, nElem, nElemPerMatrix);	
	t_end = useconds();

	printf("mimo64_block_kernel() costs %ld us \n", (t_end - t_start) );
	
	t_start = useconds();
	mimo64_block_kernel(G, Y, X_base, nElem, nElemPerMatrix);	
	t_end = useconds();

	printf("mimo64_block_kernel() costs %ld us \n", (t_end - t_start) );


	t_start = useconds();
	mimo64_cache_kernel(G, Y, X, nElem, nElemPerMatrix);
	t_end = useconds();

	printf("mimo64_cache_kernel() costs %ld us \n", (t_end - t_start) );

	t_start = useconds();
	mimo64_cache_kernel(G, Y, X, nElem, nElemPerMatrix);
	t_end = useconds();

	printf("mimo64_cache_kernel() costs %ld us \n", (t_end - t_start) );
	
	checkResult(X, X_base, nElem * 64);

	return 0;
}