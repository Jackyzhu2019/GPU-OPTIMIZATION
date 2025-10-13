#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "./common.h"
#include <stdint.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

#define DOUBLE(pointer) (reinterpret_cast<double*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define float16_t half
//#define printf(...);

void Mimo64_alloc_host_mem(void** host_ptr_addr, size_t size)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    err = cudaMallocHost((void **)host_ptr_addr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate host memory (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	return;
}

void Mimo64_free_host_mem(void* host_ptr)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    if (host_ptr != NULL)
    {
        err = cudaFreeHost(host_ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to free host memory (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
	
	return;
}

void Mimo64_alloc_device_mem(void** dev_ptr_addr, size_t size)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
  
	err = cudaMalloc((void**)dev_ptr_addr, size);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate cuda device mem (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	return;
}

void Mimo64_free_device_mem(void* dev_ptr)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    if (dev_ptr != NULL){
		err = cudaFree(dev_ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to free cuda device mem (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
	}

	return;
}


void initialData_f32(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)rand() / (float)RAND_MAX;
		//printf("val[%d]: %d \n", i, ip[i]);
    }

	return;
}


void initialData_Y_f32(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
		if (i < size)
			ip[i] = (float) 1.0f; //(rand() & 0xFF);
		//printf("val[%d]: %d \n", i, ip[i]);
		else
			ip[i] = (float)0.0; //(rand() & 0xFF);
		//printf("val[%d]: %d \n", i, ip[i]);
    }

	return;
}

void Mimo64_init_device_const_mem()
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

 

	return;
}


void float2half(float16_t *dst, float *src, int nElem){
	for (int i = 0; i < nElem; i++){
		dst[i] = __float2half(src[i]);
	}
	
	return;
}

cudaStream_t *streams;

void Mimo64_createStreams(int numOfStreams){

	streams = (cudaStream_t *)malloc(numOfStreams * sizeof(cudaStream_t));

	for (int i = 0; i < numOfStreams; i++)
		cudaStreamCreate(&streams[i]);

	return;
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-2;
    bool match = 1;
	
//	printf("hello world. \n");
//		printf("host: %f gpu: %f \n", hostRef[0], gpuRef[0]);
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



void conv_naive_kernel(
	float *input,      // input image: [batch_size][in_channels][height][width]
    float *kernel,     // conv kernel: [out_channels][in_channels][kernel_h][kernel_w]
    float *bias,       // bias： [out_channels] (可为NULL)
    float *output,     // output: [batch_size][out_channels][out_h][out_w]
    int batch_size,    // batch
    int in_channels,   // input channel: RGB
    int out_channels,  // output channel: num of kernels
    int height,        // image height
    int width,         // image width
    int kernel_h,      // kernel height
    int kernel_w,      // kernel width
    int stride_h,      // stride height
    int stride_w,      // stride width
    int padding_h,     // padding height
    int padding_w      // padding width
) 
{
	int iBatch, iKernel, iChannel;
	int iRow, jCol;
	
	// output size in one batch one kernel
	int out_height = (height + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * padding_w - kernel_w) / stride_w + 1;
  
	// input size per batch
 	int numPixelperChannel = height * width; // input image
	int numPixelperBatch = numPixelperChannel * in_channels;
	
	int numEleperKernelperChannel = kernel_h * kernel_w; // input kernel
	int numEleperKernel = numEleperKernelperChannel * in_channels;
	
	int numElePerOutKernel = out_height * out_width;
	int numElePerOutBatch = out_channels *  numElePerOutKernel; // output

	// output initialization
	int output_size = batch_size * numElePerOutBatch;
    for (int i = 0; i < output_size; i++) {
        output[i] = 0.0f;
    }
	

	for (iBatch = 0; iBatch < batch_size; iBatch++){
		// for each kernel
		for (iKernel = 0; iKernel < out_channels; iKernel++){
			// locate to given img
			float *imgIn = input + iBatch * numPixelperBatch;
			float *kerIn = kernel + iKernel * numEleperKernel;
			float *convOut = output + iBatch * numElePerOutBatch + iKernel * numElePerOutKernel;
			
			// for each RGB input channels
			for (iChannel = 0; iChannel < in_channels; iChannel++){
				float *imgIn_Ch = imgIn + iChannel * numPixelperChannel;
				float *kerIn_Ch = kerIn + iChannel * numEleperKernelperChannel;
				
				// for each element in output 
				for (int oh = 0; oh < out_height; oh++) {
                    for (int ow = 0; ow < out_width; ow++) {
                        // start pos in the given imag
                        int start_h = oh * stride_h - padding_h;
                        int start_w = ow * stride_w - padding_w;
                        int output_idx = oh * out_width + ow;
		
								
						// every output element need loop conv kernel
						for (int kh = 0; kh < kernel_h; kh++) {
                            for (int kw = 0; kw < kernel_w; kw++) {
                                int ih = start_h + kh;
                                int iw = start_w + kw;
                                
                                // check if it is padding
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    int input_idx = ih * width + iw;
                                    int kernel_idx = kh * kernel_w + kw;
                                    
                                    // conv
                                    convOut[output_idx] += imgIn_Ch[input_idx] * kerIn_Ch[kernel_idx];
                                } else {
									// if padding 0, then do nothing
									
								}
							}
                        }
					}
				}
			}
			
			// add bias
			float bias_val = bias ? bias[iKernel] : 0.0f;
			
			//printf("bias_val : %f \n", bias_val);
			if (bias) {
                for (int oh = 0; oh < out_height; oh++) {
                    for (int ow = 0; ow < out_width; ow++) {
                        int output_idx = oh * out_width + ow;
                        convOut[output_idx] += bias_val;
                    }
                }
            }
			
		}
	}
	

	return;
}


/**
 * 
 * im2col - covert input image data to col matrix
 * 
 */
void im2col(
	const float* input, 
	float* data_col,
	int channels, 
	int height, 
	int width,
	int kernel_h, 
	int kernel_w,
	int pad_h, 
	int pad_w,
	int stride_h, 
	int stride_w)
{
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // col matrix size
    int col_rows = channels * kernel_h * kernel_w;
    int col_cols = out_h * out_w;
    
	// loop every element in output col matrix
	for (int oh = 0; oh < out_h; oh++) {
		for (int ow = 0; ow < out_w; ow++) {
            // start pos in the given imag
			int start_h = oh * stride_h - pad_h;
			int start_w = ow * stride_w - pad_w;
						
			// col idx in output col matrix
			int col_idx = oh * out_w + ow;
			
			// every output element need loop conv kernel
			for (int kh = 0; kh < kernel_h; kh++) {
				for (int kw = 0; kw < kernel_w; kw++) {
					// actual idx in the orig imag
					int ih = start_h + kh;
					int iw = start_w + kw;
					
					// every channel (RGB)
					for (int c = 0; c < channels; c++) {
						// row idx in output col matrix
						int row_idx = c * (kernel_h * kernel_w) + kh * kernel_w + kw;
			
						
						// input index
						int input_idx = c * (height * width) + ih * width + iw;
						
                        // check if it is padding
						if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
							data_col[row_idx * col_cols + col_idx] = input[input_idx];
						} else {
							data_col[row_idx * col_cols + col_idx] = 0.0f;
						}
					}
				}
			}
		}
	}
	
	return;
}


/**
 * 
 * matmul_add_bias - coverted input image matrix * kernel + bias
 * 
 */
void matmul_add_bias(
	const float* data_col, 
	const float* kernel, 
	const float* bias, 
	float* output,
    int out_channels, 
	int col_rows, 
	int col_cols, 
	int out_h, 
	int out_w) 
{
    // for each filter kernel
    for (int oc = 0; oc < out_channels; oc++) {
        float bias_val = bias ? bias[oc] : 0.0f;
        
        
        for (int col = 0; col < col_cols; col++) {
            float sum = 0.0f;
            
            // convolution
            for (int row = 0; row < col_rows; row++) {
                sum += data_col[row * col_cols + col] * kernel[oc * col_rows + row];
            }
            
            // bias
            sum += bias_val;
            
            // output index
           // int pos = col % (out_h * out_w);
           // int oh = pos / out_w;
           // int ow = pos % out_w;
            
            int output_idx = oc * (out_h * out_w) + col;
            
            output[output_idx] = sum;
        }
    }
}



void conv_im2col_kernel(	
	float *input,      // input image: [batch_size][in_channels][height][width]
    float *kernel,     // conv kernel: [out_channels][in_channels][kernel_h][kernel_w]
    float *bias,       // bias： [out_channels] (可为NULL)
    float *output,     // output: [batch_size][out_channels][out_h][out_w]
	float *data_col,   // converted input image --> col matrix: [batch_size][kernel height * kernel width * in_channels][out_h * out_w]
    int batch_size,    // batch
    int in_channels,   // input channel: RGB
    int out_channels,  // output channel: num of kernels
    int height,        // image height
    int width,         // image width
    int kernel_h,      // kernel height
    int kernel_w,      // kernel width
    int stride_h,      // stride height
    int stride_w,      // stride width
    int padding_h,     // padding height
    int padding_w      // padding width
){
	int iBatch;
    int out_h = (height + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * padding_w - kernel_w) / stride_w + 1;
    
	// input size per batch
 	int numPixelperChannel = height * width; // input image
	int numPixelperBatch = numPixelperChannel * in_channels;
	int numElePerOutKernel = out_h * out_w;
	int numElePerOutBatch = out_channels *  numElePerOutKernel; // output

	int numColMatrixperBatch = kernel_h * kernel_w * in_channels * out_h * out_w; 
    // converted input image --> col matrix
    int col_rows = in_channels * kernel_h * kernel_w;
    int col_cols = out_h * out_w;
	
	for (iBatch = 0; iBatch < batch_size; iBatch++){
		float *input_batch = input + numPixelperBatch * iBatch;
		float *data_col_batch = data_col + numColMatrixperBatch * iBatch; // actually we can use one batch for intermediate use
		float *output_batch = output + numElePerOutBatch * iBatch;
		
		im2col(input_batch, data_col_batch, in_channels, height, width,
           kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w);
    
		matmul_add_bias(data_col_batch, kernel, bias, output_batch, 
                    out_channels, col_rows, col_cols, out_h, out_w);
 
	}

	return;
}


__global__ void conv_im2col_block_gpu_kernel(float* Q,
												float* K,
												float* V,
												float* O,
												int N,
												int d,
												int Br,
												int Bc)
{
	int i, j, k;
	int iRow, jCol, id;

	// inside a block
	int xIdx = threadIdx.x;
	int yIdx = threadIdx.y;
	// block shape
	int xLen = blockDim.x;
	int yLen = blockDim.y;
	// block index
	int Block_X_Idx = blockIdx.x;
	int Block_Y_Idx = blockIdx.y; // 0
	
	int tid_x = yIdx * xLen + xIdx;


	return;
}


int main(int argc, char **argv)
{
    printf("> %s Starting...\n", argv[0]);
	
    // GPT-2 parameters
    const int nBatch = 32; // number of batchs
	const int imgHeight = 128; // image height
	const int imgWidth = 128; // image width
	const int kernelHeight = 5; // kernel Height
	const int kernelWidth = 5; // kernel Height
	const int stride_h = 1; // stride in height direction
	const int stride_w = 1; // stride in width direction
	const int padding_h = 2; // padding in height direction
	const int padding_w = 2; // paddign in width direction
	const int inChannel = 3; // 3 channels per image
	const int outChannel = 2; // 2 kernels debug

    float *inImg;
	float *inImgCol;
	float *kernel;
	float *bias;
	float *O_base; // output for reference
	float *O; // output
	float *S; // temporary result 
	
	int out_height = (imgHeight + 2 * padding_h - kernelHeight) / stride_h + 1;
    int out_width = (imgWidth + 2 * padding_w - kernelWidth) / stride_w + 1;
    // converted input image --> col matrix
	int inImgColSize = nBatch * out_height * out_width * inChannel * kernelHeight * kernelWidth;
	
	int imgSize = nBatch * inChannel * imgHeight * imgHeight; // 1.5MB
	int kernelSize = outChannel * kernelHeight * kernelWidth; // 50B
	// keep same size with input
	int oSize = nBatch * outChannel * out_height * out_width; // 1MB

	printf("img size: %d kernel size: %d converted col matrix: %d out size: %d \n", imgSize, kernelSize, inImgColSize, oSize);
		
	Mimo64_alloc_host_mem((void **)&inImg, imgSize * sizeof(float)); // 6MB
	Mimo64_alloc_host_mem((void **)&kernel, kernelSize * sizeof(float));
	Mimo64_alloc_host_mem((void **)&bias, outChannel * sizeof(float));
	Mimo64_alloc_host_mem((void **)&inImgCol, inImgColSize * sizeof(float));


	Mimo64_alloc_host_mem((void **)&O_base, oSize * sizeof(float));
	Mimo64_alloc_host_mem((void **)&O, oSize * sizeof(float));

	float *d_inImg;
	float *d_kernel;
	float *d_bias;
	float *d_O;

	Mimo64_alloc_device_mem((void **)&d_inImg, imgSize * sizeof(float));
	Mimo64_alloc_device_mem((void **)&d_kernel, kernelSize * sizeof(float));
	Mimo64_alloc_device_mem((void **)&d_O, oSize * sizeof(float));
	Mimo64_alloc_device_mem((void **)&d_bias, outChannel * sizeof(float));

    memset(O, 0, oSize * sizeof(float));
    memset(O_base,  0, oSize * sizeof(float));

 	initialData_f32(inImg, imgSize);
 	initialData_f32(kernel, kernelSize);
 	initialData_f32(bias, outChannel);
	
	//bias = NULL;
#if 1
	long t_start = useconds();
	conv_naive_kernel
	(
		inImg,
		kernel,
		bias,
		O_base,
		nBatch,
		inChannel,
		outChannel,
		imgHeight,
		imgWidth,
		kernelHeight,
		kernelWidth,
		stride_h,
		stride_w,
		padding_h,
		padding_w
	);
	//printf("O_base: %f %f \n", O_base[0], O_base[1]);
	
	long t_end = useconds();
	printf("conv_naive_kernel() costs %ld us \n", (t_end - t_start) );
#endif

#if 1
	long t_start1 = useconds();
	
	conv_im2col_kernel
	(
		inImg,
		kernel,
		bias,
		O,
		inImgCol,
		nBatch,
		inChannel,
		outChannel,
		imgHeight,
		imgWidth,
		kernelHeight,
		kernelWidth,
		stride_h,
		stride_w,
		padding_h,
		padding_w
	);
	
	long t_end1 = useconds();
	printf("conv_im2col_kernel() costs %ld us \n", (t_end1 - t_start1) );

	checkResult(O_base, O, oSize);
#endif






	Mimo64_free_host_mem(inImg);
	Mimo64_free_host_mem(kernel);
	Mimo64_free_host_mem(bias);
	Mimo64_free_host_mem(O);
	Mimo64_free_host_mem(O_base);
	
	Mimo64_free_device_mem(d_bias);
	Mimo64_free_device_mem(d_inImg);
	Mimo64_free_device_mem(d_kernel);
	Mimo64_free_device_mem(d_O);

	return 0;
}