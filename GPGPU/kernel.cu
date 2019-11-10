
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
int const OUT = 10000;
int const IN = 100000;


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void initkernel()
{
}

void floatCPU()
{
	float* data = (float*)malloc(sizeof(float) * OUT);

	for (int i = 1; i < OUT; i++)
	{
		data[i] = 1.0f * i / OUT;
		for (int j = 1; j < IN; j++)
			data[i] = data[i] / IN * data[i] / IN - 0.50f;
	}
}


__global__ void floatkernel(float* buf)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	buf[i] = 1.0f * i / OUT;
	for (int j = 0; j < IN; j++)
		buf[i] = buf[i] / IN * buf[i] / IN - 0.50f;
}
void floatGPU()
{
	int count = 0;
	float* data = (float*)malloc(sizeof(float) * OUT);
	float* d_data;
	cudaMalloc(&d_data, OUT * sizeof(float));
	floatkernel << <OUT / 1024, 1024 >> > (d_data);
	cudaDeviceSynchronize();
	cudaMemcpy(data, d_data, OUT * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_data);
}


void floatcomputing()
{
	auto start = std::chrono::high_resolution_clock::now();
	floatGPU();
	auto finish = std::chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	std::cout << "GPU time: " << elapsed.count() << " s\n";

	start = std::chrono::high_resolution_clock::now();
	floatCPU();
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << "CPU time: " << elapsed.count() << " s\n";
}

void RunCPU(int size)
{
	int* input1 = (int*)malloc(sizeof(int) * size);
	int* input2 = (int*)malloc(sizeof(int) * size);
	int* result = (int*)malloc(sizeof(int) * size);

	for (int i = 0; i < size; i++)
	{
		input1[i] = i;
		input2[i] = i;
		result[i] = 0;
	}

	for (int i = 0; i < size; i++)
	{
		result[i] = input1[i] + input2[i];
	}

	free(input1);
	free(input2);
	free(result);
}

__global__ void GPUadd(int* input1, int* input2, int* result, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size) // vypnut
	{
		result[tid] = input1[tid] + input2[tid];
	}
}


void RunGPU(int size)
{
	int* input1 = (int*)malloc(sizeof(int) * size);
	int* input2 = (int*)malloc(sizeof(int) * size);
	int* result = (int*)malloc(sizeof(int) * size);

	int* dev_input1, * dev_input2, * dev_result;

	cudaMalloc(&dev_input1, sizeof(int) * size);
	cudaMalloc(&dev_input2, sizeof(int) * size);
	cudaMalloc(&dev_result, sizeof(int) * size);

	for (int i = 0; i < size; i++)
	{
		input1[i] = i;
		input2[i] = i;
		result[i] = 0;
	}

	int block_size = 1024;
	int grid_size = (int)ceil((float)size / block_size);

	cudaMemcpy(dev_input1, input1, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_input2, input2, sizeof(int) * size, cudaMemcpyHostToDevice);

	GPUadd << <grid_size, block_size >> > (dev_input1, dev_input2, dev_result, size);
	cudaDeviceSynchronize();

	cudaMemcpy(result, dev_result, sizeof(int) * size, cudaMemcpyDeviceToHost);

	free(input1);
	free(input2);
	free(result);

	cudaFree(dev_input1);
	cudaFree(dev_input2);
	cudaFree(dev_result);
}

void memory_copy(int size)
{

	auto start = std::chrono::high_resolution_clock::now();
	RunGPU(size);
	auto finish = std::chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	cout << "GPU time: " << elapsed.count() << " s\n";


	start = std::chrono::high_resolution_clock::now();
	RunCPU(size);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << "CPU time: " << elapsed.count() << " s\n";
}

__global__ void matrix_kernel(int* m, int* n, int* result, int size)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int result_sum = 0;

	for (int i = 0; i < size; i++)
	{
		result_sum += m[row * size + i] * n[i * size + column];
	}

	result[row * size + column] = result_sum;
}


void matrix_cpu(int* m, int* n, int* result, int size)
{
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			for (int k = 0; k < size; k++)
			{
				result[i * size + j] += m[i * size + k] * n[k * size + j];
			}
}
void Matrix_mul(int size)
{
	int bytes = size * size * sizeof(int);

	int* host_m = (int*)malloc(bytes);
	int* host_n = (int*)malloc(bytes);
	int* host_result = (int*)malloc(bytes);
	int* cpu_result = (int*)malloc(bytes);

	int* dev_matrix1, * dev_matrix1atrix2, * dev_result;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			//host_m[i * size + j] = rand() % 1024;
			//host_n[i * size + j] = rand() % 1024;
			host_m[i * size + j] = 5;
			host_n[i * size + j] = 5;
			cpu_result[i * size + j] = 0;
		}
	}


	int threads_max = 16;
	dim3 block_size(threads_max, threads_max);
	dim3 grid_size(size / block_size.x, size / block_size.y);

	auto start = std::chrono::high_resolution_clock::now();
	cudaMalloc(&dev_matrix1atrix2, bytes);
	cudaMalloc(&dev_matrix1, bytes);
	cudaMalloc(&dev_result, bytes);
	cudaMemcpy(dev_matrix1atrix2, host_n, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matrix1, host_m, bytes, cudaMemcpyHostToDevice);
	matrix_kernel << <1, 1024 >> > (dev_matrix1, dev_matrix1atrix2, dev_result, size);
	cudaDeviceSynchronize();
	cudaMemcpy(host_result, dev_result, bytes, cudaMemcpyDeviceToHost);
	auto finish = std::chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	cout << "GPU time: " << elapsed.count() << " s\n";

	start = std::chrono::high_resolution_clock::now();
	matrix_cpu(host_m, host_n, cpu_result, size);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << "CPU time: " << elapsed.count() << " s\n";


	printf("%d %d\n", host_result[1], cpu_result[1]);

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (cpu_result[size * i + j] != host_result[size * i + j])
			{
				printf("Chybne vypocitana matica!\n");
				bool exit = true;
				break;
			}
		}
		if (exit) break;
	}

	free(host_n);
	free(host_m);
	free(host_result);

	cudaFree(dev_matrix1atrix2);
	cudaFree(dev_matrix1);
	cudaFree(dev_result);
}

__global__ void fibonaci_kernel(int size)
{
	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//if (tid) // vypnut
	//{
	//	
	//}

	for (int i = 0; 1000000 < 0; i++)
	{

		int t1 = 0, t2 = 1, nextTerm = 0;
		nextTerm = t1 + t2;
		while (nextTerm <= size)
		{
			//printf("%d, ", nextTerm);
			t1 = t2;
			t2 = nextTerm;
			nextTerm = t1 + t2;
		}

	}


}

void fibonaciGPU(int size)
{
	fibonaci_kernel << <1, 1 >> > (size);

}

void fibonaciCPU(int size)
{
	for (int i = 0; i < 1000000; i++)
	{

		int t1 = 0, t2 = 1, nextTerm = 0;
		nextTerm = t1 + t2;
		while (nextTerm <= size)
		{
			//printf("%d, ", nextTerm);
			t1 = t2;
			t2 = nextTerm;
			nextTerm = t1 + t2;
		}

	}
}

void fibonaci(int size)
{
	auto start = std::chrono::high_resolution_clock::now();
	fibonaciCPU(size);
	auto finish = std::chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = finish - start;
	cout << "CPU time: " << elapsed.count() << " s\n";

	start = std::chrono::high_resolution_clock::now();
	fibonaciGPU(size);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << "GPU time: " << elapsed.count() << " s\n";


}

__global__ void Image_kernel(int* m, int* n, int* result, int size)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int result_sum = 0;

	for (int i = 0; i < size; i++)
	{
		result_sum += m[row * size + i] * n[i * size + column];
	}

	result[row * size + column] = result_sum;
}

__global__ void bgr_to_gray_kernel(unsigned char* input,
	unsigned char* output,
	int width,
	int height,
	int colorWidthStep,
	int grayWidthStep)
{
	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

		//Location of gray pixel in output
		const int gray_tid = yIndex * grayWidthStep + xIndex;

		const unsigned char blue = input[color_tid];
		const unsigned char green = input[color_tid + 1];
		const unsigned char red = input[color_tid + 2];

		const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

		output[gray_tid] = static_cast<unsigned char>(gray);
	}
}

void convert_to_gray(const cv::Mat& input, cv::Mat& output)
{
	//Calculate total number of bytes of input and output image
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;

	unsigned char* d_input, * d_output;

	//Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");

	//Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	//Specify a reasonable block size
	const dim3 block(16, 16);

	//Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	//Launch the color conversion kernel
	bgr_to_gray_kernel << <grid, block >> > (d_input, d_output, input.cols, input.rows, input.step, output.step);

	//Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	//Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	//Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}


void processImageCPU() {

}

void processImage()
{

	std::string imagePath = "C:/Users/sninc/Desktop/image.jpg";

	//Read input image from the disk
	cv::Mat input = cv::imread(imagePath);

	if (input.empty())
	{
		std::cout << "Image Not Found!" << std::endl;
		std::cin.get();
		return;
	}

	//Create output image
	cv::Mat output(input.rows, input.cols, CV_8UC1);

	//Call the wrapper function
	convert_to_gray(input, output);

	//Show the input and output
	cv::imshow("Input", input);
	cv::imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	processImageCPU();

}

int main()
{
	cout << "CUDA version:   v" << CUDART_VERSION << endl;

	int devCount;
	cudaGetDeviceCount(&devCount);
	cudaDeviceProp GPU;
	cudaGetDeviceProperties(&GPU, 0);
	cout << GPU.name << ": " << GPU.major << "." << GPU.minor << endl;
	cout << "  Global memory:   " << GPU.totalGlobalMem / (1024 * 1024) << "mb" << endl;
	cout << "  Shared memory:   " << GPU.sharedMemPerBlock / 1024 << "kb" << endl;
	cout << "  Constant memory: " << GPU.totalConstMem / 1024 << "kb" << endl;

	cout << "  Warp size:         " << GPU.warpSize << endl;
	cout << "  Threads per block: " << GPU.maxThreadsPerBlock << endl;
	cout << "  Max block dimensions: [ " << GPU.maxThreadsDim[0] << ", " << GPU.maxThreadsDim[1] << ", " << GPU.maxThreadsDim[2] << " ]" << endl;
	//cout << "  Max grid dimensions:  [ " << GPU.maxGridSize[0] << ", " << GPU.maxGridSize[1] << ", " << GPU.maxGridSize[2] << " ]" << endl;
	cout << endl << endl;

	cout << "1 -matrix multiplication\n2 -memory copy\n3 -float operations\n4 -fibonaci\n5 -image\n9 -exit" << endl;

	initkernel << <1, 1024 >> > ();
	cudaDeviceSynchronize();
	char input;
	while (true)
	{
		cout << ">" << " ";
		scanf(" %c", &input);
		if (input == '9') return 0;

		if (input == '1')
		{
			cout << "\nmatrix size: ";

			int size;
			scanf(" %d", &size);
			Matrix_mul(size);
		}
		if (input == '2')
		{
			cout << "\narray size: ";

			int size;
			scanf(" %d", &size);
			memory_copy(size);
		}
		if (input == '3')
		{
			floatcomputing();
		}
		if (input == '4')
		{
			int size;
			scanf(" %d", &size);
			fibonaci(size);
		}

		if (input == '5')
		{
			processImage();
		}
	}

	return 0;
}
