
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
int const OUT = 10000;
int const IN = 100000;

__global__ void initkernel(){}

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
	auto start = chrono::high_resolution_clock::now();
	floatGPU();
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> duration = finish - start;
	std::cout << "GPU time: " << duration.count() << " s\n";

	start = chrono::high_resolution_clock::now();
	floatCPU();
	finish = chrono::high_resolution_clock::now();
	duration = finish - start;
	cout << "CPU time: " << duration.count() << " s\n";
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
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	int* input1 = (int*)malloc(sizeof(int) * size);
	int* input2 = (int*)malloc(sizeof(int) * size);
	int* result = (int*)malloc(sizeof(int) * size);

	int* dev_input1, * dev_input2, * dev_result;

	cudaEventRecord(start);

	cudaMalloc(&dev_input1, sizeof(int) * size);
	cudaMalloc(&dev_input2, sizeof(int) * size);
	cudaMalloc(&dev_result, sizeof(int) * size);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "malloc: " << milliseconds/1000 << " s\n";

	for (int i = 0; i < size; i++)
	{
		input1[i] = i;
		input2[i] = i;
		result[i] = 0;
	}

	int block_size = 1024;
	int grid_size = (int)ceil((float)size / block_size);

	
	cudaEventRecord(start);
	cudaMemcpy(dev_input1, input1, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_input2, input2, sizeof(int) * size, cudaMemcpyHostToDevice);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	 milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "memcpy: " << milliseconds/1000 << " s\n";

	cudaEventRecord(start);
	GPUadd << <grid_size, block_size >> > (dev_input1, dev_input2, dev_result, size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "kernel: " << milliseconds/1000 << " s\n";
	

	free(input1);
	free(input2);
	free(result);

	cudaFree(dev_input1);
	cudaFree(dev_input2);
	cudaFree(dev_result);
}

void memory_copy(int size)
{

	auto start = chrono::high_resolution_clock::now();
	RunGPU(size);
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> duration = finish - start;
	cout << "GPU time: " << duration.count() << " s\n";


	start = chrono::high_resolution_clock::now();
	RunCPU(size);
	finish = chrono::high_resolution_clock::now();
	duration = finish - start;
	cout << "CPU time: " << duration.count() << " s\n";
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

	auto start = chrono::high_resolution_clock::now();
	cudaMalloc(&dev_matrix1atrix2, bytes);
	cudaMalloc(&dev_matrix1, bytes);
	cudaMalloc(&dev_result, bytes);
	cudaMemcpy(dev_matrix1atrix2, host_n, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matrix1, host_m, bytes, cudaMemcpyHostToDevice);
	matrix_kernel << <1, 1024 >> > (dev_matrix1, dev_matrix1atrix2, dev_result, size);
	cudaDeviceSynchronize();
	cudaMemcpy(host_result, dev_result, bytes, cudaMemcpyDeviceToHost);
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> duration = finish - start;
	cout << "GPU time: " << duration.count() << " s\n";

	start = chrono::high_resolution_clock::now();
	matrix_cpu(host_m, host_n, cpu_result, size);
	finish = chrono::high_resolution_clock::now();
	duration = finish - start;
	cout << "CPU time: " << duration.count() << " s\n";


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
	auto start = chrono::high_resolution_clock::now();
	fibonaciCPU(size);
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> duration = finish - start;
	cout << "CPU time: " << duration.count() << " s\n";

	start = chrono::high_resolution_clock::now();
	fibonaciGPU(size);
	finish = chrono::high_resolution_clock::now();
	duration = finish - start;
	cout << "GPU time: " << duration.count() << " s\n";


}

__global__ void black_white_kernel(unsigned char* input, unsigned char* output, int input_row_length, int output_row_length, int input_columns, int input_rows)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < 100; i++)
	{
		if ((row < input_rows) && (column < input_columns))
		{
			int tid_input = row * input_row_length + (column * 3);
			int tid_output = row * output_row_length + column;

			float black_white = (input[tid_input + 2] + input[tid_input + 1] + input[tid_input]) / 3;	//RGB

			output[tid_output] = static_cast<unsigned char>(black_white);
		}
	}
}

void ImageGPU(unsigned char* input, unsigned char* output, int input_row_length, int output_row_length, int input_columns, int input_rows, int output_width, cudaEvent_t start, cudaEvent_t stop)
{
	unsigned char* dev_input, * dev_output;
	
	cudaEventRecord(start);
	cudaMalloc<unsigned char>(&dev_input, input_row_length * input_rows);
	cudaMalloc<unsigned char>(&dev_output, output_row_length * output_width);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "malloc: " << milliseconds << " s\n";
	cudaMemcpy(dev_input, input, input_row_length * input_rows, cudaMemcpyHostToDevice);

	//Specify a reasonable block size
	const dim3 block(16, 16);

	//Calculate grid size to cover the whole image
	const dim3 grid((input_columns + block.x - 1) / block.x, (input_rows + block.y - 1) / block.y);

	//auto start = chrono::high_resolution_clock::now();
	black_white_kernel<<<grid, block >>> (dev_input, dev_output, input_row_length, output_row_length, input_columns, input_rows);
	cudaDeviceSynchronize();
	//auto finish = chrono::high_resolution_clock::now();
	//chrono::duration<double> duration = finish - start;
	//cout << "GPU kernel: " << duration.count() << " s\n";
	cudaEventRecord(start);
	cudaMemcpy(output, dev_output, output_row_length * output_width, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;

	cudaEventElapsedTime(&milliseconds, start, stop);
	
	cout << "omg: " << milliseconds << " s\n";
	cudaFree(dev_input);
	cudaFree(dev_output);
	
	
}


void ImageCPU(unsigned char* input, unsigned char* output, int input_row_length, int output_row_length, int input_columns, int input_rows)
{
	for(int k = 0 ; k < 100 ; k++)
		for(int i = 0; i < input_rows; i++)
			for (int j = 0; j < input_columns; j++)
			{
				int input_position = i* input_row_length + (3 * j);
				int output_position = i * output_row_length + j;

				float black_white = (input[input_position + 2] + input[input_position + 1] + input[input_position])/3;  //RGB

				output[output_position] = static_cast<unsigned char>(black_white);

			}
}

void processImage()
{
	cv::Mat input = cv::imread("image.jpg");
	cudaEvent_t start_cuda, stop_cuda;
	cudaEventCreate(&start_cuda);
	cudaEventCreate(&stop_cuda);

	if (input.empty())
	{
		cout << "Obrazok sa musi volat image.jpg a byt v rovnakom priecinku ako exe subor" << std::endl;
		return;
	}

	cv::Mat output(input.rows, input.cols, CV_8UC1);

	auto start = chrono::high_resolution_clock::now();
	ImageCPU(input.ptr(), output.ptr(), input.step, output.step, input.cols, input.rows);
	auto finish = chrono::high_resolution_clock::now();
	chrono::duration<double> duration = finish - start;
	cout << "CPU time: " << duration.count() << " s\n";

	start = chrono::high_resolution_clock::now();
	ImageGPU(input.ptr(), output.ptr(), input.step, output.step, input.cols, input.rows, output.rows, start_cuda, stop_cuda);
	finish = chrono::high_resolution_clock::now();
	duration = finish - start;
	cout << "GPU time: " << duration.count() << " s\n";


	cv::imshow("original", input);
	cv::imshow("processed", output);


	cv::waitKey();

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
