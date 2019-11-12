
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace std::chrono;
int const OUT = 10000;
int const IN = 100000;
cudaEvent_t startCuda, stopCuda;
float timeCudaMalloc, timeCudaMemcpyh2d, timeKernel, timeCudaMemcpyd2h, CPUMalloc;

__global__ void kernelInit(){}

void cudaMemcpyd2hTimer(void* dst, const void* src, size_t size, cudaMemcpyKind kind)
{
	cudaEventRecord(startCuda);
	cudaMemcpy(dst, src, size, kind);
	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeCudaMemcpyd2h, startCuda, stopCuda);
}

void printTime()
{
	cout << "GPU malloc: " << timeCudaMalloc << " s\n"
		<< "memory copy to GPU: " << timeCudaMemcpyh2d / 1000 << " s\n"
		<< "memory copy from GPU: " << timeCudaMemcpyd2h / 1000 << " s\n"
		<< "kernel: " << timeKernel / 1000 << " s\n";
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


__global__ void floatKernel(float* buf)
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
	float* devData;
	cudaMalloc(&devData, OUT * sizeof(float));

	cudaEventRecord(startCuda);
	floatKernel << <OUT / 1024, 1024 >> > (devData);
	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeKernel, startCuda, stopCuda);

	cudaDeviceSynchronize();
	cudaMemcpyd2hTimer(data, devData, OUT * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(devData);
}


void floatComputing()
{
	high_resolution_clock::time_point start = high_resolution_clock::now();
	floatCPU();
	high_resolution_clock::time_point stop = high_resolution_clock::now();
	duration<double> duration = stop - start;
	cout << "CPU time: " << duration.count() << " s\n";

	start = high_resolution_clock::now();
	floatGPU();
	stop = high_resolution_clock::now();
	duration = stop - start;
	cout << "GPU time: " << duration.count() << " s\n";

	printTime();
}

void arrAddCPU(int size, int* arr1, int* arr2, int* result)
{
	for (int i = 0; i < size; i++)
	{
		result[i] = arr1[i] + arr2[i];
	}
}

__global__ void arrAddKernel(int* arr1, int* arr2, int* result, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		result[tid] = arr1[tid] + arr2[tid];
	}
}


void arrAddGPU(int size, int* arr1, int* arr2, int* result)
{
	int* devArr1, * devArr2, * devResult;


	high_resolution_clock::time_point startMal = high_resolution_clock::now();

	cudaMalloc(&devArr1, sizeof(int) * size);
	cudaMalloc(&devArr2, sizeof(int) * size);
	cudaMalloc(&devResult, sizeof(int) * size);

	cudaDeviceSynchronize();
	high_resolution_clock::time_point stopMal = high_resolution_clock::now();
	duration<double> duration = stopMal - startMal;
	timeCudaMalloc = duration.count();

	int blockSize = 1024;
	int gridSize = (int)ceil((float)size / blockSize);
	
	cudaEventRecord(startCuda);
	
	cudaMemcpy(devArr1, arr1, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(devArr2, arr2, sizeof(int) * size, cudaMemcpyHostToDevice);

	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeCudaMemcpyh2d, startCuda, stopCuda);

	cudaEventRecord(startCuda);
	arrAddKernel << <gridSize, blockSize >> > (devArr1, devArr2, devResult, size);
	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeKernel, startCuda, stopCuda);

	cudaMemcpyd2hTimer(result, devResult, sizeof(int) * size, cudaMemcpyDeviceToHost);

	cudaFree(devArr1);
	cudaFree(devArr2);
	cudaFree(devResult);
}

void memoryCopy(int size)
{
	high_resolution_clock::time_point start = high_resolution_clock::now();

	int* arr1 = (int*)malloc(sizeof(int) * size);
	int* arr2 = (int*)malloc(sizeof(int) * size);
	int* result = (int*)malloc(sizeof(int) * size);

	high_resolution_clock::time_point stop = high_resolution_clock::now();
	duration<double> duration = stop - start;
	CPUMalloc = duration.count();

	for (int i = 0; i < size; i++)
	{
		arr1[i] = i;
		arr2[i] = i;
		result[i] = 0;
	}

	start = high_resolution_clock::now();
	arrAddCPU(size, arr1, arr2, result);
	stop = high_resolution_clock::now();
	duration = stop - start;
	cout << "CPU time: " << duration.count() << " s\n" << "CPU malloc: " << CPUMalloc << " s\n";

	for (int i = 0; i < size; i++) result[i] = 0;

	start = high_resolution_clock::now();
	arrAddGPU(size, arr1, arr2, result);
	stop = high_resolution_clock::now();
	duration = stop - start;
	cout << "GPU time: " << duration.count() << " s\n";

	printTime();

	free(arr1);
	free(arr2);
	free(result);
}

__global__ void matrixKernel(int* matrix1, int* matrix2, int* arrResult, int size)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int result = 0;

	for (int i = 0; i < size; i++)
	{
		result += matrix1[row * size + i] * matrix2[i * size + column];
	}

	arrResult[row * size + column] = result;
}


void matrixCPU(int* matrix1, int* matrix2, int* result, int size)
{
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			for (int k = 0; k < size; k++)
			{
				result[i * size + j] += matrix1[i * size + k] * matrix2[k * size + j];
			}
}
void MatrixMultiplication(int size)
{
	int allocSize = size * size * sizeof(int);
	int* devMatrix1, * devMatrix2, * devResult;
	int* Matrix1 = (int*)malloc(allocSize);
	int* Matrix2 = (int*)malloc(allocSize);
	int* result = (int*)malloc(allocSize);
	int* CPUResult = (int*)malloc(allocSize);


	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			//Matrix1[i * size + j] = rand() % 1500;
			//Matrix2[i * size + j] = rand() % 1500;
			Matrix1[i * size + j] = 5;
			Matrix2[i * size + j] = 5;
			CPUResult[i * size + j] = 0;
		}
	}

	high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	matrixCPU(Matrix1, Matrix2, CPUResult, size);
	high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
	duration<double> duration = stop - start;
	cout << "CPU time: " << duration.count() << " s\n";

	int threadsMax = 16;
	dim3 blockSize(threadsMax, threadsMax);
	dim3 grid_size(size / blockSize.x, size / blockSize.y);

	start = high_resolution_clock::now();

	high_resolution_clock::time_point startMal = high_resolution_clock::now();
	cudaMalloc(&devMatrix2, allocSize);
	cudaMalloc(&devMatrix1, allocSize);
	cudaMalloc(&devResult, allocSize);
	cudaDeviceSynchronize();
	high_resolution_clock::time_point stopMal = high_resolution_clock::now();
	duration = stopMal - startMal;
	timeCudaMalloc = duration.count();


	cudaEventRecord(startCuda);

	cudaMemcpy(devMatrix2, Matrix2, allocSize, cudaMemcpyHostToDevice);
	cudaMemcpy(devMatrix1, Matrix1, allocSize, cudaMemcpyHostToDevice);

	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeCudaMemcpyh2d, startCuda, stopCuda);

	cudaEventRecord(startCuda);
	matrixKernel << <1, 1024 >> > (devMatrix1, devMatrix2, devResult, size);
	
	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeKernel, startCuda, stopCuda);
	cudaDeviceSynchronize();

	cudaMemcpyd2hTimer(result, devResult, allocSize, cudaMemcpyDeviceToHost);
	stop = high_resolution_clock::now();
	duration = stop - start;
	cout << "GPU time: " << duration.count() << " s\n";

	printTime();


	printf("%d %d\n", result[1], CPUResult[1]);

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (CPUResult[size * i + j] != result[size * i + j])
			{
				printf("Chybne vypocitana matica!\n");
				bool exit = true;
				break;
			}
		}
		if (exit) break;
	}

	free(Matrix2);
	free(Matrix1);
	free(result);

	cudaFree(devMatrix2);
	cudaFree(devMatrix1);
	cudaFree(devResult);
}

__global__ void fibonaciKernel(int size)
{
	for (int i = 0; 1000000 < 0; i++)
	{

		int a = 0, b = 1, next = 0;
		next = a + b;
		while (next <= size)
		{
			a = b;
			b = next;
			next = a + b;
		}

	}


}

void fibonaciGPU(int size)
{
	fibonaciKernel << <1, 1 >> > (size);

}

void fibonaciCPU(int size)
{
	for (int i = 0; i < 1000000; i++)
	{

		int a = 0, b = 1, next = 0;
		next = a + b;
		while (next <= size)
		{
			a = b;
			b = next;
			next = a + b;
		}

	}
}

void fibonaci(int size)
{
	high_resolution_clock::time_point start = high_resolution_clock::now();
	fibonaciCPU(size);
	high_resolution_clock::time_point stop = high_resolution_clock::now();
	duration<double> duration = stop - start;
	cout << "CPU time: " << duration.count() << " s\n";

	start = high_resolution_clock::now();
	fibonaciGPU(size);
	stop = high_resolution_clock::now();
	duration = stop - start;
	cout << "GPU time: " << duration.count() << " s\n";
}

__global__ void blackWhiteKernel(unsigned char* input, unsigned char* output, int inputRowLength, int outputRowLength, int inputColumns, int inputRows)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < 100; i++)
	{
		if ((row < inputRows) && (column < inputColumns))
		{
			int tidInput = row * inputRowLength + (column * 3);
			int tidOutput = row * outputRowLength + column;

			float blackWhite = (input[tidInput + 2] + input[tidInput + 1] + input[tidInput]) / 3;	//RGB

			output[tidOutput] = static_cast<unsigned char>(blackWhite);
		}
	}
}

void ImageGPU(unsigned char* input, unsigned char* output, int inputRowLength, int outputRowLength, int inputColumns, int inputRows, int outputWidth, cudaEvent_t start, cudaEvent_t stop)
{
	unsigned char* devInput, * devOutput;
	
	high_resolution_clock::time_point startMal = high_resolution_clock::now();
	
	cudaMalloc<unsigned char>(&devInput, inputRowLength * inputRows);
	cudaMalloc<unsigned char>(&devOutput, outputRowLength * outputWidth);

	cudaDeviceSynchronize();
	high_resolution_clock::time_point stopMal = high_resolution_clock::now();
	duration<double> duration = stopMal - startMal;
	timeCudaMalloc = duration.count();

	cudaEventRecord(startCuda);

	cudaMemcpy(devInput, input, inputRowLength * inputRows, cudaMemcpyHostToDevice);
	
	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeCudaMemcpyh2d, startCuda, stopCuda);

	//Specify a reasonable block size
	const dim3 block(16, 16);

	//Calculate grid size to cover the whole image
	const dim3 grid((inputColumns + block.x - 1) / block.x, (inputRows + block.y - 1) / block.y);
	
	cudaEventRecord(startCuda);

	blackWhiteKernel<<<grid, block >>> (devInput, devOutput, inputRowLength, outputRowLength, inputColumns, inputRows);

	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeKernel, startCuda, stopCuda);


	cudaMemcpyd2hTimer(output, devOutput, outputRowLength * outputWidth, cudaMemcpyDeviceToHost);

	cudaFree(devInput);
	cudaFree(devOutput);	
}


void ImageCPU(unsigned char* input, unsigned char* output, int inputRowLength, int outputRowLength, int inputColumns, int inputRows)
{
	for(int k = 0 ; k < 100 ; k++)
		for(int i = 0; i < inputRows; i++)
			for (int j = 0; j < inputColumns; j++)
			{
				int inputPosition = i* inputRowLength + (3 * j);
				int outputPosition = i * outputRowLength + j;

				float blackWhite = (input[inputPosition + 2] + input[inputPosition + 1] + input[inputPosition])/3;  //RGB

				output[outputPosition] = static_cast<unsigned char>(blackWhite);
			}
}

void processImage()
{
	cv::Mat input = cv::imread("image.jpg");
	cudaEvent_t startCuda, stopCuda;
	cudaEventCreate(&startCuda);
	cudaEventCreate(&stopCuda);

	if (input.empty())
	{
		cout << "Obrazok sa musi volat image.jpg a byt v rovnakom priecinku ako exe subor" << std::endl;
		return;
	}

	cv::Mat output(input.rows, input.cols, CV_8UC1);

	high_resolution_clock::time_point start = high_resolution_clock::now();
	ImageCPU(input.ptr(), output.ptr(), input.step, output.step, input.cols, input.rows);
	high_resolution_clock::time_point stop = high_resolution_clock::now();
	duration<double> duration = stop - start;
	cout << "CPU time: " << duration.count() << " s\n";

	start = high_resolution_clock::now();
	ImageGPU(input.ptr(), output.ptr(), input.step, output.step, input.cols, input.rows, output.rows, startCuda, stopCuda);
	stop = high_resolution_clock::now();
	duration = stop - start;
	cout << "GPU time: " << duration.count() << " s\n";

	printTime();

	cv::imshow("original", input);
	cv::imshow("processed", output);

	cv::waitKey();

}

int main()
{
	cudaEventCreate(&startCuda);
	cudaEventCreate(&stopCuda);
	timeCudaMemcpyh2d = 0;
	timeCudaMemcpyd2h = 0;
	timeCudaMalloc = 0;
	timeKernel = 0;

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

	//kernelInit << <1, 1024 >> > ();
	//cudaDeviceSynchronize();
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
			MatrixMultiplication(size);
		}
		if (input == '2')
		{
			cout << "\narray size: ";

			int size;
			scanf(" %d", &size);
			memoryCopy(size);
		}
		if (input == '3')
		{
			floatComputing();
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
