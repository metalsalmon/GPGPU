
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <fstream>

using namespace std;
using namespace std::chrono;
cudaEvent_t startCuda, stopCuda;
float timeCudaMalloc, timeCudaMemcpyh2d, timeKernel, timeCudaMemcpyd2h, CPUMalloc;
stringstream ss;

struct Settings {
	int	test1repeat, test1Base, test1Increment, test1blocksize, test1gridsize;
	int	test2repeat, test2Base, test2Increment, test2blocksize;
	int	test3repeat, test3Base, test3Increment, test3blocksize;
	int	test4repeat, test4Base, test4Increment, test4blocksize;
	int	test5repeat, test5Base, test5Increment, test5blocksize;
};

__global__ void kernelInit(){}

void resetTime()
{
	timeCudaMalloc = timeCudaMemcpyh2d = timeKernel = timeCudaMemcpyd2h = CPUMalloc = 0;
}

void readConfig(Settings& settings)
{
	ifstream fileConfig("config.txt");
	string line;
	while (getline(fileConfig, line)) {
		istringstream readline(line.substr(line.find(":") + 1));
		if (line.find("base1") != -1) readline >> settings.test1Base;
		else if (line.find("base2") != -1) readline >> settings.test2Base;
		else if (line.find("base3") != -1) readline >> settings.test3Base;
		else if (line.find("base4") != -1) readline >> settings.test4Base;
		else if (line.find("base5") != -1) readline >> settings.test5Base;
		else if (line.find("increment1") != -1) readline >> settings.test1Increment;
		else if (line.find("increment2") != -1) readline >> settings.test2Increment;
		else if (line.find("increment3") != -1) readline >> settings.test3Increment;
		else if (line.find("increment4") != -1) readline >> settings.test4Increment;
		else if (line.find("blocksize1") != -1) readline >> settings.test1blocksize;
		else if (line.find("gridsize1") != -1) readline >> settings.test1gridsize;
		else if (line.find("increment5") != -1) readline >> settings.test5Increment;
		else if (line.find("repeat1") != -1) readline >> settings.test1repeat;
		else if (line.find("repeat2") != -1) readline >> settings.test2repeat;
		else if (line.find("repeat3") != -1) readline >> settings.test3repeat;
		else if (line.find("repeat4") != -1) readline >> settings.test4repeat;
		else if (line.find("repeat5") != -1) readline >> settings.test5repeat;
		else if (line.find("blocksize2") != -1) readline >> settings.test2blocksize;
		else if (line.find("blocksize3") != -1) readline >> settings.test3blocksize;
		else if (line.find("blocksize4") != -1) readline >> settings.test4blocksize;
		else if (line.find("blocksize5") != -1) readline >> settings.test5blocksize;
	}
	fileConfig.close();
}

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
	cout <<"GPU malloc: " << timeCudaMalloc << " s\n"
		<< "memory copy to GPU: " << timeCudaMemcpyh2d / 1000 << " s\n"
		<< "memory copy from GPU: " << timeCudaMemcpyd2h / 1000 << " s\n"
		<< "kernel execution time: " << timeKernel / 1000 << " s\n\n";

	ss  << "GPU malloc: " << timeCudaMalloc << " s\n"
		<< "memory copy to GPU: " << timeCudaMemcpyh2d / 1000 << " s\n"
		<< "memory copy from GPU: " << timeCudaMemcpyd2h / 1000 << " s\n"
		<< "kernel execution time: " << timeKernel / 1000 << " s\n\n";


}

void CPUGPUTime(string text)
{
	cout << text;
	ss << text;
}

void printTestNumber(string text)
{
	cout << text;
	ss << text;
}

void floatCPU(int in, int out)
{
	float* data = (float*)malloc(sizeof(float) * out);

	for (int i = 1; i < out; i++)
	{
		data[i] = 1.0f * i / out;
		for (int j = 1; j < in; j++)
		{
			data[i] = data[i] / in * data[i] / in - 0.50f;
		}
	}
}

__global__ void floatKernel(float* arr, int in, int out)

{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	arr[i] = 1.0f * i / out;
	for (int j = 0; j < in; j++)
	{
		arr[i] = arr[i] / in * arr[i] / in - 0.50f;
	}
}

void floatGPU(int in, int out, int blockSize)
{
	float* data = (float*)malloc(sizeof(float) * out);
	float* devData;

	high_resolution_clock::time_point startMal = high_resolution_clock::now();

	cudaMalloc(&devData, out * sizeof(float));


	cudaDeviceSynchronize();
	high_resolution_clock::time_point stopMal = high_resolution_clock::now();
	duration<double> duration = stopMal - startMal;
	timeCudaMalloc = duration.count();

	cudaEventRecord(startCuda);
	floatKernel <<<out / blockSize, blockSize >> > (devData, in, out);
	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeKernel, startCuda, stopCuda);

	cudaDeviceSynchronize();
	cudaMemcpyd2hTimer(data, devData, out * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(devData);
}


void floatComputing(int in, int out, int blockSize)
{
	high_resolution_clock::time_point start = high_resolution_clock::now();
	floatCPU(in, out);
	high_resolution_clock::time_point stop = high_resolution_clock::now();
	duration<double> duration = stop - start;
	CPUGPUTime("CPU time: " + to_string(duration.count()) + " s\n");

	start = high_resolution_clock::now();
	floatGPU(in, out, blockSize);
	stop = high_resolution_clock::now();
	duration = stop - start;
	CPUGPUTime("GPU time: " + to_string(duration.count()) + " s\n");

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


void arrAddGPU(int size, int* arr1, int* arr2, int* result, int blockSize)
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
	
	cudaEventRecord(startCuda);
	
	cudaMemcpy(devArr1, arr1, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(devArr2, arr2, sizeof(int) * size, cudaMemcpyHostToDevice);

	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeCudaMemcpyh2d, startCuda, stopCuda);

	cudaEventRecord(startCuda);
	arrAddKernel << < (int)ceil((float)size) / blockSize, blockSize >> > (devArr1, devArr2, devResult, size);
	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeKernel, startCuda, stopCuda);

	cudaMemcpyd2hTimer(result, devResult, sizeof(int) * size, cudaMemcpyDeviceToHost);

	cudaFree(devArr1);
	cudaFree(devArr2);
	cudaFree(devResult);
}

void memoryCopy(int size, int blockSize)
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
	CPUGPUTime("CPU time: " + to_string(duration.count()) + " s\n" + "CPU malloc: " +to_string(CPUMalloc) + " s\n");

	for (int i = 0; i < size; i++) result[i] = 0;

	start = high_resolution_clock::now();
	arrAddGPU(size, arr1, arr2, result, blockSize);
	stop = high_resolution_clock::now();
	duration = stop - start;
	CPUGPUTime("GPU time: " + to_string(duration.count()) + " s\n");

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
void MatrixMultiplication(int size, int blocksize, int gridsize)
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
	CPUGPUTime("CPU time: " + to_string(duration.count()) + " s\n");

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
	matrixKernel << <gridsize, blocksize >> > (devMatrix1, devMatrix2, devResult, size);
	
	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeKernel, startCuda, stopCuda);
	cudaDeviceSynchronize();

	cudaMemcpyd2hTimer(result, devResult, allocSize, cudaMemcpyDeviceToHost);
	stop = high_resolution_clock::now();
	duration = stop - start;
	CPUGPUTime("GPU time: " + to_string(duration.count()) + " s\n");

	printTime();


	//printf("%d %d\n", result[1], CPUResult[1]);

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (CPUResult[size * i + j] != result[size * i + j])
			{
				cout << "Chybne vypocitana matica!\n";
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
	for (int i = 0; i < size; i++)
	{

		int a = 0, b = 1, next = 0;
		next = a + b;
		while (next <= 100000000)
		{
			a = b;
			b = next;
			next = a + b;

		}

	}

}

void fibonaciGPU(int size, int blockSize)
{
	cudaEventRecord(startCuda);
	fibonaciKernel <<<1, blockSize >> > (size);


	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeKernel, startCuda, stopCuda);
	cudaDeviceSynchronize();

}

void fibonaciCPU(int size)
{
	for (int i = 0; i < 100000000; i++)
	{
			int a = 0, b = 1, next = 0;
			next = a + b;
			while (next <= 100000000)
			{
				a = b;
				b = next;
				next = a + b;
			}

	}
}

void fibonaci(int size, int blockSize)
{
	high_resolution_clock::time_point start = high_resolution_clock::now();
	fibonaciCPU(size);
	high_resolution_clock::time_point stop = high_resolution_clock::now();
	duration<double> duration = stop - start;
	CPUGPUTime("CPU time: " + to_string(duration.count()) + " s\n");

	start = high_resolution_clock::now();
	fibonaciGPU(size, blockSize);
	stop = high_resolution_clock::now();
	duration = stop - start;
	CPUGPUTime("GPU time: " + to_string(duration.count()) + " s\n");

	printTime();
}

__global__ void blackWhiteKernel(unsigned char* input, unsigned char* output, int inputRowLength, int outputRowLength, int inputColumns, int inputRows,int size)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int tidInput, tidOutput;
	float blackWhite;
	for (int i = 0; i < size; i++)
	{
		if ((row < inputRows) && (column < inputColumns))
		{
			tidInput = row * inputRowLength + (column * 3);
			tidOutput = row * outputRowLength + column;

			blackWhite = (input[tidInput + 2] + input[tidInput + 1] + input[tidInput]) / 3;	//RGB

			output[tidOutput] = static_cast<unsigned char>(blackWhite);
		}
	}
}

void ImageGPU(unsigned char* input, unsigned char* output, int inputRowLength, int outputRowLength, int inputColumns, int inputRows, int outputWidth, int size, int blockSize)
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

	dim3 block(blockSize, blockSize);
	dim3 grid((ceil((int)(float)inputColumns + block.x)) / block.x, ((ceil((int)(float)inputRows + block.y)) / block.y));
	cudaEventRecord(startCuda);

	blackWhiteKernel<<<grid, block >>> (devInput, devOutput, inputRowLength, outputRowLength, inputColumns, inputRows, size);

	cudaEventRecord(stopCuda);
	cudaEventSynchronize(stopCuda);
	cudaEventElapsedTime(&timeKernel, startCuda, stopCuda);

	cudaMemcpyd2hTimer(output, devOutput, outputRowLength * outputWidth, cudaMemcpyDeviceToHost);

	cudaFree(devInput);
	cudaFree(devOutput);	
}


void ImageCPU(unsigned char* input, unsigned char* output, int inputRowLength, int outputRowLength, int inputColumns, int inputRows, int size)
{
	for(int k = 0 ; k < size ; k++)
		for(int i = 0; i < inputRows; i++)
			for (int j = 0; j < inputColumns; j++)
			{
				int inputPosition = i* inputRowLength + (3 * j);
				int outputPosition = i * outputRowLength + j;

				float blackWhite = (input[inputPosition + 2] + input[inputPosition + 1] + input[inputPosition])/3;  //RGB

				output[outputPosition] = static_cast<unsigned char>(blackWhite);
			}
}

void processImage(int size, int blockSize)
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
	ImageCPU(input.ptr(), output.ptr(), input.step, output.step, input.cols, input.rows, size);
	high_resolution_clock::time_point stop = high_resolution_clock::now();
	duration<double> duration = stop - start;
	CPUGPUTime("CPU time: " + to_string(duration.count()) + " s\n");

	start = high_resolution_clock::now();
	ImageGPU(input.ptr(), output.ptr(), input.step, output.step, input.cols, input.rows, output.rows, size, blockSize);
	stop = high_resolution_clock::now();
	duration = stop - start;
	CPUGPUTime("GPU time: " + to_string(duration.count()) + " s\n");

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

	cout << "CUDA version:   v" << CUDART_VERSION << "\n";

	int devCount;
	cudaGetDeviceCount(&devCount);
	if(devCount == 0)
	{
		cout << "0 cuda devices\n";
		return 0;
	}
	cudaDeviceProp GPU;
	cudaGetDeviceProperties(&GPU, 0);
	cout << GPU.name << ": " << GPU.major << "." << GPU.minor << "\n";
	cout << "  Global memory:   " << GPU.totalGlobalMem / (1024 * 1024) << "mb" << "\n";
	cout << "  Shared memory:   " << GPU.sharedMemPerBlock / 1024 << "kb" << "\n";

	cout << "  Warp size:         " << GPU.warpSize << "\n";
	cout << "  Threads per block: " << GPU.maxThreadsPerBlock << "\n";
	cout << "  Max block dimensions: [ " << GPU.maxThreadsDim[0] << ", " << GPU.maxThreadsDim[1] << ", " << GPU.maxThreadsDim[2] << " ]" << "\n\n\n\n";

	cout << "Test 1 -matrix multiplication\nTest 2 -memory copy\nTest 3 -aritmetic operations\nTest 4 -fibonaci\nTest 5 -image processing\n" << "\n\n";

	kernelInit <<<1, 1024 >>> ();
	
	cudaDeviceSynchronize();

	Settings settings;
	readConfig(settings);

	char input;

	for (int i = 1; i < 6; i++)
	{
		printTestNumber("****************************************************\n\nTest "+ to_string(i) + "\n");
		for (int j = 1; j < 31; j++)
		{

			if (i == 1)
			{
				if (j > settings.test1repeat) break;
				printTestNumber(to_string(j) +" : matrix size: "+to_string(settings.test1Base) + "\n");
				MatrixMultiplication(settings.test1Base, settings.test1blocksize, settings.test1gridsize);
				settings.test1Base += settings.test1Increment;
				resetTime();
			}

			else if (i == 2)
			{
				if (j > settings.test2repeat) break;
				printTestNumber(to_string(j) + " : array of int length : " + to_string(settings.test2Base) + "\n");
				memoryCopy(settings.test2Base, settings.test2blocksize);
				settings.test2Base *= settings.test2Increment;
				resetTime();
			}

			else if (i == 3)
			{
				if (j > settings.test3repeat) break;
				printTestNumber(to_string(j) + " number of operations: " + to_string(10000 * settings.test3Base) + "\n");
				floatComputing(settings.test3Base, 10000, settings.test3blocksize);
				settings.test3Base *= settings.test3Increment;

				resetTime();
			}

			else if (i == 4)
			{
				if (j > settings.test4repeat) break;
				printTestNumber(to_string(j) +" cycles: " + to_string(settings.test4Base)+"\n");
				fibonaci(settings.test4Base, settings.test4blocksize);
				settings.test4Base *= settings.test4Increment;
				resetTime();
			}

			else if (i == 5)
			{
				if (j > settings.test5repeat) break;
				printTestNumber(to_string(j) + " number of images " + to_string(settings.test5Base) + "\n");
				processImage(settings.test5Base, settings.test5blocksize);
				settings.test5Base *= settings.test5Increment;
				resetTime();
			}
		}
	}
	

	ofstream resultsFile;
	resultsFile.open("results.txt");
	resultsFile << ss.rdbuf();
	resultsFile.close();
/*
	while (true)
	{
		cout << ">" << " ";
		scanf(" %c", &input);
		if (input == '9')
		{
			ofstream resultsFile;
			resultsFile.open("results.txt");
			resultsFile << ss.rdbuf();
			resultsFile.close();
			return 0;
		}

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
			floatComputing(10000, 10000);
		}
		if (input == '4')
		{
			int size;
			scanf(" %d", &size);
			fibonaci(size);
		}

		if (input == '5')
		{
			processImage(100);
		}
	}
*/
	cin.get();

	return 0;
}
