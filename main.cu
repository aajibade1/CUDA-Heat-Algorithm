#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <cmath>

using namespace std;

//FDM Calculate function -GPU version
__global__ void transferHeat (double *gen, double *hold, int plateSize)
{

    int threadIDX = (blockIdx.x * blockDim.x) + threadIdx.x; // thread id x
    int threadIDY = (blockIdx.y * blockDim.y) + threadIdx.y; // thread id y

    if (threadIDX > 0 && threadIDX < plateSize - 1 && threadIDY > 0 && threadIDY < plateSize - 1)
    {
        gen[threadIDX + (plateSize * threadIDY)] = 0.25 * (hold[(threadIDX - 1) + (plateSize * threadIDY)]
                                                           + hold[(threadIDX + 1) + (plateSize * threadIDY)]
                                                           + hold[threadIDX + (plateSize * (threadIDY + 1))] +
                                                           hold[threadIDX + (plateSize * (threadIDY - 1))]);
    }
}

//copy generated array to holding array function - GPU version
__global__ void copyOver(double *gen, double *hold, int plateSize)
{
    int threadIDX = (blockIdx.x * blockDim.x) + threadIdx.x;
    int threadIDY = (blockIdx.y * blockDim.y)+ threadIdx.y;

    hold[threadIDX + (plateSize * threadIDY)] = gen[threadIDX + (plateSize * threadIDY)];
}

//verify input function
bool is_numeric(const string& strIn, unsigned int& nInputNumber)
{
    //this function checks if the passed input number is a valid ulong value
    //this includes checking for negatives, strings and floating point numbers
    bool bRC = all_of(strIn.begin(), strIn.end(),[](unsigned char c)
                      {
                          return ::isdigit(c);
                      }
    );
    if (bRC)
    {
        nInputNumber = stoi(strIn);//converts string to unsigned long number
        return true;
    }
    else
    {
        return false;
    }
}

int main(int argc, char **argv)
{

    int opt; //for input parameters
    unsigned int N; //N splits for inside data i think
    unsigned int I; //iterations coming soon from a terminal near you
    opterr = 0;

    if (argc > 1)
    {
        if (argv[optind] == NULL || argv[optind + 2] == NULL)
        {
            cout << "Invalid parameter, please check your values." << endl;
            return 0;
        }

        while ((opt = getopt(argc, argv, "N:I:")) != -1)
        {
            bool bIsValid;
            switch (opt)
            {
                case 'N':
                    bIsValid = is_numeric(optarg, N);
                    if (!bIsValid)
                    {
                        cout << "Invalid parameter, please check your values." << endl;
                        return 0;
                    }
                    break;
                case 'I':
                    bIsValid = is_numeric(optarg, I);
                    if (!bIsValid)
                    {
                        cout << "Invalid parameter, please check your values." << endl;
                        return 0;
                    }
                    break;
                case '?':
                    cout << "Invalid parameter, please check your values." << endl;
                    return 0;
                    break;
            }
        }

    }
    else
    {
        cout << "Invalid parameter, please check your values." << endl;
        return 0;
    }

    //cout << "Input N: " << N << endl;
    //cout << "Input I: " << I << endl;
    int plateSize = N + 2; //grid width/height
    int size = sizeof(double) * plateSize * plateSize; //size of an array N*N enough to hold double types
    double *gen = (double *) malloc(size); //host array of gen allocate
    double *hold = (double *) malloc(size); //host array of hold allocate

    //initialize host arrays
    for (int i = 0; i < plateSize; i++)
    {
        for (int j = 0; j < plateSize; j++)
        {
            if ((j > 0.3 * (N + 2 - 1) && j < 0.7 * (N + 2 - 1)) && i == 0) {
                gen[i * plateSize + j] = 100.0;
                hold[i * plateSize + j]  = 100.0;
            }
            else
            {
                gen[i * plateSize + j] = 20.0;
                hold[i * plateSize + j]  = 20.0;
            }
        }
    }

    //allocate device memory
    double *d_gen;
    double *d_hold;

    cudaMalloc((void**)&d_gen, size);
    cudaMalloc((void**)&d_hold, size);

    //start timing event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Transfer data from host to device memory
    cudaMemcpy(d_gen, gen, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hold, hold, size, cudaMemcpyHostToDevice);

    //get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int threadsPerBlock=pow(prop.maxThreadsPerBlock, 0.5); //threads in each direction
    dim3 dimBlock(threadsPerBlock, threadsPerBlock );

    // calculate number of blocks along X and Y
    dim3 dimGrid( ceil(double(plateSize)/double(dimBlock.x)), ceil(double(plateSize)/double (dimBlock.y)));

    //cout << "block size: " << ThreadsPerBlock <<  endl;

    cudaEventRecord(start); //start the time

    //run kernel functions iteratively
    for (auto iteration = 0; iteration < I; iteration++)
    {
        transferHeat<<<dimGrid, dimBlock>>>(d_gen, d_hold, plateSize);
        copyOver<<<dimGrid, dimBlock>>>(d_gen, d_hold, plateSize);
    }
    cudaEventRecord(stop); // stop time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << setprecision(2) << fixed << milliseconds << endl;

    // Transfer data back to host memory
    cudaMemcpy(hold, d_hold, size, cudaMemcpyDeviceToHost);
    //cudaThreadSynchronize();


    // Deallocate device memory
    cudaFree(d_gen);
    cudaFree(d_hold);

    // Deallocate host memory
    delete(gen);
    delete(hold);
    return 0;
}
