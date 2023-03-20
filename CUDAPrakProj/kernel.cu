
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <time.h>

#define MATRIXLEN 10
#define BLOCKSIZE 10

struct Matrix
{
    int dimension = MATRIXLEN;
    int* elements;
};

cudaError_t MultiMatrixesCUDA(int** mat1, int** mat2, int*** res);

void GenerateMatrix(int*** matrix);
void OutputMatrix(int** matrix);
void Convert1D(Matrix* dst, int** src);
int** Convert2D(Matrix src);
int** MultiMatrixes(int** matrix1, int** matrix2);

__global__ void MultiKernel(Matrix dA, Matrix dB, Matrix dC)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    dC.elements[row * MATRIXLEN + col] = 0;
    for (int inner = 0; inner < MATRIXLEN; ++inner) {
        dC.elements[row * MATRIXLEN + col] += dA.elements[row * MATRIXLEN + inner] * dB.elements[inner * MATRIXLEN + col];
    }
}

int main()
{
    int** matrix1 = nullptr;
    int** matrix2 = nullptr;
    int** matrixRes = nullptr;

    srand(time(0));

    GenerateMatrix(&matrix1);
    GenerateMatrix(&matrix2);
    GenerateMatrix(&matrixRes);

    OutputMatrix(matrix1);
    OutputMatrix(matrix2);

    // Add vectors in parallel.
    cudaError_t cudaStatus = MultiMatrixesCUDA(matrix1, matrix2, &matrixRes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MultiMatrixesCUDA failed!");
        return 1;
    }
    printf("\n-------------CUDA Result-------------\n");
    OutputMatrix(matrixRes);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    int** x = MultiMatrixes(matrix1, matrix2);
    printf("\n-------------CPU Result-------------\n");
    OutputMatrix(x);

    free(matrix1);
    free(matrix2);
    free(matrixRes);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t MultiMatrixesCUDA(int** mat1, int** mat2, int*** res)
{
    Matrix A, B, C, dA, dB, dC;
    size_t size = MATRIXLEN * MATRIXLEN * sizeof(int);
    cudaError_t cudaStatus;

    A.elements = (int*)malloc(MATRIXLEN * MATRIXLEN * sizeof(int));
    B.elements = (int*)malloc(MATRIXLEN * MATRIXLEN * sizeof(int));
    C.elements = (int*)malloc(MATRIXLEN * MATRIXLEN * sizeof(int));
    Convert1D(&A, mat1);
    Convert1D(&B, mat2);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //// Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc(&dA.elements, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&dB.elements, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&dC.elements, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dA.elements, A.elements, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dB.elements, B.elements, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 dimGrid(MATRIXLEN / BLOCKSIZE, MATRIXLEN / BLOCKSIZE);

    // Launch a kernel on the GPU with one thread for each element.
    MultiKernel<<<dimGrid, dimBlock>>>(dA, dB, dC);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MultiMatrixesCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching MultiMatrixesCUDA!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(C.elements, dC.elements, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy2D failed!");
        goto Error;
    }
    
    int** tmp = Convert2D(C);
    free(*res);
    *res = tmp;

Error:
    cudaFree(dA.elements);
    cudaFree(dB.elements);
    cudaFree(dC.elements);
    free(A.elements);
    free(B.elements);
    free(C.elements);
    
    return cudaStatus;
}

void GenerateMatrix(int*** matrix) {
    int** x = nullptr;
    x = (int**)malloc(MATRIXLEN * sizeof(int*));
    for (int i = 0; i < MATRIXLEN; ++i) {
        x[i] = (int*)malloc(MATRIXLEN * sizeof(int));
        for (int j = 0; j < MATRIXLEN; ++j) {
            x[i][j] = rand() % 100;
        }
    }
    *matrix = x;
}

void OutputMatrix(int** matrix) {
    printf("\n");
    for (int i = 0; i < MATRIXLEN; ++i) {
        for (int j = 0; j < MATRIXLEN; ++j) {
            printf("%7d", matrix[i][j]);
        }
        printf("\n");
    }
}

int** MultiMatrixes(int** matrix1, int** matrix2) {
    int** x = nullptr;
    GenerateMatrix(&x);
    for (int row = 0; row < MATRIXLEN; ++row) {
        for (int col = 0; col < MATRIXLEN; ++col) {
            // Multiply the row of A by the column of B to get the row, column of product.
            x[row][col] = 0;
            for (int inner = 0; inner < MATRIXLEN; ++inner) {
                x[row][col] += matrix1[row][inner] * matrix2[inner][col];
            }
        }
    }
    return x;
}

void Convert1D(Matrix* dst, int** src) {
    for (int i = 0, k = 0; i < MATRIXLEN; ++i) {
        for (int j = 0; j < MATRIXLEN; ++j) {
            dst->elements[k] = src[i][j];
            ++k;
        }
    }
}

int** Convert2D(Matrix src) {
    int** x = nullptr;
    GenerateMatrix(&x);
    for (int i = 0, k = 0; i < MATRIXLEN; ++i) {
        for (int j = 0; j < MATRIXLEN; ++j) {
            x[i][j] = src.elements[k];
            ++k;
        }
    }
    return x;
}