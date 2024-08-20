#define ROCBLAS_BETA_FEATURES_API
#include "helpers.hpp"
#include <hip/hip_runtime_api.h>
#include <math.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// Assuming hD and hGold are defined elsewhere as 2D arrays
void printMatrix(const std::vector<float>& matrix, int rows, int cols, const std::string& name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv)
{
    helpers::ArgParser options("MNKabc");
    if(!options.validArgs(argc, argv))
        return EXIT_FAILURE;

    rocblas_status rstatus = rocblas_status_success;

    typedef float dataType;

    rocblas_int M = options.M;
    rocblas_int N = options.N;
    rocblas_int K = options.K;

    float hAlpha = options.alpha;
    std::cout <<"hAlpha:"<< hAlpha << "\n";
   
    float hBeta  = options.beta;
    std::cout <<"hBeta:"<< hBeta << "\n";

    rocblas_int batchCount = options.batchCount;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_int lda, ldb, ldc, ldd; // leading dimension of matrices
    rocblas_int strideA1, strideA2, strideB1, strideB2;

    rocblas_stride strideA, strideB, strideC, strideD;

    if(transA == rocblas_operation_none)
    {
        lda      = M;
        strideA  = rocblas_stride(K) * lda;
        strideA1 = 1;
        strideA2 = lda;
    }
    else
    {
        lda      = K;
        strideA  = rocblas_stride(M) * lda;
        strideA1 = lda;
        strideA2 = 1;
    }
    if(transB == rocblas_operation_none)
    {
        ldb      = K;
        strideB  = rocblas_stride(N) * ldb;
        strideB1 = 1;
        strideB2 = ldb;
    }
    else
    {
        ldb      = N;
        strideB  = rocblas_stride(K) * ldb;
        strideB1 = ldb;
        strideB2 = 1;
    }
    ldc     = M;
    strideC = rocblas_stride(N) * ldc;
    ldd     = M;
    strideD = rocblas_stride(N) * ldd;

    rocblas_int cnt        = std::max(batchCount, 1);
    size_t      totalSizeA = size_t(strideA) * cnt;
    size_t      totalSizeB = size_t(strideB) * cnt;
    size_t      totalSizeC = size_t(strideC) * cnt;
    size_t      totalSizeD = size_t(strideD) * cnt;

    // using rocblas API
    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory

    // using single block of contiguous data for all batches
    std::vector<dataType> hA(totalSizeA, 1);
    std::vector<dataType> hB(totalSizeB);
    std::vector<dataType> hC(totalSizeC, 1);
    std::vector<dataType> hD(totalSizeD, 0); // output buffer
    std::vector<dataType> hGold(totalSizeD);

    for(int i = 0; i < batchCount; i++)
    {
        helpers::matIdentity(hB.data() + i * strideB, K, N, ldb);
    }
    hGold = hC;

    {
        // allocate memory on device
        helpers::DeviceVector<dataType> dA(totalSizeA);
        helpers::DeviceVector<dataType> dB(totalSizeB);
        helpers::DeviceVector<dataType> dC(totalSizeC);
        helpers::DeviceVector<dataType> dD(totalSizeD);

        if(!dA || !dB || !dC || !dD)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return EXIT_FAILURE;
        }

        // copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dA,
                                  static_cast<void*>(hA.data()),
                                  sizeof(dataType) * totalSizeA,
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dB,
                                  static_cast<void*>(hB.data()),
                                  sizeof(dataType) * totalSizeB,
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dC,
                                  static_cast<void*>(hC.data()),
                                  sizeof(dataType) * totalSizeC,
                                  hipMemcpyHostToDevice));

        // enable passing alpha parameter from pointer to host memory
        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        // asynchronous calculation on device, returns before finished calculations
        rstatus = rocblas_gemm_strided_batched_ex3(handle,
                                                transA,
                                                transB,
                                                M,
                                                N,
                                                K,
                                                &hAlpha,
                                                dA,
                                                rocblas_datatype_f8_r,
                                                lda,
                                                strideA,
                                                dB,
                                                rocblas_datatype_f8_r,
                                                ldb,
                                                strideB,
                                                &hBeta,
                                                dC,
                                                rocblas_datatype_f32_r,
                                                ldc,
                                                strideC,
                                                dD,
                                                rocblas_datatype_f32_r,
                                                ldd,
                                                strideD,
                                                batchCount,
                                                rocblas_compute_type_f32,
                                                rocblas_gemm_algo_standard,
                                                0,
                                                0);

        // check that calculation was launched correctly on device, not that result
        // was computed yet
        CHECK_ROCBLAS_STATUS(rstatus);

        // fetch device memory results, automatically blocked until results ready
        CHECK_HIP_ERROR(
            hipMemcpy(hD.data(), dD, sizeof(dataType) * totalSizeD, hipMemcpyDeviceToHost));

    } // release device memory via helpers::DeviceVector destructors

    std::cout << "M, N, K, lda, ldb, ldc = " << M << ", " << N << ", " << K << ", " << lda << ", "
              << ldb << ", " << ldc << std::endl;

    // calculate gold standard using CPU
    for(int i = 0; i < batchCount; i++)
    {
        float* aPtr = &hA[i * strideA];
        float* bPtr = &hB[i * strideB];
        float* cPtr = &hGold[i * strideD];

        helpers::matMatMult<dataType>(hAlpha,
                                      hBeta,
                                      M,
                                      N,
                                      K,
                                      aPtr,
                                      strideA1,
                                      strideA2,
                                      bPtr,
                                      strideB1,
                                      strideB2,
                                      cPtr,
                                      1,
                                      ldd);
    }

    dataType maxRelativeError = (dataType)helpers::maxRelativeError(hD, hGold);
    dataType eps              = std::numeric_limits<dataType>::epsilon();
    float    tolerance        = 10;
    if(maxRelativeError > eps * tolerance)
    {
        std::cout << "FAIL";
    }
    else
    {
        std::cout << "PASS";
    }
    std::cout << ": max. relative err. = " << maxRelativeError << std::endl;
    
    printMatrix(hD, M, K, "hD");
    printMatrix(hGold, M, K, "hGold");
    printMatrix(hA, M, N, "hA");
    printMatrix(hB, N, K, "hB");
    printMatrix(hC, M, K, "hC");
    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    return EXIT_SUCCESS;
}
