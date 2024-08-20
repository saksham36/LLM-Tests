import torch
import time
# from tabulate import tabulate
import pandas as pd

def matmul_function(A, B, C, out_dtype):
    if out_dtype in [torch.float32, torch.bfloat16]:
        torch.matmul(A, B, out=C)
    else:
        
        f32_type = torch.float32
        bf16_type = torch.bfloat16
        e4m3_type = torch.float8_e4m3fn
        e5m2_type = torch.float8_e5m2
        device="cuda:0"
        output, output_amax = torch._scaled_mm(
            torch.randn(16,16, device=device).to(e4m3_type),
            torch.randn(16,16, device=device).to(e4m3_type).t(),
            bias=torch.randn(16, device=device).to(bf16_type),
            out_dtype=e4m3_type,
            scale_a=torch.tensor(1.0, device=device),
            scale_b=torch.tensor(1.0, device=device)
        )
        import pdb; pdb.set_trace()
        C = torch._scaled_mm(A, B.t())
def time_matrix_multiplication(A, B, C, out_dtype, warmup_steps=10, test_steps=100):
    # Warmup steps
    for _ in range(warmup_steps):
        matmul_function(A, B, C, out_dtype)
        torch.cuda.synchronize()

    # Time the actual computation
    start_time = time.time()
    for _ in range(test_steps):
        matmul_function(A, B, C, out_dtype)
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / test_steps
    return avg_time


def main():
    if torch.cuda.is_available():
        print(f"Num devices: {torch.cuda.device_count()}")

        # collect finfo for each type
        # table = []
        # for dtype in [f32_type, bf16_type, e4m3_type, e5m2_type]:
        #     numbits = 32 if dtype == f32_type else 16 if dtype == bf16_type else 8
        #     info = torch.finfo(dtype)
        #     table.append([info.dtype, numbits, info.max, 
        #                 info.min, info.smallest_normal, info.eps])

        # headers = ['data type', 'bits', 'max', 'min', 'smallest normal', 'eps']
        # print(tabulate(table, headers=headers))


        # Initialize matrices
        matrix_size = 1024  # Adjust the size as needed
        A_fp32 = torch.randn((matrix_size, matrix_size), dtype=torch.float32, device='cuda')
        B_fp32 = torch.randn((matrix_size, matrix_size), dtype=torch.float32, device='cuda')

        # Convert matrices to other data types
        A_bf16 = A_fp32.to(torch.bfloat16)
        B_bf16 = B_fp32.to(torch.bfloat16)
        A_float8_e4m3fn = A_fp32.to(torch.float8_e4m3fn)  # Placeholder for float8_e4m3fn
        B_float8_e4m3fn = B_fp32.to(torch.float8_e4m3fn)  # Placeholder for float8_e4m3fn
        A_float8_e5m2 = A_fp32.to(torch.float8_e5m2)  # Placeholder for float8_e5m2
        B_float8_e5m2 = B_fp32.to(torch.float8_e5m2)  # Placeholder for float8_e5m2

        # Dictionary to store results
        results = {}

        # Pre-allocate result matrices
        C_fp32 = torch.empty((matrix_size, matrix_size), dtype=torch.float32, device='cuda')
        C_bf16 = torch.empty((matrix_size, matrix_size), dtype=torch.bfloat16, device='cuda')
        C_float8_e4m3fn = torch.empty((matrix_size, matrix_size), dtype=torch.float8_e4m3fn, device='cuda')  # Placeholder
        C_float8_e5m2 = torch.empty((matrix_size, matrix_size), dtype=torch.float8_e5m2, device='cuda')  # Placeholder

        # Test each data type
        results['fp32'] = time_matrix_multiplication(A_fp32, B_fp32, C_fp32, torch.float32)
        results['bf16'] = time_matrix_multiplication(A_bf16, B_bf16, C_bf16, torch.bfloat16)
        results['float8_e4m3fn'] = time_matrix_multiplication(A_float8_e4m3fn, B_float8_e4m3fn, C_bf16, torch.float8_e4m3fn)
        results['float8_e5m2'] = time_matrix_multiplication(A_float8_e5m2, B_float8_e5m2, C_bf16, torch.float8_e5m2)

        df = pd.DataFrame.from_dict(results, orient='index', columns=['Average Time (s)'])
        print(df)
    else:
        print(f"No GPUs. Exiting")


if __name__ == "__main__":
    main()