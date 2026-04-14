# SIMD-Optimized Moving Average Benchmark

This project benchmarks three implementations of a Simple Moving Average (SMA) over 10 million `float` prices:

- Scalar baseline using a standard `for` loop (`moving_average_scalar_no_vec`)
- Compiler auto-vectorized loop at `-O3` (`moving_average_auto_vec`)
- Manual AVX2 intrinsics processing 8 floats at a time (`moving_average_avx2`)

Google Benchmark is used to measure throughput and report `cycles_per_element`.

## What is implemented

- Dataset: 10,000,000 synthetic prices (`std::vector<float>`)
- SMA window size: 32
- Prefix-sum formulation for efficient O(n) moving-average output
- 32-byte aligned buffers for SIMD loads/stores via aligned allocator
- One explicit `alignas(32)` alignment probe and aligned AVX2 loads (`_mm256_load_ps`)

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```bash
./build/SIMD_Optimized_Financial_Calculator --benchmark_min_time=0.2 --benchmark_counters_tabular=true
```

On Windows, the executable is typically:

```bash
./build/Release/SIMD_Optimized_Financial_Calculator.exe --benchmark_min_time=0.2 --benchmark_counters_tabular=true
```

## Notes on comparison

- `bench_scalar_no_vec`: scalar loop with vectorization disabled for a baseline
- `bench_auto_vec`: same loop compiled with optimization (`-O3`), allowing auto-vectorization
- `bench_avx2`: manual AVX2 intrinsics with aligned memory and aligned loads/stores

The key comparison metric is `cycles_per_element` (lower is better).

# SIMD-Optimized-Moving-Average-Benchmark
