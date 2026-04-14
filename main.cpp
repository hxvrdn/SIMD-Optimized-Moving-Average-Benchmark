#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kPriceCount = 10'000'000;
constexpr std::size_t kWindowSize = 32;
constexpr float kInvWindow = 1.0F / static_cast<float>(kWindowSize);

template <typename T, std::size_t Alignment>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n > static_cast<std::size_t>(-1) / sizeof(T)) {
            throw std::bad_alloc();
        }
        auto* ptr = static_cast<T*>(::operator new[](n * sizeof(T), std::align_val_t(Alignment)));
        return ptr;
    }

    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete[](p, std::align_val_t(Alignment));
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&)
{
    return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) {
    return false;
}

using AlignedFloatVector = std::vector<float, AlignedAllocator<float, 32>>;

std::vector<float> make_prices() {
    std::vector<float> prices(kPriceCount);
    for (std::size_t i = 0; i < prices.size(); ++i) {
        prices[i] = 90.0F + static_cast<float>(i % 512) * 0.03125F;
    }
    return prices;
}

AlignedFloatVector make_aligned_prefix(const std::vector<float>& prices) {
    AlignedFloatVector prefix(prices.size() + 1, 0.0F);
    for (std::size_t i = 0; i < prices.size(); ++i) {
        prefix[i + 1] = prefix[i] + prices[i];
    }
    return prefix;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
void moving_average_scalar_no_vec(const AlignedFloatVector& prefix, std::vector<float>& output) {
#if defined(_MSC_VER)
#pragma loop(no_vector)
#endif
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] = (prefix[i + kWindowSize] - prefix[i]) * kInvWindow;
    }
}

void moving_average_auto_vec(const AlignedFloatVector& prefix, std::vector<float>& output) {
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] = (prefix[i + kWindowSize] - prefix[i]) * kInvWindow;
    }
}

void moving_average_avx2(const AlignedFloatVector& prefix, AlignedFloatVector& output) {
    const std::size_t n = output.size();
    const __m256 inv_window = _mm256_set1_ps(kInvWindow);

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 hi = _mm256_load_ps(prefix.data() + i + kWindowSize);
        const __m256 lo = _mm256_load_ps(prefix.data() + i);
        const __m256 avg = _mm256_mul_ps(_mm256_sub_ps(hi, lo), inv_window);
        _mm256_store_ps(output.data() + i, avg);
    }

    for (; i < n; ++i) {
        output[i] = (prefix[i + kWindowSize] - prefix[i]) * kInvWindow;
    }
}

inline std::uint64_t rdtsc_now() {
#if defined(_MSC_VER)
    return __rdtsc();
#else
    unsigned int aux = 0;
    return __rdtscp(&aux);
#endif
}

template <typename Fn>
void run_benchmark(benchmark::State& state, Fn&& fn) {
    static const std::vector<float> prices = make_prices();
    static const AlignedFloatVector prefix = make_aligned_prefix(prices);
    static std::vector<float> out_scalar(prices.size() - kWindowSize + 1, 0.0F);
    static AlignedFloatVector out_aligned(prices.size() - kWindowSize + 1, 0.0F);
    alignas(32) static std::array<float, 8> alignment_probe{};

    benchmark::DoNotOptimize(alignment_probe.data());

    std::uint64_t total_cycles = 0;
    for (auto _ : state) {
        const std::uint64_t start = rdtsc_now();
        fn(prefix, out_scalar, out_aligned);
        const std::uint64_t stop = rdtsc_now();
        total_cycles += (stop - start);

        benchmark::DoNotOptimize(out_scalar.data());
        benchmark::DoNotOptimize(out_aligned.data());
        benchmark::ClobberMemory();
    }

    const double elements = static_cast<double>(out_scalar.size()) * static_cast<double>(state.iterations());
    state.counters["cycles_per_element"] = benchmark::Counter(
        static_cast<double>(total_cycles) / elements,
        benchmark::Counter::kAvgThreads
    );
    state.counters["elements"] = benchmark::Counter(static_cast<double>(out_scalar.size()), benchmark::Counter::kDefaults);
}

void bench_scalar_no_vec(benchmark::State& state) {
    run_benchmark(state, [](const AlignedFloatVector& prefix, std::vector<float>& out_scalar, AlignedFloatVector&) {
        moving_average_scalar_no_vec(prefix, out_scalar);
    });
}

void bench_auto_vec(benchmark::State& state) {
    run_benchmark(state, [](const AlignedFloatVector& prefix, std::vector<float>& out_scalar, AlignedFloatVector&) {
        moving_average_auto_vec(prefix, out_scalar);
    });
}

void bench_avx2(benchmark::State& state) {
    run_benchmark(state, [](const AlignedFloatVector& prefix, std::vector<float>&, AlignedFloatVector& out_aligned) {
        moving_average_avx2(prefix, out_aligned);
    });
}

BENCHMARK(bench_scalar_no_vec);
BENCHMARK(bench_auto_vec);
BENCHMARK(bench_avx2);

}  // namespace

BENCHMARK_MAIN();
