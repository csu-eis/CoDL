// command line GEMM lowp
#ifdef __APPLE__
#include <sys/time.h>
#endif

#include <cstdint>
#include "gflags/gflags.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <vector>
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "test.h"

#ifndef GEMMLOWP_TEST_BIT_DEPTH_PARAMS
#define GEMMLOWP_TEST_BIT_DEPTH_PARAMS DefaultL8R8BitDepthParams
// #define GEMMLOWP_TEST_BIT_DEPTH_PARAMS DefaultL32R32BitDepthParams
#endif

#if defined(__arm__) && !defined(GEMMLOWP_NEON)
#warning "Building without NEON support on ARM, check your compiler setup!"
#endif

#if defined(__mips) && !defined(GEMMLOWP_MSA)
#warning "Building without MSA support on MIPS, check your compiler setup!"
#endif

#if defined(__AVX2__) && !defined(GEMMLOWP_AVX2)
#warning \
    "Building without AVX2 support on AVX2 enabled machine, check your compiler setup!"
#endif

#if defined(__SSE4_2__) && !defined(GEMMLOWP_AVX2) && !defined(GEMMLOWP_SSE4)
#warning \
    "Building without SSE4.2 support on SSE4.2 enabled machine, check your compiler setup!"
#endif

// lhs: left hand side, rhs: right hand side
DEFINE_int32(num_rows, 1024, "number of lhs matrix rows");
DEFINE_int32(num_cols, 1024, "number of rhs matrix columns");
DEFINE_int32(num_depth, 1024, "number of lhs matrix columns");
DEFINE_string(data_type, "uint8", "matrix element data type, choices: `uint8`");

int PrintFlags() {
  std::cout << "number of rows: " << FLAGS_num_rows << std::endl;
  std::cout << "number of cols: " << FLAGS_num_cols << std::endl;
  std::cout << "number of depth: " << FLAGS_num_depth << std::endl;
  std::cout << "data type: " << FLAGS_data_type << std::endl;
  return 0;
}

namespace gemmlowp {
const double min_accurate_duration = 1e-1;
const std::size_t min_working_set_size = 16 * 1024 * 1024;

struct gemm_t {
  int rows, depth, cols;
  gemm_t() : rows(0), depth(0), cols(0) {}
  gemm_t(int r, int d, int c) : rows(r), depth(d), cols(c) {}

  friend std::ostream& operator<<(std::ostream& os, const gemm_t& obj) {
    os << obj.rows << " " << obj.depth << " " << obj.cols;
    return os;
  }
};

bool operator<(const gemm_t& a, const gemm_t& b) {
  return a.rows < b.rows ||
         (a.rows <= b.rows &&
          (a.depth < b.depth || (a.depth <= b.depth && (a.cols < b.cols))));
}

template <typename LhsType, typename RhsType, typename ResultType, typename Scalar>
double time_for_gemms(GemmContext* context, const std::vector<gemm_t>& gemms) {
//   typedef std::uint8_t Scalar;

  // set up the matrix pool

  std::size_t combined_gemm_sizes = 0;
  for (auto gemm : gemms) {
    int rows = gemm.rows;
    int depth = gemm.depth;
    int cols = gemm.cols;
    combined_gemm_sizes +=
        sizeof(Scalar) * (rows * depth + depth * cols + rows * cols);
  }

  const std::size_t pool_size = 1 + min_working_set_size / combined_gemm_sizes;

  std::vector<LhsType> lhs(pool_size * gemms.size());
  std::vector<RhsType> rhs(pool_size * gemms.size());
  std::vector<ResultType> result(pool_size * gemms.size());

  for (std::size_t i = 0; i < pool_size; i++) {
    for (std::size_t j = 0; j < gemms.size(); j++) {
      int k = i * gemms.size() + j;
      lhs[k].Resize(gemms[j].rows, gemms[j].depth);
      MakeConstant(&lhs[k], 0);
      rhs[k].Resize(gemms[j].depth, gemms[j].cols);
      MakeConstant(&rhs[k], 0);
      result[k].Resize(gemms[j].rows, gemms[j].cols);
      MakeConstant(&result[k], 0);
    }
  }

  // main benchmark loop

  int iters_at_a_time = 1;
  float time_per_iter = 0.0f;
  std::size_t pool_index = 0;

  while (true) {
    double starttime = real_time_in_seconds();
    for (int i = 0; i < iters_at_a_time; i++) {
      for (size_t j = 0; j < gemms.size(); j++) {
        size_t k = pool_index * gemms.size() + j;
        Gemm<Scalar, GEMMLOWP_TEST_BIT_DEPTH_PARAMS>(
            context, lhs[k].const_map(), rhs[k].const_map(), &result[k].map(),
            -75, -91, 74980, 123, 20);
      }
      pool_index++;
      if (pool_index == pool_size) {
        pool_index = 0;
      }
    }
    double endtime = real_time_in_seconds();

    const float timing = static_cast<float>(endtime - starttime);

    // double increase `iters` to fit `min_accurate_duration`
    if (timing >= min_accurate_duration) {
      time_per_iter = timing / iters_at_a_time;
      break;
    }

    iters_at_a_time *= 2;
  }

  return time_per_iter;
}

template<typename LhsType, typename RhsType, typename ResultType, typename Scalar>
void benchmark_gemm_sizes(GemmContext* context,
                          const std::vector<gemm_t>& gemms, double mintime) {
//   typedef Matrix<std::uint8_t, MapOrder::RowMajor> LhsType;
//   typedef Matrix<std::uint8_t, MapOrder::ColMajor> RhsType;
//   typedef Matrix<std::uint8_t, MapOrder::ColMajor> ResultType;

  std::vector<float> gemm_times;
  std::cout << "running for " << mintime << " seconds..." << std::endl;

#ifdef GEMMLOWP_TEST_PROFILE
  gemmlowp::RegisterCurrentThreadForProfiling();
  gemmlowp::StartProfiling();
#endif

  double starttime = real_time_in_seconds();
  while (real_time_in_seconds() < starttime + mintime) {
    gemm_times.push_back(
        time_for_gemms<LhsType, RhsType, ResultType, Scalar>(context, gemms));
  }

#ifdef GEMMLOWP_TEST_PROFILE
  gemmlowp::FinishProfiling();
#endif

  std::sort(gemm_times.begin(), gemm_times.end());

  double sum_gemm_times = 0;
  double sum_gemm_times_trimmed = 0;
  int count_gemm_times_trimmed = 0;
  const float trim_ratio = 0.25;
  const size_t count_trimmed = gemm_times.size() * trim_ratio;
  double sum_gemm_times_best = 0;
  int count_gemm_times_best = 0;
  const float best_ratio = 0.1;
  const size_t count_best = gemm_times.size() * best_ratio;

  for (size_t i = 0; i < gemm_times.size(); i++) {
    sum_gemm_times += gemm_times[i];
    if (i >= count_trimmed && i < gemm_times.size() - count_trimmed) {
      sum_gemm_times_trimmed += gemm_times[i];
      count_gemm_times_trimmed++;
    }
    if (i < count_best) {
      sum_gemm_times_best += gemm_times[i];
      count_gemm_times_best++;
    }
  }

  const double min_latency = gemm_times.front();
  const double max_latency = gemm_times.back();
  const double mean_latency = sum_gemm_times / gemm_times.size();
  const double trimmed_mean_latency =
      sum_gemm_times_trimmed / count_gemm_times_trimmed;
  const double best_mean_latency = sum_gemm_times_best / count_gemm_times_best;

  std::cout << "Graph latency (over " << gemm_times.size()
            << " iterations):" << std::endl;
  std::cout << "  Best:             " << min_latency << "s" << std::endl;
  std::cout << "  Worst:            " << max_latency << "s" << std::endl;
  std::cout << "  Mean:             " << mean_latency << "s" << std::endl;
  std::cout << "  " << 100 * trim_ratio
            << "% trimmed mean: " << trimmed_mean_latency << "s" << std::endl;
  std::cout << "  Mean of " << 100 * best_ratio
            << "% best: " << best_mean_latency << "s" << std::endl;
}

void run() {

    double mintime = 60;
    std::vector<gemm_t> gemms(1);
    gemms.emplace_back(FLAGS_num_rows, FLAGS_num_depth, FLAGS_num_cols);

    // FIXME: change code to fit `int32` data type
    // if (FLAGS_data_type == "int32") {
    //     gemmlowp::GemmContext context;
    //     typedef std::int32_t Scalar;
    //     typedef Matrix<Scalar, MapOrder::RowMajor> LhsType2;
    //     typedef Matrix<Scalar, MapOrder::ColMajor> RhsType2;
    //     typedef Matrix<Scalar, MapOrder::ColMajor> ResultType2;

    //     benchmark_gemm_sizes<LhsType2, RhsType2, ResultType2, Scalar>(&context, gemms, mintime);
    // }

    if (FLAGS_data_type == "uint8") {
        gemmlowp::GemmContext context;
        typedef std::uint8_t Scalar;
        typedef Matrix<Scalar, MapOrder::RowMajor> LhsType;
        typedef Matrix<Scalar, MapOrder::ColMajor> RhsType;
        typedef Matrix<Scalar, MapOrder::ColMajor> ResultType; 

        benchmark_gemm_sizes<LhsType, RhsType, ResultType, Scalar>(&context, gemms, mintime);
    }
    
}
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    PrintFlags();
    // std::cout << "hello world" << std::endl;
    gemmlowp::run();
}