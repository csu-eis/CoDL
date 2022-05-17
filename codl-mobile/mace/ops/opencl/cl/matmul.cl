#include <common.h>

// C = A * B
// Origin implementation of MACE, we can name it as matmul_mace.
__kernel void matmul(
    OUT_OF_RANGE_PARAMS
    GLOBAL_WORK_GROUP_SIZE_DIM2
    __read_only image2d_t A,  // [K/4, M]
    __read_only image2d_t B,  // [N, K/4]
#ifdef BIAS
    __read_only image2d_t bias,
#endif
    __private const int M,
    __private const int N,
    __private const int K,
    __private const int height_blocks,
    __private const int k_blocks,
#if  defined(USE_RELU) || defined(USE_LEAKYRELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
    __private const float relux_max_limit,
    __private const float leakyrelu_coefficient,
#endif
    __write_only image2d_t C  // [N, M/4]
) {
  /**
   * gx: col_idx
   * ty: row_blk_idx
   * gy: row_blk_idx
   * bm: row_idx
   * bk: depth_blk_base
   * pos: depth_blk_idx
   */
  const int gx = get_global_id(0) << 2;
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (get_global_id(0) >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  const int batch = hb / height_blocks;
  const int ty = hb - mul24(batch, height_blocks);
  const int gy = mad24(batch, height_blocks, ty);
  const int bm = mad24(batch, M, ty << 2);
  const int bk = mul24(batch, k_blocks);

#ifdef BIAS
  DATA_TYPE4 c0 = READ_IMAGET(bias, SAMPLER, (int2)(bm, 0));
  DATA_TYPE4 c1 = c0;
  DATA_TYPE4 c2 = c0;
  DATA_TYPE4 c3 = c0;
#else
  DATA_TYPE4 c0 = 0;
  DATA_TYPE4 c1 = 0;
  DATA_TYPE4 c2 = 0;
  DATA_TYPE4 c3 = 0;
#endif

  DATA_TYPE4 a0, a1, a2, a3;
  DATA_TYPE4 b0, b1, b2, b3;
  for (short pos = 0; pos < k_blocks; pos += 1) {
    a0 = READ_IMAGET(A, SAMPLER, (int2)(pos, (bm)));
    a1 = READ_IMAGET(A, SAMPLER, (int2)(pos, (bm + 1)));
    a2 = READ_IMAGET(A, SAMPLER, (int2)(pos, (bm + 2)));
    a3 = READ_IMAGET(A, SAMPLER, (int2)(pos, (bm + 3)));

    b0 = READ_IMAGET(B, SAMPLER, (int2)(gx, (bk + pos)));
    b1 = READ_IMAGET(B, SAMPLER, (int2)(gx + 1, (bk + pos)));
    b2 = READ_IMAGET(B, SAMPLER, (int2)(gx + 2, (bk + pos)));
    b3 = READ_IMAGET(B, SAMPLER, (int2)(gx + 3, (bk + pos)));

    c0 += (DATA_TYPE4)(dot(a0, b0), dot(a1, b0), dot(a2, b0), dot(a3, b0));
    c1 += (DATA_TYPE4)(dot(a0, b1), dot(a1, b1), dot(a2, b1), dot(a3, b1));
    c2 += (DATA_TYPE4)(dot(a0, b2), dot(a1, b2), dot(a2, b2), dot(a3, b2));
    c3 += (DATA_TYPE4)(dot(a0, b3), dot(a1, b3), dot(a2, b3), dot(a3, b3));
  }

#if  defined(USE_RELU) || defined(USE_LEAKYRELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
  c0 = do_activation(c0, relux_max_limit, leakyrelu_coefficient);
  c1 = do_activation(c1, relux_max_limit, leakyrelu_coefficient);
  c2 = do_activation(c2, relux_max_limit, leakyrelu_coefficient);
  c3 = do_activation(c3, relux_max_limit, leakyrelu_coefficient);
#endif

  WRITE_IMAGET(C, (int2)(gx, gy), c0);

  if ((gx + 1) >= N) return;
  WRITE_IMAGET(C, (int2)(gx + 1, gy), c1);

  if ((gx + 2) >= N) return;
  WRITE_IMAGET(C, (int2)(gx + 2, gy), c2);

  if ((gx + 3) >= N) return;
  WRITE_IMAGET(C, (int2)(gx + 3, gy), c3);
}

// NOTE(fucheng): For Conv2dK1x1S1.
// LIMIT(fucheng): Width of input/output tensor must be 1.
__kernel void matmul_conv_2d_k1x1s1(
    OUT_OF_RANGE_PARAMS
    GLOBAL_WORK_GROUP_SIZE_DIM2
    __read_only image2d_t A,  // [K, M/4], CONV2D_FILTER
    __read_only image2d_t B,  // [K/4, N], IN_OUT_CHANNEL
#ifdef BIAS
    __read_only image2d_t bias,
#endif
    __private const int M,
    __private const int N,
    __private const int K,
    __private const int height_blocks,
    __private const int k_blocks,
#if  defined(USE_RELU) || defined(USE_LEAKYRELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
    __private const float relux_max_limit,
    __private const float leakyrelu_coefficient,
#endif
    __write_only image2d_t C  // [M/4, N], IN_OUT_CHANNEL
) {
  /**
   * gx: col_idx
   * ty: row_blk_idx
   * gy: batch_row_blk_idx
   * bm: batch_row_idx
   * bk: batch_depth_blk_base
   * pos: depth_blk_idx
   */
  const int gx = get_global_id(0) << 2;
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (get_global_id(0) >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  const int batch = hb / height_blocks;
  const int ty = hb - mul24(batch, height_blocks);
  const int gy = mad24(batch, height_blocks, ty);
  const int bm = mad24(batch, M, ty << 2);
  const int bk = mul24(batch, k_blocks);

#ifdef BIAS
  DATA_TYPE4 c0 = READ_IMAGET(bias, SAMPLER, (int2)(gy, 0));
  DATA_TYPE4 c1 = c0;
  DATA_TYPE4 c2 = c0;
  DATA_TYPE4 c3 = c0;
#else
  DATA_TYPE4 c0 = 0;
  DATA_TYPE4 c1 = 0;
  DATA_TYPE4 c2 = 0;
  DATA_TYPE4 c3 = 0;
#endif

  DATA_TYPE4 a0, a1, a2, a3;
  DATA_TYPE4 b0, b1, b2, b3;
  for (short pos = 0; pos < k_blocks; pos += 1) {
    int k_idx = pos << 2;
    
    // 4 k and 1 m block from A.
    a0 = READ_IMAGET(A, SAMPLER, (int2)(k_idx, gy));
    a1 = READ_IMAGET(A, SAMPLER, (int2)(k_idx + 1, gy));
    a2 = READ_IMAGET(A, SAMPLER, (int2)(k_idx + 2, gy));
    a3 = READ_IMAGET(A, SAMPLER, (int2)(k_idx + 3, gy));

    // 1 k block and 4 n from B.
    b0 = READ_IMAGET(B, SAMPLER, (int2)(pos, gx));
    b1 = READ_IMAGET(B, SAMPLER, (int2)(pos, gx + 1));
    b2 = READ_IMAGET(B, SAMPLER, (int2)(pos, gx + 2));
    b3 = READ_IMAGET(B, SAMPLER, (int2)(pos, gx + 3));

#define CALC_N(i) \
    c##i = mad(b##i.x, a0, c##i); \
    c##i = mad(b##i.y, a1, c##i); \
    c##i = mad(b##i.z, a2, c##i); \
    c##i = mad(b##i.w, a3, c##i);

    CALC_N(0);
    CALC_N(1);
    CALC_N(2);
    CALC_N(3);
#undef CALC_N
  }

#if  defined(USE_RELU) || defined(USE_LEAKYRELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
  c0 = do_activation(c0, relux_max_limit, leakyrelu_coefficient);
  c1 = do_activation(c1, relux_max_limit, leakyrelu_coefficient);
  c2 = do_activation(c2, relux_max_limit, leakyrelu_coefficient);
  c3 = do_activation(c3, relux_max_limit, leakyrelu_coefficient);
#endif

  WRITE_IMAGET(C, (int2)(gy, gx), c0);

  if ((gx + 1) >= N) return;
  WRITE_IMAGET(C, (int2)(gy, gx + 1), c1);

  if ((gx + 2) >= N) return;
  WRITE_IMAGET(C, (int2)(gy, gx + 2), c2);

  if ((gx + 3) >= N) return;
  WRITE_IMAGET(C, (int2)(gy, gx + 3), c3);
}

// NOTE(fucheng): No transpose, rank is 2.
__kernel void matmul_r2(
    OUT_OF_RANGE_PARAMS
    GLOBAL_WORK_GROUP_SIZE_DIM2
    __read_only image2d_t A,  // [M, K] -> [K/4, M], IN_OUT_CHANNEL
    __read_only image2d_t B,  // [K, N] -> [N/4, K], IN_OUT_CHANNEL
    __private const int M,
    __private const int N,
    __private const int K,
    __private const int height_blocks,
    __private const int k_blocks,
    __write_only image2d_t C  // [M, N] -> [N/4, M], IN_OUT_CHANNEL
) {
  /**
   * gx: col_idx
   * ty: row_blk_idx
   * gy: batch_row_blk_idx
   * m_idx: row_idx
   * bm: batch_row_idx
   * bk: depth_blk_base
   * pos: depth_blk_idx
   */
  const int gx_blk = get_global_id(0);
  const int gx = get_global_id(0) << 2;
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (get_global_id(0) >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  const int batch = hb / height_blocks;
  const int ty = hb - mul24(batch, height_blocks);
  const int gy = mad24(batch, height_blocks, ty);
  const int m_idx = ty << 2;
  const int bm = mad24(batch, M, m_idx);
  const int bk = mul24(batch, k_blocks);

  DATA_TYPE4 c0 = 0;
  DATA_TYPE4 c1 = 0;
  DATA_TYPE4 c2 = 0;
  DATA_TYPE4 c3 = 0;

  DATA_TYPE4 a0, a1, a2, a3;
  DATA_TYPE4 b0, b1, b2, b3;
  for (short pos = 0; pos < k_blocks; pos += 1) {
    const int k_idx = pos << 2;
    // 1 k block and 4 m from A.
    a0 = READ_IMAGET(A, SAMPLER, (int2)(pos, m_idx));
    a1 = READ_IMAGET(A, SAMPLER, (int2)(pos, m_idx + 1));
    a2 = READ_IMAGET(A, SAMPLER, (int2)(pos, m_idx + 2));
    a3 = READ_IMAGET(A, SAMPLER, (int2)(pos, m_idx + 3));

    // 1 n block and 4 k from B.
    b0 = READ_IMAGET(B, SAMPLER, (int2)(gx_blk, k_idx));
    b1 = READ_IMAGET(B, SAMPLER, (int2)(gx_blk, k_idx + 1));
    b2 = READ_IMAGET(B, SAMPLER, (int2)(gx_blk, k_idx + 2));
    b3 = READ_IMAGET(B, SAMPLER, (int2)(gx_blk, k_idx + 3));

#define CALC_M(i) \
    c##i = mad(a##i.x, b0, c##i); \
    c##i = mad(a##i.y, b1, c##i); \
    c##i = mad(a##i.z, b2, c##i); \
    c##i = mad(a##i.w, b3, c##i);

    CALC_M(0);
    CALC_M(1);
    CALC_M(2);
    CALC_M(3);
#undef CALC_M
  }

#if 0
  DATA_TYPE4 out = (DATA_TYPE4)(c0.x, c1.x, c2.x, c3.x);
  WRITE_IMAGET(C, (int2)(gy, gx), out);

  if ((gx + 1) >= N) return;
  out = (DATA_TYPE4)(c0.y, c1.y, c2.y, c3.y);
  WRITE_IMAGET(C, (int2)(gy, gx + 1), out);

  if ((gx + 2) >= N) return;
  out = (DATA_TYPE4)(c0.z, c1.z, c2.z, c3.z);
  WRITE_IMAGET(C, (int2)(gy, gx + 2), out);

  if ((gx + 3) >= N) return;
  out = (DATA_TYPE4)(c0.w, c1.w, c2.w, c3.w);
  WRITE_IMAGET(C, (int2)(gy, gx + 3), out);
#endif

#if 1
  WRITE_IMAGET(C, (int2)(gx_blk, m_idx), c0);

  if ((m_idx + 1) >= M) return;
  WRITE_IMAGET(C, (int2)(gx_blk, m_idx + 1), c1);

  if ((m_idx + 2) >= M) return;
  WRITE_IMAGET(C, (int2)(gx_blk, m_idx + 2), c2);

  if ((m_idx + 3) >= M) return;
  WRITE_IMAGET(C, (int2)(gx_blk, m_idx + 3), c3);
#endif
}

// NOTE(fucheng): No transpose, rank is 4.
__kernel void matmul_r4(
    OUT_OF_RANGE_PARAMS
    GLOBAL_WORK_GROUP_SIZE_DIM2
    __read_only image2d_t A,  // [1, B, M, K] -> [K/4 * M, B], IN_OUT_CHANNEL
    __read_only image2d_t B,  // [1, B, K, N] -> [N/4 * K, B], IN_OUT_CHANNEL
    __private const int M,
    __private const int N,
    __private const int K,
    __private const int height_blocks,
    __private const int k_blocks,
    __write_only image2d_t C  // [1, B, M, N] -> [N/4 * M, B], IN_OUT_CHANNEL
) {
  /**
   * gx: col_idx
   * ty: row_blk_idx
   * gy: batch_row_blk_idx
   * bm: batch_row_idx
   * bk: depth_blk_base
   * pos: depth_blk_idx
   */
  const int gx_blk = get_global_id(0);
  const int gx = gx_blk << 2;
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (get_global_id(0) >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  const int batch = hb / height_blocks;
  const int ty = hb - mul24(batch, height_blocks);
  const int gy = mad24(batch, height_blocks, ty);
  const int m_idx = ty << 2;
  const int bm = mad24(batch, M, m_idx);
  const int bk = mul24(batch, k_blocks);

  DATA_TYPE4 c0 = 0;
  DATA_TYPE4 c1 = 0;
  DATA_TYPE4 c2 = 0;
  DATA_TYPE4 c3 = 0;

  DATA_TYPE4 a0, a1, a2, a3;
  DATA_TYPE4 b0, b1, b2, b3;
  for (short pos = 0; pos < k_blocks; pos += 1) {
    const int k_idx = pos << 2;
    // 1 k block and 4 m from A.
    const int m_base = pos * M + m_idx;
    a0 = READ_IMAGET(A, SAMPLER, (int2)(m_base, batch));
    a1 = READ_IMAGET(A, SAMPLER, (int2)(m_base + 1, batch));
    a2 = READ_IMAGET(A, SAMPLER, (int2)(m_base + 2, batch));
    a3 = READ_IMAGET(A, SAMPLER, (int2)(m_base + 3, batch));

    // 1 n block and 4 k from B.
    const int k_base = gx_blk * K + k_idx;
    b0 = READ_IMAGET(B, SAMPLER, (int2)(k_base, batch));
    b1 = READ_IMAGET(B, SAMPLER, (int2)(k_base + 1, batch));
    b2 = READ_IMAGET(B, SAMPLER, (int2)(k_base + 2, batch));
    b3 = READ_IMAGET(B, SAMPLER, (int2)(k_base + 3, batch));

#define CALC_M(i) \
    c##i = mad(a##i.x, b0, c##i); \
    c##i = mad(a##i.y, b1, c##i); \
    c##i = mad(a##i.z, b2, c##i); \
    c##i = mad(a##i.w, b3, c##i);

    CALC_M(0);
    CALC_M(1);
    CALC_M(2);
    CALC_M(3);
#undef CALC_M
  }

#if 0
  const int n_base = ty * N + gx;
  DATA_TYPE4 out = (DATA_TYPE4)(c0.x, c1.x, c2.x, c3.x);
  WRITE_IMAGET(C, (int2)(n_base, batch), out);

  if ((gx + 1) >= N) return;
  out = (DATA_TYPE4)(c0.y, c1.y, c2.y, c3.y);
  WRITE_IMAGET(C, (int2)(n_base + 1, batch), out);

  if ((gx + 2) >= N) return;
  out = (DATA_TYPE4)(c0.z, c1.z, c2.z, c3.z);
  WRITE_IMAGET(C, (int2)(n_base + 2, batch), out);

  if ((gx + 3) >= N) return;
  out = (DATA_TYPE4)(c0.w, c1.w, c2.w, c3.w);
  WRITE_IMAGET(C, (int2)(n_base + 3, batch), out);
#endif

  const int n_base = gx_blk * M + m_idx;
  WRITE_IMAGET(C, (int2)(n_base, batch), c0);

  if ((m_idx + 1) >= M) return;
  WRITE_IMAGET(C, (int2)(n_base + 1, batch), c1);

  if ((m_idx + 2) >= M) return;
  WRITE_IMAGET(C, (int2)(n_base + 2, batch), c2);

  if ((m_idx + 3) >= M) return;
  WRITE_IMAGET(C, (int2)(n_base + 3, batch), c3);
}

// NOTE(fucheng): transpose_b is true.
__kernel void matmul_r4_tb(
    OUT_OF_RANGE_PARAMS
    GLOBAL_WORK_GROUP_SIZE_DIM2
    __read_only image2d_t A,  // [1, B, M, K] -> [K/4 * M, B], IN_OUT_CHANNEL
    __read_only image2d_t B,  // [1, B, N, K] -> [K/4 * N, B], IN_OUT_CHANNEL
    __private const int M,
    __private const int N,
    __private const int K,
    __private const int height_blocks,
    __private const int k_blocks,
    __write_only image2d_t C  // [1, B, M, N] -> [N/4 * M, B], IN_OUT_CHANNEL
) {
  /**
   * gx: col_idx
   * ty: row_blk_idx
   * gy: batch_row_blk_idx
   * bm: batch_row_idx
   * bk: batch_depth_blk_base
   * bn: batch_col_idx
   * pos: depth_blk_idx
   */
  const int gx_blk = get_global_id(0);
  const int gx = gx_blk << 2;
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (get_global_id(0) >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  const int batch = hb / height_blocks;
  const int ty = hb - mul24(batch, height_blocks);
  const int gy = mad24(batch, height_blocks, ty);
  const int m_idx = ty << 2;
  const int bm = mad24(batch, M, m_idx);
  const int bk = mul24(batch, k_blocks);
  const int bn = mad24(batch, N, gx);

  DATA_TYPE4 c0 = 0;
  DATA_TYPE4 c1 = 0;
  DATA_TYPE4 c2 = 0;
  DATA_TYPE4 c3 = 0;

  DATA_TYPE4 a0, a1, a2, a3;
  DATA_TYPE4 b0, b1, b2, b3;
  for (short pos = 0; pos < k_blocks; pos += 1) {
    // 1 k block and 4 m from A.
    const int m_base = pos * M + m_idx;
    a0 = READ_IMAGET(A, SAMPLER, (int2)(m_base, batch));
    a1 = READ_IMAGET(A, SAMPLER, (int2)(m_base + 1, batch));
    a2 = READ_IMAGET(A, SAMPLER, (int2)(m_base + 2, batch));
    a3 = READ_IMAGET(A, SAMPLER, (int2)(m_base + 3, batch));

    // 1 k block and 4 n from B.
    const int n_base = pos * N + gx;
    b0 = READ_IMAGET(B, SAMPLER, (int2)(n_base, batch));
    b1 = READ_IMAGET(B, SAMPLER, (int2)(n_base + 1, batch));
    b2 = READ_IMAGET(B, SAMPLER, (int2)(n_base + 2, batch));
    b3 = READ_IMAGET(B, SAMPLER, (int2)(n_base + 3, batch));

#define CALC_N(i) \
    c##i += (DATA_TYPE4)(dot(a0, b##i), dot(a1, b##i), dot(a2, b##i), dot(a3, b##i));

    CALC_N(0);
    CALC_N(1);
    CALC_N(2);
    CALC_N(3);
#undef CALC_N
  }

#if 0
  const int n_base = ty * N + gx;
  
  WRITE_IMAGET(C, (int2)(n_base, batch), c0);

  if ((gx + 1) >= N) return;
  WRITE_IMAGET(C, (int2)(n_base + 1, batch), c1);

  if ((gx + 2) >= N) return;
  WRITE_IMAGET(C, (int2)(n_base + 2, batch), c2);

  if ((gx + 3) >= N) return;
  WRITE_IMAGET(C, (int2)(n_base + 3, batch), c3);
#endif

  const int n_base = gx_blk * M + m_idx;
  DATA_TYPE4 out = (DATA_TYPE4)(c0.x, c1.x, c2.x, c3.x);
  WRITE_IMAGET(C, (int2)(n_base, batch), out);

  if ((m_idx + 1) >= M) return;
  out = (DATA_TYPE4)(c0.y, c1.y, c2.y, c3.y);
  WRITE_IMAGET(C, (int2)(n_base + 1, batch), out);

  if ((m_idx + 2) >= M) return;
  out = (DATA_TYPE4)(c0.z, c1.z, c2.z, c3.z);
  WRITE_IMAGET(C, (int2)(n_base + 2, batch), out);

  if ((m_idx + 3) >= M) return;
  out = (DATA_TYPE4)(c0.w, c1.w, c2.w, c3.w);
  WRITE_IMAGET(C, (int2)(n_base + 3, batch), out);
}


