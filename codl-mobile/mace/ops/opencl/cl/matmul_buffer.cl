#include <common.h>

// C = A * B + bias
__kernel void matmul_cpu_buffer(
    BUFFER_OUT_OF_RANGE_PARAMS
    GLOBAL_WORK_GROUP_SIZE_DIM2
    __global DATA_TYPE *A,  // [M, K], CPU_BUFFER
    __global DATA_TYPE *B,  // [K, N], CPU_BUFFER
#ifdef BIAS
    __global DATA_TYPE *bias,
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
    __global DATA_TYPE *C  // [M, N], CPU_BUFFER
) {
  /**
   * gx: col_idx
   * ty: row_blk_idx
   * gy: batch_base + row_blk_idx
   * bm: batch_base + row_idx
   * bk: batch_base + depth_blk_idx
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
  const int bm_idx = bm << 2;
  DATA_TYPE4 c0 = CONVERT4(vload4(0, bias + bm_idx));
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
    const int a_pos_offset = pos << 2;
    a0 = CONVERT4(vload4(0, A + bm * K + a_pos_offset));
    a1 = CONVERT4(vload4(0, A + (bm + 1) * K + a_pos_offset));
    a2 = CONVERT4(vload4(0, A + (bm + 2) * K + a_pos_offset));
    a3 = CONVERT4(vload4(0, A + (bm + 3) * K + a_pos_offset));

    const int b_pos_offset = (bk + pos) << 2;
    b0.x = CONVERT(B[b_pos_offset * N + gx]);
    b0.y = CONVERT(B[(b_pos_offset + 1) * N + gx]);
    b0.z = CONVERT(B[(b_pos_offset + 2) * N + gx]);
    b0.w = CONVERT(B[(b_pos_offset + 3) * N + gx]);

    b1.x = CONVERT(B[b_pos_offset * N + gx + 1]);
    b1.y = CONVERT(B[(b_pos_offset + 1) * N + gx + 1]);
    b1.z = CONVERT(B[(b_pos_offset + 2) * N + gx + 1]);
    b1.w = CONVERT(B[(b_pos_offset + 3) * N + gx + 1]);

    b2.x = CONVERT(B[b_pos_offset * N + gx + 2]);
    b2.y = CONVERT(B[(b_pos_offset + 1) * N + gx + 2]);
    b2.z = CONVERT(B[(b_pos_offset + 2) * N + gx + 2]);
    b2.w = CONVERT(B[(b_pos_offset + 3) * N + gx + 2]);

    b3.x = CONVERT(B[b_pos_offset * N + gx + 3]);
    b3.y = CONVERT(B[(b_pos_offset + 1) * N + gx + 3]);
    b3.z = CONVERT(B[(b_pos_offset + 2) * N + gx + 3]);
    b3.w = CONVERT(B[(b_pos_offset + 3) * N + gx + 3]);

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

  C[bm * N + gx] = CONVERT_TO(c0.x, DATA_TYPE);
  C[(bm + 1) * N + gx] = CONVERT_TO(c0.y, DATA_TYPE);
  C[(bm + 2) * N + gx] = CONVERT_TO(c0.z, DATA_TYPE);
  C[(bm + 3) * N + gx] = CONVERT_TO(c0.w, DATA_TYPE);

  if ((gx + 1) >= N) return;
  C[bm * N + gx + 1] = CONVERT_TO(c1.x, DATA_TYPE);
  C[(bm + 1) * N + gx + 1] = CONVERT_TO(c1.y, DATA_TYPE);
  C[(bm + 2) * N + gx + 1] = CONVERT_TO(c1.z, DATA_TYPE);
  C[(bm + 3) * N + gx + 1] = CONVERT_TO(c1.w, DATA_TYPE);

  if ((gx + 2) >= N) return;
  C[bm * N + gx + 2] = CONVERT_TO(c2.x, DATA_TYPE);
  C[(bm + 1) * N + gx + 2] = CONVERT_TO(c2.y, DATA_TYPE);
  C[(bm + 2) * N + gx + 2] = CONVERT_TO(c2.z, DATA_TYPE);
  C[(bm + 3) * N + gx + 2] = CONVERT_TO(c2.w, DATA_TYPE);

  if ((gx + 3) >= N) return;
  C[bm * N + gx + 3] = CONVERT_TO(c3.x, DATA_TYPE);
  C[(bm + 1) * N + gx + 3] = CONVERT_TO(c3.y, DATA_TYPE);
  C[(bm + 2) * N + gx + 3] = CONVERT_TO(c3.z, DATA_TYPE);
  C[(bm + 3) * N + gx + 3] = CONVERT_TO(c3.w, DATA_TYPE);
}

// C = A * B + bias
__kernel void matmul_conv_2d_k1x1s1(
    BUFFER_OUT_OF_RANGE_PARAMS
    GLOBAL_WORK_GROUP_SIZE_DIM2
    __global DATA_TYPE *A,  // [M/4, K], CONV2D_FILTER
    __global DATA_TYPE *B,  // [N, K/4], IN_OUT_CHANNEL
#ifdef BIAS
    __global DATA_TYPE *bias,
#endif
    __private const int M,
    __private const int N,
    __private const int K,
    __private const int m_blocks,
    __private const int k_blocks,
#if  defined(USE_RELU) || defined(USE_LEAKYRELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
    __private const float relux_max_limit,
    __private const float leakyrelu_coefficient,
#endif
    __global DATA_TYPE *C  // [N, M/4], IN_OUT_CHANNEL
) {
  /**
   * gx: col_idx = n_idx
   * ty: row_blk_idx = m
   * gy: batch_base + row_blk_idx = m
   * bm: batch_base + row_idx = m_idx
   * bk: batch_base + depth_blk_idx = k
   * pos: depth_blk_idx = k
   */
  const int gx = get_global_id(0) << 2;
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (get_global_id(0) >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  const int batch = hb / m_blocks;
  const int ty = hb - mul24(batch, m_blocks);
  const int gy = mad24(batch, m_blocks, ty);
  const int bm = mad24(batch, M, ty << 2);
  const int bk = mul24(batch, k_blocks);
  const int n_idx = gx;

#ifdef BIAS
  DATA_TYPE4 c0 = CONVERT4(vload4(0, bias + bm));
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
  for (int k = 0; k < k_blocks; k ++) {
    int k_idx = k << 2;
    // 1 m block and 4 k from A.
    int a_base_offset = mad24(gy, K, k_idx) << 2;
    a0 = CONVERT4(vload4(0, A + a_base_offset));
    a1 = CONVERT4(vload4(0, A + a_base_offset + 4));
    a2 = CONVERT4(vload4(0, A + a_base_offset + 8));
    a3 = CONVERT4(vload4(0, A + a_base_offset + 12));

    // 4 n and 1 k block from B.
    int b_base_offset = mad24(n_idx, K, k_idx);
    b0 = CONVERT4(vload4(0, B + b_base_offset));
    b1 = CONVERT4(vload4(0, B + b_base_offset + K));
    b2 = CONVERT4(vload4(0, B + b_base_offset + K * 2));
    b3 = CONVERT4(vload4(0, B + b_base_offset + K * 3));

#define CALC_N(i) \
    c##i = mad((DATA_TYPE4)(b##i.x), a0, c##i); \
    c##i = mad((DATA_TYPE4)(b##i.y), a1, c##i); \
    c##i = mad((DATA_TYPE4)(b##i.z), a2, c##i); \
    c##i = mad((DATA_TYPE4)(b##i.w), a3, c##i);

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

  //const int n_offset_size = m_blocks << 2;
  //int out_offset = (n_idx * m_blocks + gy) << 2;
  const int n_offset_size = M;
  int out_offset = mad24(n_idx, M, bm);

#define WRITE_OUTPUT(i) \
  if (bm + 4 > M) { \
    const int diff = M - bm; \
    switch(diff) { \
      case 3: \
        C[out_offset + 2] = CONVERT_TO(c##i.z, DATA_TYPE); \
      case 2: \
        C[out_offset + 1] = CONVERT_TO(c##i.y, DATA_TYPE); \
      case 1: \
        C[out_offset] = CONVERT_TO(c##i.x, DATA_TYPE); \
    } \
    CHECK_OUT_OF_RANGE_FOR_BUFFER(out_offset + diff - 1); \
  } else { \
    VSTORE4(CONVERT_TO(c##i, DATA_TYPE4), C, out_offset); \
  }

  WRITE_OUTPUT(0);

  if ((n_idx + 1) >= N) return;
  out_offset += n_offset_size;
  WRITE_OUTPUT(1);
  
  if ((n_idx + 2) >= N) return;
  out_offset += n_offset_size;
  WRITE_OUTPUT(2);
  
  if ((n_idx + 3) >= N) return;
  out_offset += n_offset_size;
  WRITE_OUTPUT(3);
#undef WRITE_OUTPUT
}

// C = A * B + bias
__kernel void matmul(
    BUFFER_OUT_OF_RANGE_PARAMS
    GLOBAL_WORK_GROUP_SIZE_DIM2
    __global DATA_TYPE *A,  // [1, B, M, K/4], IN_OUT_CHANNEL
    __global DATA_TYPE *B,  // [1, B, K, N/4], IN_OUT_CHANNEL
    __private const int M,
    __private const int N,
    __private const int K,
    __private const int m_blocks,
    __private const int k_blocks,
    __global DATA_TYPE *C  // [1, B, M, N/4], IN_OUT_CHANNEL
) {
  /**
   * gx: col_idx = n_idx
   * ty: row_blk_idx = m
   * gy: batch_base + row_blk_idx = m
   * bm: batch_base + row_idx = m_idx
   * bk: batch_base + depth_blk_idx = k
   * bn: batch_base + col_idx = n_idx
   * pos: depth_blk_idx = k
   */
  const int gx = get_global_id(0) << 2;
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (get_global_id(0) >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  const int batch = hb / m_blocks;
  const int ty = hb - mul24(batch, m_blocks);
  const int m_idx = ty << 2;
  const int gy = mad24(batch, m_blocks, ty);
  const int bm = mad24(batch, M, m_idx);
  const int bk = mul24(batch, k_blocks);
  const int bn = mad24(batch, N, gx);

#if 0
  printf("(%d,%d): batch %d, gx %d, bn %d\n", gx, hb, batch, gx, bn);
#endif

  DATA_TYPE4 c0 = 0;
  DATA_TYPE4 c1 = 0;
  DATA_TYPE4 c2 = 0;
  DATA_TYPE4 c3 = 0;

  DATA_TYPE4 a0, a1, a2, a3;
  DATA_TYPE4 b0, b1, b2, b3;
  for (int k = 0; k < k_blocks; k ++) {
    int k_idx = k << 2;
    // 1 m block and 4 k from A.
    // base = ((batch * M + m_idx) * k_blocks + k) * 4
    //      = (bm * k_blocks + k) * 4
    //      = bm * K + k_idx
    // d = 4
    int a_base_offset = mad24(bm, K, k_idx);
    a0 = CONVERT4(vload4(0, A + a_base_offset));
    a1 = CONVERT4(vload4(0, A + a_base_offset + 4));
    a2 = CONVERT4(vload4(0, A + a_base_offset + 8));
    a3 = CONVERT4(vload4(0, A + a_base_offset + 12));

    // 1 n block and 4 k from B.
    // base = ((batch * K + k_idx) * n_blocks + n) * 4
    //      = (batch * K + k_idx) * N + n_idx
    // d = 1 * n_blocks * 4 = N
    int b_base_offset = mad24(mad24(batch, K, k_idx), N, gx);
    b0 = CONVERT4(vload4(0, B + b_base_offset));
    b1 = CONVERT4(vload4(0, B + b_base_offset + N));
    b2 = CONVERT4(vload4(0, B + b_base_offset + N * 2));
    b3 = CONVERT4(vload4(0, B + b_base_offset + N * 3));

    // TODO(fucheng)
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
  // start = ((batch * N + n) * m_blocks + m) * 4
  //       = bn * M + m_idx
  // d = 1 * m_blocks * 4 = M
  //const int n_offset_size = m_blocks << 2;
  //int out_offset = (bn * m_blocks + gy) << 2;
  const int n_offset_size = M;
  int out_offset = mad24(bn, M, m_idx);

#define WRITE_OUTPUT(i) \
  if (m_idx + 4 > M) { \
    const int diff = M - bm; \
    switch(diff) { \
      case 3: \
        C[out_offset + 2] = CONVERT_TO(c2.i, DATA_TYPE); \
      case 2: \
        C[out_offset + 1] = CONVERT_TO(c1.i, DATA_TYPE); \
      case 1: \
        C[out_offset] = CONVERT_TO(c0.i, DATA_TYPE); \
    } \
    CHECK_OUT_OF_RANGE_FOR_BUFFER(out_offset + diff - 1); \
  } else { \
    C[out_offset] = CONVERT_TO(c0.i, DATA_TYPE); \
    C[out_offset + 1] = CONVERT_TO(c1.i, DATA_TYPE); \
    C[out_offset + 2] = CONVERT_TO(c2.i, DATA_TYPE); \
    C[out_offset + 3] = CONVERT_TO(c3.i, DATA_TYPE); \
  }

  WRITE_OUTPUT(x);

  if ((gx + 1) >= N) return;
  out_offset += n_offset_size;
  WRITE_OUTPUT(y);
  
  if ((gx + 2) >= N) return;
  out_offset += n_offset_size;
  WRITE_OUTPUT(z);
  
  if ((gx + 3) >= N) return;
  out_offset += n_offset_size;
  WRITE_OUTPUT(w);
#undef WRITE_OUTPUT
#endif

  // start = ((batch * M + m_idx) * n_blocks + n) * 4
  //       = bm * N + n_idx
  //       = bm * N + gx
  // d = 1 * n_blocks * 4 = N
  const int m_offset_size = N;
  int out_offset = mad24(bm, N, gx);

#define WRITE_OUTPUT(i) \
  if (gx + 4 > N) { \
    const int diff = N - gx; \
    switch(diff) { \
      case 3: \
        C[out_offset + 2] = CONVERT_TO(c##i.z, DATA_TYPE); \
      case 2: \
        C[out_offset + 1] = CONVERT_TO(c##i.y, DATA_TYPE); \
      case 1: \
        C[out_offset] = CONVERT_TO(c##i.x, DATA_TYPE); \
    } \
    CHECK_OUT_OF_RANGE_FOR_BUFFER(out_offset + diff - 1); \
  } else { \
    VSTORE4(CONVERT_TO(c##i, DATA_TYPE4), C, out_offset); \
  }

  WRITE_OUTPUT(0);

  if ((m_idx + 1) >= M) return;
  out_offset += m_offset_size;
  WRITE_OUTPUT(1);
  
  if ((m_idx + 2) >= M) return;
  out_offset += m_offset_size;
  WRITE_OUTPUT(2);
  
  if ((m_idx + 3) >= M) return;
  out_offset += m_offset_size;
  WRITE_OUTPUT(3);
#undef WRITE_OUTPUT
}

// C = A * B + bias
__kernel void matmul_transpose_b(
    BUFFER_OUT_OF_RANGE_PARAMS
    GLOBAL_WORK_GROUP_SIZE_DIM2
    __global DATA_TYPE *A,  // [1, B, M, K/4], IN_OUT_CHANNEL
    __global DATA_TYPE *B,  // [1, B, N, K/4], IN_OUT_CHANNEL
    __private const int M,
    __private const int N,
    __private const int K,
    __private const int m_blocks,
    __private const int k_blocks,
    __global DATA_TYPE *C  // [1, B, N, M/4], IN_OUT_CHANNEL
) {
  /**
   * gx: col_idx = n_idx
   * ty: row_blk_idx = m
   * gy: batch_base + row_blk_idx = m
   * bm: batch_base + row_idx = m_idx
   * bk: batch_base + depth_blk_idx = k
   * bn: batch_base + col_idx = n_idx
   * pos: depth_blk_idx = k
   */
  const int gx = get_global_id(0) << 2;
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (get_global_id(0) >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  const int batch = hb / m_blocks;
  const int ty = hb - mul24(batch, m_blocks);
  const int m_idx = ty << 2;
  const int gy = mad24(batch, m_blocks, ty);
  const int bm = mad24(batch, M, m_idx);
  const int bk = mul24(batch, k_blocks);
  const int bn = mad24(batch, N, gx);
  const int n_idx = gx;

  DATA_TYPE4 c0 = 0;
  DATA_TYPE4 c1 = 0;
  DATA_TYPE4 c2 = 0;
  DATA_TYPE4 c3 = 0;

  DATA_TYPE4 a0, a1, a2, a3;
  DATA_TYPE4 b0, b1, b2, b3;
  for (int k = 0; k < k_blocks; k ++) {
    int k_idx = k << 2;
    // 1 m block and 4 k from A.
    // base = ((batch * M + m_idx) * k_blocks + k) * 4
    //      = (bm * k_blocks + k) * 4
    //      = bm * K + k_idx
    // d = 4
    int a_base_offset = mad24(bm, K, k_idx);
    a0 = CONVERT4(vload4(0, A + a_base_offset));
    a1 = CONVERT4(vload4(0, A + a_base_offset + 4));
    a2 = CONVERT4(vload4(0, A + a_base_offset + 8));
    a3 = CONVERT4(vload4(0, A + a_base_offset + 12));

    // 4 n and 1 k block from B.
    // base = ((batch * N + n) * k_blocks + k) * 4
    //      = (batch * N + n) * K + k_idx
    //      = bn * K + k_idx
    // d = 1 * k_blocks * 4 = K
    int b_base_offset = mad24(bn, K, k_idx);
    b0 = CONVERT4(vload4(0, B + b_base_offset));
    b1 = CONVERT4(vload4(0, B + b_base_offset + K));
    b2 = CONVERT4(vload4(0, B + b_base_offset + K * 2));
    b3 = CONVERT4(vload4(0, B + b_base_offset + K * 3));

#define CALC_N(i) \
    c##i += (DATA_TYPE4)(dot(a0, b##i), dot(a1, b##i), dot(a2, b##i), dot(a3, b##i));

    CALC_N(0);
    CALC_N(1);
    CALC_N(2);
    CALC_N(3);

#undef CALC_N
  }

#if 0
  // start = ((batch * N + n) * m_blocks + m) * 4
  //       = bn * M + m_idx
  // d = 1 * m_blocks * 4 = M
  const int n_offset_size = M;
  int out_offset = mad24(bn, M, m_idx);

#define WRITE_OUTPUT(i) \
  if (m_idx + 4 > M) { \
    const int diff = M - bm; \
    switch(diff) { \
      case 3: \
        C[out_offset + 2] = CONVERT_TO(c##i.z, DATA_TYPE); \
      case 2: \
        C[out_offset + 1] = CONVERT_TO(c##i.y, DATA_TYPE); \
      case 1: \
        C[out_offset] = CONVERT_TO(c##i.x, DATA_TYPE); \
    } \
    CHECK_OUT_OF_RANGE_FOR_BUFFER(out_offset + diff - 1); \
  } else { \
    VSTORE4(CONVERT_TO(c##i, DATA_TYPE4), C, out_offset); \
  }

  WRITE_OUTPUT(0);

  if ((n_idx + 1) >= N) return;
  out_offset += n_offset_size;
  WRITE_OUTPUT(1);
  
  if ((n_idx + 2) >= N) return;
  out_offset += n_offset_size;
  WRITE_OUTPUT(2);
  
  if ((n_idx + 3) >= N) return;
  out_offset += n_offset_size;
  WRITE_OUTPUT(3);
#undef WRITE_OUTPUT
#endif

  // start = ((batch * M + m_idx) * n_blocks + n) * 4
  //       = bm * N + n_idx
  //       = bm * N + gx
  // d = 1 * n_blocks * 4 = N
  const int m_offset_size = N;
  int out_offset = mad24(bm, N, gx);

#define WRITE_OUTPUT(i) \
  if (gx + 4 > N) { \
    const int diff = N - gx; \
    switch(diff) { \
      case 3: \
        C[out_offset + 2] = CONVERT_TO(c2.i, DATA_TYPE); \
      case 2: \
        C[out_offset + 1] = CONVERT_TO(c1.i, DATA_TYPE); \
      case 1: \
        C[out_offset] = CONVERT_TO(c0.i, DATA_TYPE); \
    } \
    CHECK_OUT_OF_RANGE_FOR_BUFFER(out_offset + diff - 1); \
  } else { \
    C[out_offset] = CONVERT_TO(c0.i, DATA_TYPE); \
    C[out_offset + 1] = CONVERT_TO(c1.i, DATA_TYPE); \
    C[out_offset + 2] = CONVERT_TO(c2.i, DATA_TYPE); \
    C[out_offset + 3] = CONVERT_TO(c3.i, DATA_TYPE); \
  }

  WRITE_OUTPUT(x);

  if ((m_idx + 1) >= M) return;
  out_offset += m_offset_size;
  WRITE_OUTPUT(y);
  
  if ((m_idx + 2) >= M) return;
  out_offset += m_offset_size;
  WRITE_OUTPUT(z);
  
  if ((m_idx + 3) >= M) return;
  out_offset += m_offset_size;
  WRITE_OUTPUT(w);
#undef WRITE_OUTPUT
}
