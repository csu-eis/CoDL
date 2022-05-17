
#ifndef TEST_FUCHENG_GEMM_TEST_H_
#define TEST_FUCHENG_GEMM_TEST_H_

#ifdef MACE_BUILD_LIBRARY
class TestParam {};
#else
#include "test/fucheng/test_param.h"
#endif

int jni_main(TestParam *param);

#endif // TEST_FUCHENG_GEMM_TEST_H_
