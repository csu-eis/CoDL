
#include <unistd.h>

#include "mace/utils/logging.h"
#include "test/fucheng/conv2d_part_plan.h"

#define StringToFloat(s) strtof(s, NULL)

typedef struct {
    float part_radio;
} TestParam;

int ParseArg(int argc, char *argv[], TestParam *param_out) {
    int ch;
    while ((ch = getopt(argc, argv, "r:")) != -1) {
        switch (ch) {
            case 'r':
                param_out->part_radio = StringToFloat(optarg);
                break;
            case '?':
            default:
                break;
        }
    }
    
    return 0;
}

void PrintPlanInputInfo(const float radio,
                        const std::vector<index_t> &input_shape,
                        const std::vector<index_t> &filter_shape,
                        const std::vector<int> &strides,
                        const std::vector<index_t> &output_shape) {
    printf("Radio: %.2f\n", radio);
#define PRINT_SHAPE(name, var)        \
    printf("%s: [%ld,%ld,%ld,%ld]\n", \
            name, var.at(0), var.at(1), var.at(2), var.at(3));
            
    PRINT_SHAPE("Input Shape", input_shape)
    PRINT_SHAPE("Filter Shape", filter_shape)
    printf("Strides: [%d,%d]\n", strides.at(0), strides.at(1));
    PRINT_SHAPE("Output Shape", output_shape)
}

std::vector<index_t>* ComputeConv2dOutputShape(
    const std::vector<index_t> input_shape,
    const std::vector<index_t> filter_shape,
    const std::vector<int> strides) {
    
    std::vector<index_t> *output_shape = new std::vector<index_t>(4);
    MACE_CHECK(output_shape->size() == 4);
    
    (*output_shape)[0] = 1;
    (*output_shape)[1] = (input_shape[1] - filter_shape[2]) / strides[0] + 1;
    (*output_shape)[2] = (input_shape[2] - filter_shape[3]) / strides[1] + 1;
    (*output_shape)[3] = filter_shape[0];
    
    return output_shape;
}

int Conv2dPartPlanTestSample1(TestParam *test_param) {
    
    printf("===== Sample 1 ======\n");
    
    std::vector<index_t> input_shape{1, 114, 114, 64};
    std::vector<index_t> filter_shape{64, 64, 3, 3};
    std::vector<int> strides{2, 2};
    std::vector<index_t> output_shape = *ComputeConv2dOutputShape(input_shape,
                                                                  filter_shape,
                                                                  strides);
                                                                  
    PrintPlanInputInfo(test_param->part_radio, input_shape, filter_shape, strides, output_shape);
    
    Conv2dPartPlan plan(test_param->part_radio);
    plan.Make(input_shape, filter_shape, strides, output_shape);
    plan.Show();
    
    return 0;
}

int Conv2dPartPlanTestSample2(TestParam *test_param) {
    
    printf("===== Sample 2 ======\n");
    
    std::vector<index_t> input_shape{1, 56, 56, 64};
    std::vector<index_t> filter_shape{256, 64, 1, 1};
    std::vector<int> strides{1, 1};
    std::vector<index_t> output_shape = *ComputeConv2dOutputShape(input_shape,
                                                                  filter_shape,
                                                                  strides);
                                                                  
    PrintPlanInputInfo(test_param->part_radio, input_shape, filter_shape, strides, output_shape);
    
    Conv2dPartPlan plan(test_param->part_radio);
    plan.Make(input_shape, filter_shape, strides, output_shape);
    plan.Show();
    
    return 0;
}

int Conv2dPartPlanTestSample3(TestParam *test_param) {
    
    printf("===== Sample 3 ======\n");
    
    std::vector<index_t> input_shape{1, 28, 28, 512};
    std::vector<index_t> filter_shape{1024, 512, 1, 1};
    std::vector<int> strides{2, 2};
    std::vector<index_t> output_shape = *ComputeConv2dOutputShape(input_shape,
                                                                  filter_shape,
                                                                  strides);
                                                                  
    PrintPlanInputInfo(test_param->part_radio, input_shape, filter_shape, strides, output_shape);
    
    Conv2dPartPlan plan(test_param->part_radio);
    plan.Make(input_shape, filter_shape, strides, output_shape);
    plan.Show();
    
    return 0;
}

int main(int argc, char *argv[]) {
    
    TestParam test_param;
    
    ParseArg(argc, argv, &test_param);
    
    Conv2dPartPlanTestSample1(&test_param);
    Conv2dPartPlanTestSample2(&test_param);
    Conv2dPartPlanTestSample3(&test_param);
    
    return 0;
}
