#include <gtest/gtest.h>
#include "series.h"

// namespace py = pybind11;

TEST(DummyTest, DummyTestCase)
{

    auto x = ExponentialSum(1.0);
    auto v1 = x.updated_value(1, 1.0);

    EXPECT_EQ(v1, 1);
    auto v2 = x.updated_value(2, -1.0);

    EXPECT_DOUBLE_EQ(v2, -0.63212055882855767);

    // py::array_t<double> future_xs = py::array_t<double>(10);
    // py::buffer_info future_xs_buf = future_xs.request();
    // double *future_xs_ptr = static_cast<double *>(future_xs_buf.ptr);   
 }

int main() {
    // PyGILState_STATE gstate;
    // gstate = PyGILState_Ensure();

    testing::InitGoogleTest();
    auto ret = RUN_ALL_TESTS();

    // PyGILState_Release(gstate);
    return ret;
}