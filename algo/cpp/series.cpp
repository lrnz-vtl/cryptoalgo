#include <iostream>
#include "series.h"
#include <math.h>
#include <pybind11/stl.h>

namespace py = pybind11;



EMA::EMA(double decay_scale) : m_decay_scale(decay_scale) {
    m_last_t = 0;
    m_last_value = std::numeric_limits<double>::quiet_NaN();
}

double EMA::updated_value(ulong t, double value) {
    if (std::isnan(value)) {
        std::cout << "value is nan" << std::endl;
        return m_last_value;
    }

    double decay_factor = exp(-double(t-m_last_t)/m_decay_scale);

    if (std::isnan(m_last_value)) {
        m_last_value = value;
    } else {
        m_last_value = value * (1-decay_factor) + m_last_value * decay_factor;
    }
    m_last_t = t;
    return m_last_value;
}

ExpSum::ExpSum(double alpha) : m_alpha(alpha) {
    m_last_value = std::numeric_limits<double>::quiet_NaN();
}

double ExpSum::updated_value(double value) {
    if (std::isnan(value)) {
        std::cout << "value is nan" << std::endl;
        return m_last_value;
    }

    if (std::isnan(m_last_value)) {
        m_last_value = value;
    } else {
        m_last_value = value  + m_alpha * m_last_value;
    }
    return m_last_value;
}


py::array_t<double> compute_ema(py::array_t<ulong> ts, py::array_t<double> xs, double decay_scale)
{
    py::buffer_info ts_buf = ts.request();
    unsigned long int *ts_ptr = static_cast<unsigned long int *>(ts_buf.ptr);

    py::buffer_info xs_buf = xs.request();
    double *xs_ptr = static_cast<double *>(xs_buf.ptr);

    if (ts_buf.size != xs_buf.size) {
        throw std::runtime_error("Input shapes must match");
    }

    auto ret = py::array_t<double>(ts_buf.size);
    auto ptr = static_cast<double *> (ret.request().ptr);
    auto ema = EMA(decay_scale);

    for(auto i=0; i<ts_buf.size; i++) {
        ptr[i] = ema.updated_value(ts_ptr[i], xs_ptr[i]);
    }
    return ret;
}

py::array_t<double> compute_expsum(py::array_t<double> xs, double alpha)
{
    py::buffer_info xs_buf = xs.request();
    double *xs_ptr = static_cast<double *>(xs_buf.ptr);

    auto ret = py::array_t<double>(xs_buf.size);
    auto ptr = static_cast<double *> (ret.request().ptr);
    auto expsum = ExpSum(alpha);

    for(auto i=0; i<xs_buf.size; i++) {
        ptr[i] = expsum.updated_value(xs_ptr[i]);
    }
    return ret;
}



py::array_t<double> shift_forward(py::array_t<ulong> ts, py::array_t<double> xs, ulong delta) {
    py::buffer_info ts_buf = ts.request();
    unsigned long int *ts_ptr = static_cast<ulong *>(ts_buf.ptr);

    py::buffer_info xs_buf = xs.request();
    double *xs_ptr = static_cast<double *>(xs_buf.ptr);

    if (ts_buf.size != xs_buf.size) {
        throw std::runtime_error("Input shapes must match");
    }

    py::array_t<double> future_xs = py::array_t<double>(xs_buf.size);
    py::buffer_info future_xs_buf = future_xs.request();
    double *future_xs_ptr = static_cast<double *>(future_xs_buf.ptr);

    int j = 0;
    for(auto i=0; i<ts_buf.size; i++) {
        while ((j < ts_buf.size - 1) and (ts_ptr[j] - ts_ptr[i] < delta)) {
            j += 1;
        }
        future_xs_ptr[i] = xs_ptr[j];
    }
    return future_xs;
}



PYBIND11_MODULE(cseries, m) {
    m.doc() = "timeseries plugin";

    m.def("shift_forward", &shift_forward, "shift time series forward");

    m.def("compute_ema", &compute_ema, "compute_ema");

    m.def("compute_expsum", &compute_expsum, "compute_expsum");
}
