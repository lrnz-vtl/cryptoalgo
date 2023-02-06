#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

typedef unsigned long int ulong;

py::array_t<double> shift_forward(py::array_t<unsigned long int>, py::array_t<double>, ulong);

class EMA {
    ulong m_last_t;
    double m_last_value, m_decay_scale;

    public:
    EMA(double);
    double updated_value(ulong, double);
};

class ExpSum {
    double m_last_value, m_alpha;

    public:
    ExpSum(double);
    double updated_value(double);
};
