/*!
Copyright (C) 2017, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <openpose/postprocess/nms.hpp>
#include "convert.hpp"

template <typename _T, typename _TIndex>
std::list<std::tuple<_T, _T, _T> > part_peaks(pybind11::array_t<_T> part, const _T threshold)
{
	typedef Eigen::Tensor<_T, 2, Eigen::RowMajor, _TIndex> _TTensor;
	typedef Eigen::Tensor<const _T, 2, Eigen::RowMajor, _TIndex> _TConstTensor;
	const _TTensor _part = numpy_tensor<_TTensor>(part);
	return openpose::postprocess::part_peaks(Eigen::TensorMap<_TConstTensor, Eigen::Aligned>(_part.data(), _part.dimensions()), threshold);
}

template <typename _T, typename _TIndex>
std::vector<std::vector<std::tuple<_T, _T, _T> > > parts_peaks(pybind11::array_t<_T> parts, const _T threshold)
{
	typedef Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex> _TTensor;
	typedef Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex> _TConstTensor;
	const _TTensor _parts = numpy_tensor<_TTensor>(parts);
	return openpose::postprocess::parts_peaks(Eigen::TensorMap<_TConstTensor, Eigen::Aligned>(_parts.data(), _parts.dimensions()), threshold);
}
