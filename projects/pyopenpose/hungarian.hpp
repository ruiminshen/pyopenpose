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
#include <openpose/postprocess/hungarian.hpp>
#include "convert.hpp"

template <typename _T, typename _TIndex>
std::list<std::tuple<_TIndex, _TIndex, _T> > calc_limb_score(
	pybind11::array_t<_T> limb1, pybind11::array_t<_T> limb2,
	const _TIndex channel, pybind11::array_t<_T> limbs,
	std::vector<std::tuple<_T, _T, _T> > &peaks1, std::vector<std::tuple<_T, _T, _T> > &peaks2,
	const _T step, const std::pair<size_t, size_t> &step_limits, const _T min_score, const size_t min_count
)
{
	typedef Eigen::Tensor<_T, 2, Eigen::RowMajor, _TIndex> _TTensor;
	typedef Eigen::Tensor<const _T, 2, Eigen::RowMajor, _TIndex> _TConstTensor;
	const _TTensor _limb1 = numpy_tensor<_TTensor>(limb1);
	const _TTensor _limb2 = numpy_tensor<_TTensor>(limb2);
	return openpose::postprocess::calc_limb_score(
		Eigen::TensorMap<_TConstTensor, Eigen::Aligned>(_limb1.data(), _limb1.dimensions()),
		Eigen::TensorMap<_TConstTensor, Eigen::Aligned>(_limb2.data(), _limb2.dimensions()),
		peaks1, peaks2,
		step, step_limits, min_score, min_count
	);
}

template <typename _T, typename _TIndex>
std::list<std::tuple<std::vector<_TIndex>, _T, _TIndex> > clustering(
	std::vector<std::vector<std::tuple<_T, _T, _T> > > &peaks,
	pybind11::array_t<_T> limbs, const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index,
	const _T step, const std::pair<size_t, size_t> &step_limits, const _T min_score, const size_t min_count
)
{
	typedef Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex> _TTensor;
	typedef Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex> _TConstTensor;
	const _TTensor _limbs = numpy_tensor<_TTensor>(limbs);
	return openpose::postprocess::clustering(
		peaks,
		Eigen::TensorMap<_TConstTensor, Eigen::Aligned>(_limbs.data(), _limbs.dimensions()), limbs_index,
		step, step_limits, min_score, min_count
	);
}
