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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <openpose/postprocess/estimate.hpp>
#include "convert.hpp"

template <typename _T, typename _TIndex>
std::list<std::list<std::pair<std::tuple<_TIndex, _T, _T>, std::tuple<_TIndex, _T, _T> > > > estimate(
	pybind11::array_t<_T> parts,
	pybind11::array_t<_T> limbs, const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index,
	const _T threshold, const _T step, const std::pair<size_t, size_t> &step_limits, const _T min_score, const size_t min_count, const _T cluster_min_score, const size_t cluster_min_count
)
{
	typedef Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex> _TTensor;
	typedef Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex> _TConstTensor;
	const _TTensor _parts = numpy_tensor<_TTensor>(parts);
	const _TTensor _limbs = numpy_tensor<_TTensor>(limbs);
	return openpose::postprocess::estimate(
		Eigen::TensorMap<_TConstTensor, Eigen::Aligned>(_parts.data(), _parts.dimensions()),
		Eigen::TensorMap<_TConstTensor, Eigen::Aligned>(_limbs.data(), _limbs.dimensions()), limbs_index,
		threshold, step, step_limits, min_score, min_count, cluster_min_score, cluster_min_count
	);
}

template <typename _T, typename _TIndex>
std::list<std::list<std::pair<std::tuple<_TIndex, _T, _T>, std::tuple<_TIndex, _T, _T> > > > estimate_mt(
	pybind11::array_t<_T> parts,
	pybind11::array_t<_T> limbs, const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index,
	const _T threshold, const _T step, const std::pair<size_t, size_t> &step_limits, const _T min_score, const size_t min_count, const _T cluster_min_score, const size_t cluster_min_count
)
{
	typedef Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex> _TTensor;
	typedef Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex> _TConstTensor;
	const _TTensor _parts = numpy_tensor<_TTensor>(parts);
	const _TTensor _limbs = numpy_tensor<_TTensor>(limbs);
	return openpose::postprocess::estimate_mt(
		Eigen::TensorMap<_TConstTensor, Eigen::Aligned>(_parts.data(), _parts.dimensions()),
		Eigen::TensorMap<_TConstTensor, Eigen::Aligned>(_limbs.data(), _limbs.dimensions()), limbs_index,
		threshold, step, step_limits, min_score, min_count, cluster_min_score, cluster_min_count
	);
}
