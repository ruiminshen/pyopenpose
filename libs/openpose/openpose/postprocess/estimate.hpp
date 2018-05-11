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

#include <utility>
#include <vector>
#include <list>
#include "nms.hpp"
#include "hungarian.hpp"

namespace openpose
{
namespace postprocess
{
template <typename _T, typename _TIndex, int Options>
std::list<std::list<std::pair<std::tuple<_TIndex, _T, _T>, std::tuple<_TIndex, _T, _T> > > > estimate(
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> parts,
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> limbs,
	const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index,
	const _T threshold, const _T step, const std::pair<size_t, size_t> &step_limits, const _T min_score, const size_t min_count, const _T cluster_min_score, const size_t cluster_min_count
)
{
	assert(parts.dimension(1) == limbs.dimension(1));
	assert(parts.dimension(2) == limbs.dimension(2));
	auto peaks = parts_peaks(parts, threshold);
	auto clusters = clustering(peaks, limbs, limbs_index, step, step_limits, min_score, min_count);
	return filter_cluster(peaks, limbs_index, clusters, cluster_min_score, cluster_min_count);
}

template <typename _T, typename _TIndex>
std::list<std::list<std::pair<std::tuple<_TIndex, _T, _T>, std::tuple<_TIndex, _T, _T> > > > estimate(
	const _T *parts, const _T *limbs,
	const _TIndex rows, const _TIndex cols, const _TIndex num_parts, const _TIndex num_limbs,
	const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index,
	const _T threshold, const _T step, const std::pair<size_t, size_t> &step_limits, const _T min_score, const size_t min_count, const _T cluster_min_score, const size_t cluster_min_count
)
{
	typedef Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex> _TConstTensor;
	typedef Eigen::TensorMap<_TConstTensor, Eigen::Aligned> _TConstTensorMap;
	typedef typename _TConstTensorMap::Dimensions _TDimensions;
	_TConstTensorMap _parts(parts, _TDimensions(num_parts, rows, cols));
	_TConstTensorMap _limbs(limbs, _TDimensions(num_limbs, rows, cols));
	return estimate(_parts, _limbs, limbs_index, threshold, step, step_limits, min_score, min_count, cluster_min_score, cluster_min_count);
}

template <typename _T, typename _TIndex, int Options>
std::list<std::list<std::pair<std::tuple<_TIndex, _T, _T>, std::tuple<_TIndex, _T, _T> > > > estimate_mt(
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> parts,
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> limbs,
	const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index,
	const _T threshold, const _T step, const std::pair<size_t, size_t> &step_limits, const _T min_score, const size_t min_count, const _T cluster_min_score, const size_t cluster_min_count
)
{
	assert(parts.dimension(1) == limbs.dimension(1));
	assert(parts.dimension(2) == limbs.dimension(2));
	auto peaks = parts_peaks_mt(parts, threshold);
	auto clusters = clustering(peaks, limbs, limbs_index, step, step_limits, min_score, min_count);
	return filter_cluster(peaks, limbs_index, clusters, cluster_min_score, cluster_min_count);
}

template <typename _T, typename _TIndex>
std::list<std::list<std::pair<std::tuple<_TIndex, _T, _T>, std::tuple<_TIndex, _T, _T> > > > estimate_mt(
	const _T *parts, const _T *limbs,
	const _TIndex rows, const _TIndex cols, const _TIndex num_parts, const _TIndex num_limbs,
	const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index,
	const _T threshold, const _T step, const std::pair<size_t, size_t> &step_limits, const _T min_score, const size_t min_count, const _T cluster_min_score, const size_t cluster_min_count
)
{
	typedef Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex> _TConstTensor;
	typedef Eigen::TensorMap<_TConstTensor, Eigen::Aligned> _TConstTensorMap;
	typedef typename _TConstTensorMap::Dimensions _TDimensions;
	_TConstTensorMap _parts(parts, _TDimensions(num_parts, rows, cols));
	_TConstTensorMap _limbs(limbs, _TDimensions(num_limbs, rows, cols));
	return estimate_mt(_parts, _limbs, limbs_index, threshold, step, step_limits, min_score, min_count, cluster_min_score, cluster_min_count);
}
}
}
