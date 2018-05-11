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

#include <cassert>
#include <cmath>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>

namespace openpose
{
namespace data
{
template <typename _T, typename _TIndex, int Options>
void label_parts(
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> keypoints,
	const _T sigma,
	const _TIndex height, const _TIndex width,
	Eigen::TensorMap<Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex>, Options> label,
	const _T threshold = -log(0.01)
)
{
	typedef Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> _TMatrixCount;
	assert(keypoints.dimension(1) + 1 == label.dimension(0));
	std::memset(label.data(), 0, label.size() * sizeof(_T));
	const _T grid_height = (_T)height / label.dimension(1);
	const _T grid_width = (_T)width / label.dimension(2);
	const _T grid_height2 = grid_height / 2;
	const _T grid_width2 = grid_width / 2;
	const _T sigma2 = sigma * sigma;
	for (_TIndex gy = 0; gy < label.dimension(1); ++gy)
	{
		const _T y = gy * grid_height + grid_height2;
		for (_TIndex gx = 0; gx < label.dimension(2); ++gx)
		{
			const _T x = gx * grid_width + grid_width2;
			_T maximum = 0;
			for (_TIndex part = 0; part < keypoints.dimension(1); ++part)
			{
				_T &value = label(part, gy, gx);
				for (_TIndex index = 0; index < keypoints.dimension(0); ++index)
				{
					if (keypoints(index, part, 2) > 0)
					{
						const _T diff_y = keypoints(index, part, 0) - y;
						const _T diff_x = keypoints(index, part, 1) - x;
						const _T exponent = (diff_x * diff_x + diff_y * diff_y) / 2 / sigma2;
						if(exponent > threshold)
							continue;
						value += exp(-exponent);
						value = std::min<_T>(value, 1);
						maximum = std::max<_T>(value, maximum);
					}
				}
				assert(0 <= value <= 1);
			}
			assert(0 <= maximum <= 1);
			label(keypoints.dimension(1), gy, gx) = 1 - maximum;
		}
	}
}

template <typename _T>
std::pair<_T, _T> calc_norm_vec(const _T x, const _T y)
{
	const _T len = sqrt(x * x + y * y);
	return std::make_pair(x / len, y / len);
}

template <typename _T, typename _TIndex, int Options>
void label_limbs(
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> keypoints,
	const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index,
	const _T sigma,
	const _TIndex height, const _TIndex width,
	Eigen::TensorMap<Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex>, Options> label
)
{
	typedef Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> _TMatrixCount;
	std::memset(label.data(), 0, label.size() * sizeof(_T));
	const _T grid_height = (_T)height / label.dimension(1);
	const _T grid_width = (_T)width / label.dimension(2);
	const _T grid_height2 = grid_height / 2;
	const _T grid_width2 = grid_width / 2;
	for (_TIndex limb = 0; limb < limbs_index.size(); ++limb)
	{
		const std::pair<_TIndex, _TIndex> &temp = limbs_index[limb];
		const _TIndex p1 = temp.first;
		const _TIndex p2 = temp.second;
		const _TIndex channel_x = limb * 2;
		const _TIndex channel_y = channel_x + 1;
		_TMatrixCount count = _TMatrixCount::Constant(label.dimension(1), label.dimension(2), 0);
		for (_TIndex index = 0; index < keypoints.dimension(0); ++index)
			if (keypoints(index, p1, 2) > 0 && keypoints(index, p2, 2) > 0)
			{
				const _T p1y = keypoints(index, p1, 0) / grid_height, p1x = keypoints(index, p1, 1) / grid_width;
				const _T p2y = keypoints(index, p2, 0) / grid_height, p2x = keypoints(index, p2, 1) / grid_width;
				const std::pair<_T, _T> norm_vec = calc_norm_vec(p2x - p1x, p2y - p1y);
				const _TIndex gy_min = round(std::min(p1y, p2y) - sigma);
				const _TIndex gy_max = round(std::max(p1y, p2y) + sigma);
				const _TIndex gx_min = round(std::min(p1x, p2x) - sigma);
				const _TIndex gx_max = round(std::max(p1x, p2x) + sigma);
				for (_TIndex gy = std::max<_TIndex>(gy_min, 0); gy < std::min<_TIndex>(gy_max, label.dimension(1)); ++gy)
					for (_TIndex gx = std::max<_TIndex>(gx_min, 0); gx < std::min<_TIndex>(gx_max, label.dimension(2)); ++gx)
					{
						const _T dist = std::abs((gx - p1x) * norm_vec.second - (gy - p1y) * norm_vec.first);
						if (dist <= sigma)
						{
							label(channel_x, gy, gx) += norm_vec.first;
							label(channel_y, gy, gx) += norm_vec.second;
							count(gy, gx) += 1;
						}
					}
			}
		for (_TIndex gy = 0; gy < label.dimension(1); ++gy)
			for (_TIndex gx = 0; gx < label.dimension(2); ++gx)
			{
				const size_t c = count(gy, gx);
				if (c > 0)
				{
					label(channel_x, gy, gx) /= c;
					label(channel_y, gy, gx) /= c;
				}
			}
	}
}
}
}
