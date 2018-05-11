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

#include <vector>
#include <list>
#include <tuple>
#include <cmath>
#include <thread>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>

namespace openpose
{
namespace postprocess
{
template <typename _T, typename _TIndex, int Options>
std::list<std::tuple<_T, _T, _T> > part_peaks(
	Eigen::TensorMap<Eigen::Tensor<const _T, 2, Eigen::RowMajor, _TIndex>, Options> part,
	const _T threshold
)
{
	typedef std::tuple<_T, _T, _T> _TPeak;
	typedef std::list<_TPeak> _TPeaks;
	const _TIndex size_y = part.dimension(1);
	_TPeaks peaks;
	const _TIndex _height = part.dimension(0) - 1, _width = part.dimension(1) - 1;
	for (_TIndex y = 1; y < _height; ++y)
	{
		const _TIndex cache_part_iy = size_y * y;
		for (_TIndex x = 1; x < _width; ++x)
		{
			const _TIndex index = cache_part_iy + x;
			const _T value = part.data()[index];
			if (value < threshold)
				continue;
			const _T up = part.data()[index - size_y];
			const _T down = part.data()[index + size_y];
			const _T left = part.data()[index - 1];
			const _T right = part.data()[index + 1];
			if(value > up && value > down && value > left && value > right)
				peaks.push_back(std::make_tuple(y, x, value));
		}
	}
	return peaks;
}

template <typename _T>
std::vector<std::tuple<_T, _T, _T> > sort_peaks(const std::list<std::tuple<_T, _T, _T> > &peaks)
{
	typedef std::tuple<_T, _T, _T> _TPeak;
	typedef std::list<_TPeak> _TPeaks;
	typedef typename _TPeaks::const_iterator _TIterator;
	std::vector<_TIterator> sorted(peaks.size());
	{
		size_t index = 0;
		for (_TIterator i = peaks.begin(); i != peaks.end(); ++i)
		{
			sorted[index] = i;
			++index;
		}
	}
	std::sort(sorted.begin(), sorted.end(), [](_TIterator a, _TIterator b)->bool{return std::get<2>(*a) > std::get<2>(*b);});
	std::vector<_TPeak> _peaks(sorted.size());
	for (size_t i = 0; i < sorted.size(); ++i)
		_peaks[i] = *sorted[i];
	return _peaks;
}

template <typename _T, typename _TIndex, int Options>
std::vector<std::vector<std::tuple<_T, _T, _T> > > parts_peaks(
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> parts, const _T threshold
)
{
	typedef std::tuple<_T, _T, _T> _TPeak;
	typedef std::vector<_TPeak> _TPeaks;
	std::vector<_TPeaks> peaks(parts.dimension(0));
	const _TIndex size = parts.dimension(1) * parts.dimension(2);
	for (size_t channel = 0; channel < peaks.size(); ++channel)
	{
		Eigen::TensorMap<Eigen::Tensor<const _T, 2, Eigen::RowMajor, _TIndex>, Options> part(
			parts.data() + channel * size,
			parts.dimension(1), parts.dimension(2)
		);
		auto _peaks = part_peaks(part, threshold);
		peaks[channel] = sort_peaks(_peaks);
	}
	return peaks;
}

template <typename _T, typename _TIndex, int Options>
std::vector<std::vector<std::tuple<_T, _T, _T> > > parts_peaks_mt(
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> parts, const _T threshold
)
{
	typedef std::tuple<_T, _T, _T> _TPeak;
	typedef std::vector<_TPeak> _TPeaks;
	std::vector<_TPeaks> peaks(parts.dimension(0));
	const _TIndex size = parts.dimension(1) * parts.dimension(2);
	const unsigned parallel = std::thread::hardware_concurrency();
	const size_t core_channels = std::ceil((_T)peaks.size() / parallel);
	std::list<std::thread> threads;
	for (unsigned core = 0; core < parallel; ++core)
	{
		const size_t begin = core * core_channels;
		const size_t end = std::min((core + 1) * core_channels, peaks.size());
		for (size_t channel = begin; channel < end; ++channel)
			threads.push_back(std::thread([&peaks, &parts, channel, size, threshold](){
				Eigen::TensorMap<Eigen::Tensor<const _T, 2, Eigen::RowMajor, _TIndex>, Options> part(
					parts.data() + channel * size,
					parts.dimension(1), parts.dimension(2)
				);
				auto _peaks = part_peaks(part, threshold);
				peaks[channel] = sort_peaks(_peaks);
			}));
	}
	for (auto i = threads.begin(); i != threads.end(); ++i)
		i->join();
	return peaks;
}
}
}
