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
#include <boost/format.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <openpose/data/label.hpp>
#include "convert.hpp"

template <typename _T, typename _TIndex>
pybind11::array_t<_T> label_parts(
	pybind11::array_t<_T> keypoints,
	const _T sigma,
	const _TIndex height, const _TIndex width,
	const _TIndex rows, const _TIndex cols
)
{
	typedef Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex> _TTensor;
	typedef Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex> _TConstTensor;
	const _TTensor _keypoints = numpy_tensor<_TTensor>(keypoints);
	_TTensor _label(_keypoints.dimension(1) + 1, rows, cols);
	openpose::data::label_parts(
		Eigen::TensorMap<_TConstTensor, Eigen::Aligned>(_keypoints.data(), _keypoints.dimensions()),
		sigma,
		height, width,
		Eigen::TensorMap<_TTensor, Eigen::Aligned>(_label.data(), _label.dimensions())
	);
#if 0
	std::cout << _label.dimension(0) << 'x' << _label.dimension(1) << 'x' << _label.dimension(2) << std::endl;
	for (_TIndex i = 0; i < _label.dimension(0); ++i)
	{
		const _TIndex height = _label.dimension(1), width = _label.dimension(2);
		const _TIndex size = height * width;
		cv::Mat canvas(height, width, CV_32FC1, (void *)(_label.data() + i * size));
		cv::imshow((boost::format("channel%d") % i).str(), canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
	return pybind11::array_t<_T>(_label.dimensions(), _label.data());
}

template <typename _T, typename _TIndex>
pybind11::array_t<_T> label_limbs(
	pybind11::array_t<_T> keypoints,
	const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index,
	const _T sigma,
	const _TIndex height, const _TIndex width,
	const _TIndex rows, const _TIndex cols
)
{
	typedef Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex> _TTensor;
	typedef Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex> _TConstTensor;
	const _TTensor _keypoints = numpy_tensor<_TTensor>(keypoints);
	_TTensor _label((_TIndex)limbs_index.size() * 2, rows, cols);
	openpose::data::label_limbs(
		Eigen::TensorMap<_TConstTensor, Eigen::Aligned>(_keypoints.data(), _keypoints.dimensions()),
		limbs_index,
		sigma,
		height, width,
		Eigen::TensorMap<_TTensor, Eigen::Aligned>(_label.data(), _label.dimensions())
	);
#if 0
	std::cout << _label.dimension(0) << 'x' << _label.dimension(1) << 'x' << _label.dimension(2) << std::endl;
	for (_TIndex i = 0; i < _label.dimension(0); ++i)
	{
		const _TIndex height = _label.dimension(1), width = _label.dimension(2);
		const _TIndex size = height * width;
		cv::Mat canvas(height, width, CV_32FC1, (void *)(_label.data() + i * size));
		cv::imshow((boost::format("channel%d") % i).str(), canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
	return pybind11::array_t<_T>(_label.dimensions(), _label.data());
}
