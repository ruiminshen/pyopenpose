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

#include <random>
#include <utility>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <exception>
#include <boost/format.hpp>
#include <boost/exception/all.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/opencv.hpp>
#include <openpose/assert.hpp>
#include <openpose/render.hpp>

namespace openpose
{
namespace data
{
template <typename _T, typename _TIndex, int Options>
cv::Rect_<_T> calc_keypoints_rect(Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> keypoints, const _TIndex index, const cv::Size &size)
{
	ASSERT_OPENPOSE(keypoints.dimension(2) == 3);
	_T xmin = std::numeric_limits<_T>::max(), xmax = std::numeric_limits<_T>::min();
	_T ymin = std::numeric_limits<_T>::max(), ymax = std::numeric_limits<_T>::min();
	for (_TIndex i = 0; i < keypoints.dimension(1); ++i)
	{
		if (keypoints(index, i, 2) > 0)
		{
			const _T y = std::min<_T>(std::max<_T>(keypoints(index, i, 0), 0), size.height - 1);
			if (y < ymin)
				ymin = y;
			if (y > ymax)
				ymax = y;
			const _T x = std::min<_T>(std::max<_T>(keypoints(index, i, 1), 0), size.width - 1);
			if (x < xmin)
				xmin = x;
			if (x > xmax)
				xmax = x;
		}
	}
	//ASSERT_OPENPOSE(xmin <= xmax && ymin <= ymax);
	return cv::Rect_<_T>(xmin, ymin, xmax - xmin, ymax - ymin);
}

template <typename _T, typename _TIndex, int Options>
cv::Rect_<_T> calc_points_rect(const std::vector<Eigen::TensorMap<Eigen::Tensor<const _T, 2, Eigen::RowMajor, _TIndex>, Options> > &points, const _TIndex index, const cv::Size &size)
{
	_T xmin = std::numeric_limits<_T>::max(), xmax = std::numeric_limits<_T>::min();
	_T ymin = std::numeric_limits<_T>::max(), ymax = std::numeric_limits<_T>::min();
	for (size_t i = 0; i < points.size(); ++i)
	{
		const _T y = std::min<_T>(std::max<_T>(points[i](index, 0), 0), size.height - 1);
		if (y < ymin)
			ymin = y;
		if (y > ymax)
			ymax = y;
		const _T x = std::min<_T>(std::max<_T>(points[i](index, 1), 0), size.width - 1);
		if (x < xmin)
			xmin = x;
		if (x > xmax)
			xmax = x;
	}
	ASSERT_OPENPOSE(xmin <= xmax && ymin <= ymax);
	return cv::Rect_<_T>(xmin, ymin, xmax - xmin, ymax - ymin);
}

template <typename _T>
class Rotator
{
public:
	typedef _T T;
	typedef double TRotate;

	Rotator(const T angle, const cv::Point_<T> &center, const cv::Size_<T> &size);
	void operator ()(const cv::Mat &image, cv::Mat &image_result, const cv::Scalar &fill) const;
	template <typename _TIndex, int Options> void operator ()(Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor, _TIndex>, Options> keypoints) const;
	template <typename _TIndex, int Options> void operator ()(Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, _TIndex>, Options> bbox) const;

private:
	cv::Mat mat_;
	cv::Rect rect_;
};

template <typename _T>
Rotator<_T>::Rotator(const T angle, const cv::Point_<T> &center, const cv::Size_<T> &size)
	: mat_(cv::getRotationMatrix2D(center, angle, 1.0))
	, rect_(cv::RotatedRect(center, size, angle).boundingRect())
{
	typedef double _TRotate;
	mat_.at<_TRotate>(0, 2) += rect_.width / 2.0 - center.x;
	mat_.at<_TRotate>(1, 2) += rect_.height / 2.0 - center.y;
}

template <typename _T>
void Rotator<_T>::operator ()(const cv::Mat &image, cv::Mat &image_result, const cv::Scalar &fill) const
{
	cv::warpAffine(image, image_result, mat_, rect_.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, fill);
}

template <typename _T>
template <typename _TIndex, int Options> void Rotator<_T>::operator ()(Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor, _TIndex>, Options> keypoints) const
{
	ASSERT_OPENPOSE(keypoints.dimension(2) == 3);
	for (_TIndex i = 0; i < keypoints.dimension(0); ++i)
		for (_TIndex j = 0; j < keypoints.dimension(1); ++j)
			if (keypoints(i, j, 2) > 0)
			{
				cv::Mat point(3, 1, mat_.type());
				point.at<TRotate>(1, 0) = keypoints(i, j, 0);
				point.at<TRotate>(0, 0) = keypoints(i, j, 1);
				point.at<TRotate>(2, 0) = 1;
				point = mat_ * point;
				keypoints(i, j, 0) = point.at<TRotate>(1, 0);
				keypoints(i, j, 1) = point.at<TRotate>(0, 0);
			}
}

template <typename _T>
template <typename _TIndex, int Options> void Rotator<_T>::operator ()(Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, _TIndex>, Options> bbox) const
{
	ASSERT_OPENPOSE(bbox.dimension(1) == 2);
	for (_TIndex i = 0; i < bbox.dimension(0); ++i)
	{
		cv::Mat point(3, 1, mat_.type());
		point.at<TRotate>(1, 0) = bbox(i, 0);
		point.at<TRotate>(0, 0) = bbox(i, 1);
		point.at<TRotate>(2, 0) = 1;
		point = mat_ * point;
		bbox(i, 0) = point.at<TRotate>(1, 0);
		bbox(i, 1) = point.at<TRotate>(0, 0);
	}
}

template <typename _T>
cv::Rect_<_T> calc_bound_size(_T range, const cv::Size &size, const cv::Size &dsize)
{
	if (range <= 0)
		range = std::min(size.width, size.height);
	cv::Rect_<_T> bound;
	if (size.width < size.height)
	{
		bound.width = std::min<_T>(range, size.width);
		bound.height = bound.width * dsize.height / dsize.width;
	}
	else
	{
		bound.height = std::min<_T>(range, size.height);
		bound.width = bound.height * dsize.width / dsize.height;
	}
	ASSERT_OPENPOSE(bound.width <= size.width && bound.height <= size.height);
	return bound;
}

template <typename _TRandom, typename _T>
void update_bound_pos(_TRandom &random, const cv::Rect_<_T> &rect, const cv::Size &size, cv::Rect_<_T> &bound)
{
	ASSERT_OPENPOSE(bound.width <= size.width && bound.height <= size.height);
	const cv::Point_<_T> br = rect.br();
	ASSERT_OPENPOSE(rect.x >= 0 && rect.y >= 0);
	ASSERT_OPENPOSE(br.x < size.width && br.y < size.height);
	cv::Point_<_T> xy1(br.x - bound.width, br.y - bound.height);
	if (xy1.x <= 0)
		xy1.x = 0;
	if (xy1.y <= 0)
		xy1.y = 0;
	cv::Point_<_T> xy2(rect.x, rect.y);
	if (xy2.x + bound.width > size.width)
		xy2.x = size.width - bound.width;
	if (xy2.y + bound.height > size.height)
		xy2.y = size.height - bound.height;
	const std::pair<_T, _T> xrange = std::minmax(xy1.x, xy2.x);
	const std::pair<_T, _T> yrange = std::minmax(xy1.y, xy2.y);
	bound.x = std::uniform_real_distribution<_T>(xrange.first, xrange.second)(random);
	bound.y = std::uniform_real_distribution<_T>(yrange.first, yrange.second)(random);
}

template <typename _T, typename _TIndex, int Options>
void move_scale_keypoints(const cv::Rect_<_T> &bound, const cv::Size &dsize, Eigen::TensorMap<Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex>, Options> keypoints)
{
	ASSERT_OPENPOSE(keypoints.dimension(2) == 3);
	for (_TIndex i = 0; i < keypoints.dimension(0); ++i)
		for (_TIndex j = 0; j < keypoints.dimension(1); ++j)
			if (keypoints(i, j, 2) > 0)
			{
				const _T y = (keypoints(i, j, 0) - bound.y) * dsize.height / bound.height;
				const _T x = (keypoints(i, j, 1) - bound.x) * dsize.width / bound.width;
				if (0 <= x && x < dsize.width && 0 <= y && y < dsize.height)
				{
					keypoints(i, j, 0) = y;
					keypoints(i, j, 1) = x;
				}
				else
					keypoints(i, j, 2) = 0;
			}
}

template <typename _TRandom, typename _TPixel, typename _TReal, typename _TIndex, int Options>
void augmentation(
	_TRandom &random,
	Eigen::TensorMap<Eigen::Tensor<const _TPixel, 3, Eigen::RowMajor, _TIndex>, Options> image,
	Eigen::TensorMap<Eigen::Tensor<const _TPixel, 3, Eigen::RowMajor, _TIndex>, Options> mask,
	Eigen::TensorMap<Eigen::Tensor<const _TReal, 3, Eigen::RowMajor, _TIndex>, Options> keypoints,
	Eigen::TensorMap<Eigen::Tensor<_TReal, 2, Eigen::RowMajor, _TIndex>, Options> yx_min,
	Eigen::TensorMap<Eigen::Tensor<_TReal, 2, Eigen::RowMajor, _TIndex>, Options> yx_max,
	const _TReal scale, const _TReal angle,
	Eigen::TensorMap<Eigen::Tensor<_TPixel, 3, Eigen::RowMajor, _TIndex>, Options> image_result,
	Eigen::TensorMap<Eigen::Tensor<_TPixel, 3, Eigen::RowMajor, _TIndex>, Options> mask_result,
	Eigen::TensorMap<Eigen::Tensor<_TReal, 3, Eigen::RowMajor, _TIndex>, Options> keypoints_result,
	const cv::Scalar &fill, const _TIndex index
)
{
	typedef Eigen::Tensor<const _TReal, 3, Eigen::RowMajor, _TIndex> _TConstTensorReal;
	typedef Eigen::Tensor<const _TReal, 2, Eigen::RowMajor, _TIndex> _TConstTensorReal2;
	typedef Eigen::Tensor<_TReal, 2, Eigen::RowMajor, _TIndex> _TTensorReal2;
	ASSERT_OPENPOSE(scale > 1);
	ASSERT_OPENPOSE(yx_min.dimension(0) == keypoints.dimension(0));
	ASSERT_OPENPOSE(yx_max.dimension(0) == keypoints.dimension(0));
	cv::Mat _image_result = cv::Mat(image.dimension(0), image.dimension(1), CV_8UC(image.dimension(2)), (void *)image.data());
	cv::Mat _mask_result = cv::Mat(mask.dimension(0), mask.dimension(1), CV_8UC(mask.dimension(2)), (void *)mask.data());
	keypoints_result = keypoints;
	_TTensorReal2 _point1 = yx_min, _point2 = yx_max;
	Eigen::TensorMap<_TTensorReal2, Options> point1(_point1.data(), _point1.dimensions()), point2(_point2.data(), _point2.dimensions());
	for (_TIndex i = 0; i < point1.dimension(0); ++i)
	{
		point1(i, 0) = yx_max(i, 0);
		point2(i, 0) = yx_min(i, 0);
	}
#if 1
#ifdef DEBUG_SHOW
	{
		cv::Mat canvas = _image_result.clone();
		draw_mask(canvas, _mask_result);
		draw_keypoints(canvas, Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(keypoints_result.data(), keypoints_result.dimensions()), index);
		draw_points(canvas, Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(yx_min.data(), yx_min.dimensions()), index);
		draw_points(canvas, Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(yx_max.data(), yx_max.dimensions()), index);
		draw_points(canvas, Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(point1.data(), point1.dimensions()), index);
		draw_points(canvas, Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(point2.data(), point2.dimensions()), index);
		cv::imshow("bbox_points", canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
#endif
	const Rotator<_TReal> rotator(angle, cv::Point_<_TReal>(_image_result.cols / 2.0, _image_result.rows / 2.0), _image_result.size());
	rotator(_image_result, _image_result, fill);
	rotator(_mask_result, _mask_result, cv::Scalar(0, 0, 0));
	rotator(keypoints_result);
	rotator(yx_min);
	rotator(yx_max);
	rotator(point1);
	rotator(point2);
	const cv::Size size(_image_result.cols, _image_result.rows);
	const cv::Size dsize(image_result.dimension(1), image_result.dimension(0));
	const cv::Rect_<_TReal> rect = calc_points_rect(
		std::vector<Eigen::TensorMap<_TConstTensorReal2, Options> >({
			Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(yx_min.data(), yx_min.dimensions()),
			Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(yx_max.data(), yx_max.dimensions()),
			Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(point1.data(), point1.dimensions()),
			Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(point2.data(), point2.dimensions())
		}),
		index, size
	);
#if 1
#ifdef DEBUG_SHOW
	{
		cv::Mat canvas = _image_result.clone();
		draw_mask(canvas, _mask_result);
		draw_keypoints(canvas, Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(keypoints_result.data(), keypoints_result.dimensions()), index);
		draw_points(canvas, Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(yx_min.data(), yx_min.dimensions()), index);
		draw_points(canvas, Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(yx_max.data(), yx_max.dimensions()), index);
		draw_points(canvas, Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(point1.data(), point1.dimensions()), index);
		draw_points(canvas, Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(point2.data(), point2.dimensions()), index);
		cv::rectangle(canvas, rect, CV_RGB(255, 255, 255));
		cv::imshow("center_rotate", canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
#endif
	const _TReal range = std::max(rect.width, rect.height);
	cv::Rect_<_TReal> bound = calc_bound_size(range * scale, size, dsize);
	update_bound_pos(random, rect, size, bound);
	_image_result = _image_result(bound);
	_mask_result = _mask_result(bound);
#if 1
#ifdef DEBUG_SHOW
	{
		cv::Mat canvas = _image_result.clone();
		draw_mask(canvas, _mask_result);
		cv::imshow("crop", canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
#endif
	resize(_image_result, _image_result, dsize, 0, 0, cv::INTER_CUBIC);
	resize(_mask_result, _mask_result, cv::Size(mask_result.dimension(1), mask_result.dimension(0)), 0, 0, cv::INTER_CUBIC);
	move_scale_keypoints(bound, dsize, keypoints_result);
#if 1
#ifdef DEBUG_SHOW
	{
		cv::Mat canvas = _image_result.clone();
		draw_mask(canvas, _mask_result);
		draw_keypoints(canvas, Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(keypoints_result.data(), keypoints_result.dimensions()), index);
		cv::imshow("scale", canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
#endif
	std::copy(_image_result.data, _image_result.data + image_result.size(), image_result.data());
	std::copy(_mask_result.data, _mask_result.data + mask_result.size(), mask_result.data());
}

template <typename _TRandom, typename _TPixel, typename _TReal, typename _TIndex, int Options>
_TIndex augmentation(
	_TRandom &random,
	Eigen::TensorMap<Eigen::Tensor<const _TPixel, 3, Eigen::RowMajor, _TIndex>, Options> image,
	Eigen::TensorMap<Eigen::Tensor<const _TPixel, 3, Eigen::RowMajor, _TIndex>, Options> mask,
	Eigen::TensorMap<Eigen::Tensor<const _TReal, 3, Eigen::RowMajor, _TIndex>, Options> keypoints,
	Eigen::TensorMap<Eigen::Tensor<_TReal, 2, Eigen::RowMajor, _TIndex>, Options> yx_min,
	Eigen::TensorMap<Eigen::Tensor<_TReal, 2, Eigen::RowMajor, _TIndex>, Options> yx_max,
	const _TReal scale, const _TReal angle,
	Eigen::TensorMap<Eigen::Tensor<_TPixel, 3, Eigen::RowMajor, _TIndex>, Options> image_result,
	Eigen::TensorMap<Eigen::Tensor<_TPixel, 3, Eigen::RowMajor, _TIndex>, Options> mask_result,
	Eigen::TensorMap<Eigen::Tensor<_TReal, 3, Eigen::RowMajor, _TIndex>, Options> keypoints_result,
	const cv::Scalar &fill
)
{
	const _TIndex index = std::uniform_int_distribution<_TIndex>(0, keypoints_result.dimension(0) - 1)(random);
	augmentation(random,
		image, mask, keypoints,
		yx_min, yx_max,
		scale, angle,
		image_result, mask_result, keypoints_result,
		fill, index);
	return index;
}
}
}
