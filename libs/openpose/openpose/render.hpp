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
#include <tuple>
#include <cmath>
#include <type_traits>
#include <boost/format.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/opencv.hpp>

namespace openpose
{
void draw_mask(cv::Mat &canvas, const cv::Mat &mask, const uchar threshold = 128);

template <typename _T, typename _TIndex, int Options>
void draw_points(
	cv::Mat &mat,
	Eigen::TensorMap<Eigen::Tensor<const _T, 2, Eigen::RowMajor, _TIndex>, Options> points,
	const _TIndex index = -1
)
{
	for (_TIndex i = 0; i < points.dimension(0); ++i)
	{
		const auto color = i == index ? CV_RGB(255, 0, 0) : CV_RGB(0, 0, 255);
		cv::rectangle(mat, cv::Point(points(i, 1) - 3, points(i, 0) - 3), cv::Point(points(i, 1) + 3, points(i, 0) + 3), color, CV_FILLED);
	}
}

template <typename _T, typename _TIndex, int Options>
void draw_keypoints(cv::Mat &mat,
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> keypoints,
	const _TIndex index = -1
)
{
	for (_TIndex i = 0; i < keypoints.dimension(0); ++i)
	{
		for (_TIndex j = 0; j < keypoints.dimension(1); ++j)
			if (keypoints(i, j, 2) > 0)
			{
				const cv::Point org(keypoints(i, j, 1), keypoints(i, j, 0));
				const std::string text = (boost::format("%d") % j).str();
				cv::putText(mat, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255), 1, 20);
				const auto color = i == index ? CV_RGB(255, 0, 0) : CV_RGB(0, 0, 255);
				cv::circle(mat, org, 3, color, -1);
			}
	}
}

template <typename _T, typename _TIndex, int Options>
void draw_skeleton(
	cv::Mat &mat,
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> keypoints,
	const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index
)
{
	typedef std::pair<_TIndex, _TIndex> _TLimbIndex;
	for (_TIndex i = 0; i < keypoints.dimension(0); ++i)
	{
		for (size_t n = 0; n < limbs_index.size(); ++n)
		{
			const _TLimbIndex &limb_index = limbs_index[n];
			if (keypoints(i, limb_index.first, 2) > 0 && keypoints(i, limb_index.second, 2) > 0)
				cv::line(mat, cv::Point(keypoints(i, limb_index.first, 1), keypoints(i, limb_index.first, 0)), cv::Point(keypoints(i, limb_index.second, 1), keypoints(i, limb_index.second, 0)), CV_RGB(255,255,255));
		}
	}
}

template <typename _T, typename _TIndex, int Options>
void draw_grid(cv::Mat &mat, Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> bbox)
{
	assert(bbox.dimension(2) == 2);
	const _T bbox_height = (_T)mat.rows / bbox.dimension(0), bbox_width = (_T)mat.cols / bbox.dimension(1);
	for (_TIndex i = 0; i < bbox.dimension(0); ++i)
	{
		const _T y = i * bbox_height;
		cv::line(mat, cv::Point(0, y), cv::Point(mat.cols - 1, y), CV_RGB(255,255,255));
	}
	for (_TIndex j = 0; j < bbox.dimension(1); ++j)
	{
		const _T x = j * bbox_width;
		cv::line(mat, cv::Point(x, 0), cv::Point(x, mat.rows - 1), CV_RGB(255,255,255));
	}
}

template <typename _T, typename _TIndex, int Options>
void draw_bbox(
	cv::Mat &mat,
	Eigen::TensorMap<Eigen::Tensor<const _T, 2, Eigen::RowMajor, _TIndex>, Options> yx_min,
	Eigen::TensorMap<Eigen::Tensor<const _T, 2, Eigen::RowMajor, _TIndex>, Options> yx_max
)
{
	static const std::vector<cv::Scalar> colors = {
		CV_RGB(255, 0, 0),
		CV_RGB(0, 255, 0),
		CV_RGB(0, 0, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(255, 0, 255),
		CV_RGB(255, 255, 0),
		CV_RGB(127, 0, 0),
		CV_RGB(0, 127, 0),
		CV_RGB(0, 0, 127),
		CV_RGB(0, 127, 127),
		CV_RGB(127, 0, 127),
		CV_RGB(127, 127, 0)
	};

	ASSERT_OPENPOSE(yx_min.dimension(0) == yx_max.dimension(0));
	ASSERT_OPENPOSE(yx_min.dimension(1) == 2);
	ASSERT_OPENPOSE(yx_max.dimension(1) == 2);
	for (_TIndex i = 0; i < yx_min.dimension(0); ++i)
	{
		const auto &color = colors[i % colors.size()];
		cv::rectangle(mat, cv::Point(yx_min(i, 1), yx_min(i, 0)), cv::Point(yx_max(i, 1), yx_max(i, 0)), color);
	}
}

template <typename _T, typename _TIndex, int Options>
void draw_bbox(
	cv::Mat &mat,
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> xy_offset,
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> width_height
)
{
	static const std::vector<cv::Scalar> colors = {
		CV_RGB(255, 0, 0),
		CV_RGB(0, 255, 0),
		CV_RGB(0, 0, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(255, 0, 255),
		CV_RGB(255, 255, 0),
		CV_RGB(127, 0, 0),
		CV_RGB(0, 127, 0),
		CV_RGB(0, 0, 127),
		CV_RGB(0, 127, 127),
		CV_RGB(127, 0, 127),
		CV_RGB(127, 127, 0)
	};

	ASSERT_OPENPOSE(xy_offset.dimension(0) == width_height.dimension(0) && xy_offset.dimension(1) == width_height.dimension(1));
	assert(xy_offset.dimension(2) == 2);
	assert(width_height.dimension(2) == 2);
	const _T bbox_height = mat.rows / width_height.dimension(0), bbox_width = mat.cols / width_height.dimension(1);
	cv::Mat mat_grid = mat.clone();
	for (_TIndex i = 0; i < width_height.dimension(0); ++i)
		for (_TIndex j = 0; j < width_height.dimension(1); ++j)
		{
			const auto &color = colors[(i * width_height.dimension(1) + j) % colors.size()];
			if (width_height(i, j, 0) > 0 && width_height(i, j, 1) > 0)
			{
				const _T bbox_y = i * bbox_height, bbox_x = j * bbox_width;
				cv::rectangle(mat_grid, cv::Point(bbox_x, bbox_y), cv::Point(bbox_x + bbox_width, bbox_y + bbox_height), color, -1);
				const _T y = bbox_y + xy_offset(i, j, 1), x = bbox_x + xy_offset(i, j, 0);
				cv::rectangle(mat, cv::Point(x - 3, y - 3), cv::Point(x + 3, y + 3), color, CV_FILLED);
				const _T width2 = width_height(i, j, 0) / 2, height2 = width_height(i, j, 1) / 2;
				cv::rectangle(mat, cv::Point(x - width2, y - height2), cv::Point(x + width2, y + height2), color);
			}
			else
				assert(width_height(i, j, 0) == 0 && width_height(i, j, 1) == 0);
		}
	cv::addWeighted(mat, 0.7, mat_grid, 0.3, 0, mat);
}

template <typename _T, typename _TIndex, int Options>
cv::Mat render(const cv::Mat &image, Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> label, const _TIndex index)
{
	cv::Mat mat(label.dimension(1), label.dimension(2), CV_8UC1);
	for (_TIndex i = 0; i < mat.rows; ++i)
		for (_TIndex j = 0; j < mat.cols; ++j)
		{
			assert(0 <= label(index, i, j) <= 1);
			mat.at<uchar>(i, j) = label(index, i, j) * 255;
		}
	cv::resize(mat, mat, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
	applyColorMap(mat, mat, cv::COLORMAP_JET);
	cv::Mat canvas;
	cv::addWeighted(image, 0.5, mat, 0.5, 0.0, canvas);
	return canvas;
}

template <typename _T, typename _TIndex, int Options>
cv::Mat render(
	const cv::Mat &image,
	const std::pair<_TIndex, _TIndex> &limb_index,
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> parts,
	const std::vector<std::vector<std::tuple<_T, _T, _T> > > &peaks,
	const std::list<std::tuple<_TIndex, _TIndex, _T> > &connections,
	const bool enable_text = true
)
{
	static const std::vector<cv::Scalar> colors = {
		CV_RGB(255, 0, 0),
		CV_RGB(0, 255, 0),
		CV_RGB(0, 0, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(255, 0, 255),
		CV_RGB(255, 255, 0),
		CV_RGB(127, 0, 0),
		CV_RGB(0, 127, 0),
		CV_RGB(0, 0, 127),
		CV_RGB(0, 127, 127),
		CV_RGB(127, 0, 127),
		CV_RGB(127, 127, 0)
	};

	assert(image.rows > 0 && image.cols > 0);
	assert(parts.dimension(1) > 0 && parts.dimension(2) > 0);

	cv::Mat canvas = image.clone();
	for (auto c = connections.begin(); c != connections.end(); ++c)
	{
		const auto &connection = *c;
		const auto &peak1 = peaks[limb_index.first][std::get<0>(connection)];
		const auto &peak2 = peaks[limb_index.second][std::get<1>(connection)];
		if (std::get<2>(peak1) > 0 && std::get<2>(peak2) > 0)
		{
			const _T y1 = std::get<0>(peak1) * image.rows / parts.dimension(1), x1 = std::get<1>(peak1) * image.cols / parts.dimension(2);
			const _T y2 = std::get<0>(peak2) * image.rows / parts.dimension(1), x2 = std::get<1>(peak2) * image.cols / parts.dimension(2);
			cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2), CV_RGB(0, 0, 0), 3);
			if (enable_text)
			{
				cv::putText(canvas, (boost::format("%d") % limb_index.first).str(), cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[limb_index.first % colors.size()], 1, 20);
				cv::putText(canvas, (boost::format("%d") % limb_index.second).str(), cv::Point(x2, y2), cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[limb_index.second % colors.size()], 1, 20);
			}
		}
	}
	return canvas;
}

template <typename _T, typename _TIndex, int Options>
cv::Mat render(
	const cv::Mat &image,
	const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index,
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> parts,
	const std::vector<std::vector<std::tuple<_T, _T, _T> > > &peaks,
	const std::list<std::tuple<std::vector<_TIndex>, _T, _TIndex> > &clusters,
	const bool enable_text = true
)
{
	static const std::vector<cv::Scalar> colors = {
		CV_RGB(255, 0, 0),
		CV_RGB(0, 255, 0),
		CV_RGB(0, 0, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(255, 0, 255),
		CV_RGB(255, 255, 0),
		CV_RGB(127, 0, 0),
		CV_RGB(0, 127, 0),
		CV_RGB(0, 0, 127),
		CV_RGB(0, 127, 127),
		CV_RGB(127, 0, 127),
		CV_RGB(127, 127, 0)
	};

	assert(image.rows > 0 && image.cols > 0);
	assert(parts.dimension(1) > 0 && parts.dimension(2) > 0);

	cv::Mat canvas = image.clone();
	size_t index = 0;
	for (auto c = clusters.begin(); c != clusters.end(); ++c)
	{
		const auto &cluster = *c;
		const std::vector<_TIndex> &points = std::get<0>(cluster);
		assert(points.size() == parts.dimension(0));
		const auto &color = colors[index % colors.size()];
		for (size_t l = 0; l < limbs_index.size(); ++l)
		{
			const std::pair<_TIndex, _TIndex> &limb_index = limbs_index[l];
			const _TIndex p1 = points[limb_index.first], p2 = points[limb_index.second];
			if (p1 >= 0 && p2 >= 0)
			{
				const auto &_p1 = peaks[limb_index.first][p1];
				const _T y1 = std::get<0>(_p1) * image.rows / parts.dimension(1), x1 = std::get<1>(_p1) * image.cols / parts.dimension(2);
				const auto &_p2 = peaks[limb_index.second][p2];
				const _T y2 = std::get<0>(_p2) * image.rows / parts.dimension(1), x2 = std::get<1>(_p2) * image.cols / parts.dimension(2);
				cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2), color, 3);
				if (enable_text)
				{
					cv::putText(canvas, (boost::format("%d") % limb_index.first).str(), cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, 20);
					cv::putText(canvas, (boost::format("%d") % limb_index.second).str(), cv::Point(x2, y2), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, 20);
				}
			}
		}
		index += 1;
	}
	return canvas;
}

template <typename _T>
cv::Mat render(const cv::Mat &image, const std::vector<std::vector<std::tuple<_T, _T, _T> > > &peaks)
{
	static const std::vector<cv::Scalar> colors = {
		CV_RGB(255, 0, 0),
		CV_RGB(0, 255, 0),
		CV_RGB(0, 0, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(255, 0, 255),
		CV_RGB(255, 255, 0),
		CV_RGB(127, 0, 0),
		CV_RGB(0, 127, 0),
		CV_RGB(0, 0, 127),
		CV_RGB(0, 127, 127),
		CV_RGB(127, 0, 127),
		CV_RGB(127, 127, 0)
	};
	cv::Mat canvas = image.clone();
	for (size_t index = 0; index < peaks.size(); ++index)
	{
		const auto &color = colors[index % colors.size()];
		const auto &points = peaks[index];
		for (size_t i = 0; i < points.size(); ++i)
		{
			const auto &point = points[i];
			if (std::get<2>(point) > 0)
				cv::circle(canvas, cv::Point(std::get<1>(point), std::get<0>(point)), 3, color, -1);
			else
				cv::circle(canvas, cv::Point(std::get<1>(point), std::get<0>(point)), 3, color);
		}
	}
	return canvas;
}
}
