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

#include <ctime>
#include <random>
#include <boost/format.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <openpose/data/augmentation.hpp>
#include <openpose/data/bbox.hpp>
#include <openpose/npy.hpp>
#include <openpose/tsv.hpp>
#include <openpose/render.hpp>

template <typename _TPixel, typename _TReal, typename _TIndex, typename _TRandom>
void test(
	_TRandom &random,
	const std::string &prefix, const std::string &path_limbs_index,
	const _TIndex height, const _TIndex width,
	const std::pair<_TIndex, _TIndex> downsample,
	const _TIndex bbox_height, const _TIndex bbox_width,
	const _TReal scale, const _TReal angle, const _TReal scale_bbox,
	const cv::Scalar &fill,
	_TIndex index = -1
)
{
	typedef Eigen::Tensor<_TPixel, 3, Eigen::RowMajor, _TIndex> _TTensorPixel;
	typedef Eigen::Tensor<_TReal, 3, Eigen::RowMajor, _TIndex> _TTensorReal;
	typedef Eigen::Tensor<_TReal, 2, Eigen::RowMajor, _TIndex> _TTensorReal2;
	typedef Eigen::Tensor<const _TPixel, 3, Eigen::RowMajor, _TIndex> _TConstTensorPixel;
	typedef Eigen::Tensor<const _TReal, 3, Eigen::RowMajor, _TIndex> _TConstTensorReal;
	typedef Eigen::Tensor<const _TReal, 2, Eigen::RowMajor, _TIndex> _TConstTensorReal2;
	typedef std::pair<_TIndex, _TIndex> _TLimbIndex;
	typedef std::vector<_TLimbIndex> _TLimbsIndex;

	const cv::Mat image = cv::imread(prefix + ".jpg", CV_LOAD_IMAGE_COLOR);
	Eigen::TensorMap<_TConstTensorPixel, Eigen::Aligned> _image(image.data, typename _TTensorPixel::Dimensions(image.rows, image.cols, image.channels()));
	const cv::Mat mask = cv::imread(prefix + ".mask.jpg", cv::IMREAD_GRAYSCALE);
	Eigen::TensorMap<_TConstTensorPixel, Eigen::Aligned> _mask(mask.data, typename _TTensorPixel::Dimensions(mask.rows, mask.cols, mask.channels()));
	const _TTensorReal _keypoints = openpose::load_npy3<float, _TTensorReal>(prefix + ".keypoints.npy");
	_TTensorReal2 _yx_min = openpose::load_npy2<float, _TTensorReal2>(prefix + ".yx_min.npy");
	_TTensorReal2 _yx_max = openpose::load_npy2<float, _TTensorReal2>(prefix + ".yx_max.npy");
	const auto _limbs_index = openpose::load_tsv_paired<_TIndex>(path_limbs_index);
	const _TLimbsIndex limbs_index(_limbs_index.begin(), _limbs_index.end());
	_TTensorPixel _image_result(height, width, _image.dimension(2));
	_TTensorPixel _mask_result(height / downsample.first, width / downsample.second, _mask.dimension(2));
	_TTensorReal _keypoints_result(_keypoints.dimensions());
	_TTensorReal _xy_offset(bbox_height, bbox_width, 2);
	_TTensorReal _width_height(bbox_height, bbox_width, 2);
#if 1
	{
		cv::Mat canvas = image.clone();
		openpose::draw_mask(canvas, mask);
		Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned> kps(_keypoints.data(), _keypoints.dimensions());
		openpose::draw_keypoints(canvas, kps);
		openpose::draw_skeleton(canvas, kps, limbs_index);
		openpose::draw_bbox(
			canvas,
			Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(_yx_min.data(), _yx_min.dimensions()),
			Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(_yx_max.data(), _yx_max.dimensions())
		);
		cv::imshow("original", canvas);
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
#endif
	if (index != -1)
	{
		assert(0 <= index && index < _keypoints.dimension(0));
		openpose::data::augmentation(
			random,
			_image, _mask,
			Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(_keypoints.data(), _keypoints.dimensions()),
			Eigen::TensorMap<_TTensorReal2, Eigen::Aligned>(_yx_min.data(), _yx_min.dimensions()),
			Eigen::TensorMap<_TTensorReal2, Eigen::Aligned>(_yx_max.data(), _yx_max.dimensions()),
			scale, angle,
			Eigen::TensorMap<_TTensorPixel, Eigen::Aligned>(_image_result.data(), _image_result.dimensions()),
			Eigen::TensorMap<_TTensorPixel, Eigen::Aligned>(_mask_result.data(), _mask_result.dimensions()),
			Eigen::TensorMap<_TTensorReal, Eigen::Aligned>(_keypoints_result.data(), _keypoints_result.dimensions()),
			fill, index
		);
	}
	else
		index = openpose::data::augmentation(
			random,
			_image, _mask,
			Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(_keypoints.data(), _keypoints.dimensions()),
			Eigen::TensorMap<_TTensorReal2, Eigen::Aligned>(_yx_min.data(), _yx_min.dimensions()),
			Eigen::TensorMap<_TTensorReal2, Eigen::Aligned>(_yx_max.data(), _yx_max.dimensions()),
			scale, angle,
			Eigen::TensorMap<_TTensorPixel, Eigen::Aligned>(_image_result.data(), _image_result.dimensions()),
			Eigen::TensorMap<_TTensorPixel, Eigen::Aligned>(_mask_result.data(), _mask_result.dimensions()),
			Eigen::TensorMap<_TTensorReal, Eigen::Aligned>(_keypoints_result.data(), _keypoints_result.dimensions()),
			fill
		);
	openpose::data::keypoints_bbox(height, width, scale_bbox, Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(_keypoints_result.data(), _keypoints_result.dimensions()), Eigen::TensorMap<_TTensorReal, Eigen::Aligned>(_xy_offset.data(), _xy_offset.dimensions()), Eigen::TensorMap<_TTensorReal, Eigen::Aligned>(_width_height.data(), _width_height.dimensions()));
	{
		cv::Mat canvas = image.clone();
		openpose::draw_mask(canvas, mask);
		Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned> kps(_keypoints.data(), _keypoints.dimensions());
		openpose::draw_keypoints(canvas, kps);
		openpose::draw_skeleton(canvas, kps, limbs_index);
		cv::imshow("original", canvas);
	}
	{
		cv::Mat canvas(_image_result.dimension(0), _image_result.dimension(1), CV_8UC(_image_result.dimension(2)), (void *)_image_result.data());
		cv::Mat mask(_mask_result.dimension(0), _mask_result.dimension(1), CV_8UC(_mask_result.dimension(2)), (void *)_mask_result.data());
		Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned> kps(_keypoints_result.data(), _keypoints_result.dimensions());
		openpose::draw_mask(canvas, mask);
		openpose::draw_keypoints(canvas, kps, index);
		openpose::draw_skeleton(canvas, kps, limbs_index);
		cv::imshow("keypoints", canvas);
	}
	/*
	{
		const cv::Mat canvas = openpose::render(
			cv::Mat(_image_result.dimension(0), _image_result.dimension(1), CV_8UC(_image_result.dimension(2)), (void *)_image_result.data()),
			cv::Mat(_mask_result.dimension(0), _mask_result.dimension(1), CV_8UC(_mask_result.dimension(2)), (void *)_mask_result.data()),
			Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(_keypoints_result.data(), _keypoints_result.dimensions()),
			Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(_xy_offset.data(), _xy_offset.dimensions()),
			Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(_width_height.data(), _width_height.dimensions()));
		cv::imshow("bbox", canvas);
	}*/
	cv::waitKey(0);
	cv::destroyAllWindows();
}

int main(void)
{
	typedef std::mt19937 _TRandom;
	typedef uchar _TPixel;
	typedef float _TReal;
	typedef Eigen::Index _TIndex;
#ifdef NDEBUG
	_TRandom random(std::time(0));
#else
	_TRandom random;
#endif
	const std::string path_limbs_index = DUMP_DIR "/data.tsv";
	const std::pair<_TIndex, _TIndex> downsample(8, 8);
	const _TIndex bbox_height = 7, bbox_width = 7;
	const _TReal scale_bbox = 1.5;
	{
		const std::string prefix = std::string(DUMP_DIR) + "/data/COCO_train2014_000000000086";
		const _TReal scale = 1;
		const _TReal angle = 0;
		test<_TPixel, _TReal, _TIndex>(random, prefix, path_limbs_index,
			368, 368, downsample, bbox_height, bbox_width, scale, angle, scale_bbox, cv::Scalar(128, 128, 128));
	}
	{
		const std::string prefix = std::string(DUMP_DIR) + "/data/COCO_train2014_000000000077";
		const _TReal scale = std::uniform_real_distribution<_TReal>(1, 1.5)(random);
		const _TReal angle = std::uniform_real_distribution<_TReal>(-40, 40)(random);
		test<_TPixel, _TReal, _TIndex>(random, prefix, path_limbs_index,
			368, 368, downsample, bbox_height, bbox_width, scale, (_TReal)10, scale_bbox, cv::Scalar(0, 0, 0));
		test<_TPixel, _TReal, _TIndex>(random, prefix, path_limbs_index,
			368, 368, downsample, bbox_height, bbox_width, (_TReal)1000, angle, scale_bbox, cv::Scalar(0, 0, 0));
		test<_TPixel, _TReal, _TIndex>(random, prefix, path_limbs_index,
			184, 368, downsample, bbox_height, bbox_width, scale, angle, scale_bbox, cv::Scalar(255, 255, 255));
		test<_TPixel, _TReal, _TIndex>(random, prefix, path_limbs_index,
			368, 184, downsample, bbox_height, bbox_width, scale, angle, scale_bbox, cv::Scalar(255, 255, 255));
	}
	{
		const std::string prefix = std::string(DUMP_DIR) + "/data/COCO_val2014_000000000136";
		const _TReal scale = 1.4529;
		const _TReal angle = 26.8007;
		test<_TPixel, _TReal, _TIndex>(random, prefix, path_limbs_index,
			368, 368, downsample, bbox_height, bbox_width, scale, angle, scale_bbox, cv::Scalar(128, 128, 128), 1);
	}
	{
		const std::string prefix = std::string(DUMP_DIR) + "/data/COCO_val2014_000000000241";
		const _TReal scale = 1.99419;
		const _TReal angle = -3.95667;
		test<_TPixel, _TReal, _TIndex>(random, prefix, path_limbs_index,
			368, 368, downsample, bbox_height, bbox_width, scale, angle, scale_bbox, cv::Scalar(128, 128, 128), 0);
	}
	{
		const std::string prefix = std::string(DUMP_DIR) + "/data/COCO_train2014_000000000036";
		const _TReal scale = 1.38945;
		const _TReal angle = -2.13689;
		test<_TPixel, _TReal, _TIndex>(random, prefix, path_limbs_index,
			368, 368, downsample, bbox_height, bbox_width, scale, angle, scale_bbox, cv::Scalar(128, 128, 128));
	}
	return 0;
}
