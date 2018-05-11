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

#include <cstdint>
#include <boost/format.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <openpose/data/label.hpp>
#include <openpose/npy.hpp>
#include <openpose/tsv.hpp>
#include <openpose/render.hpp>

template <typename _TIndex>
std::string get_title_limbs(const _TIndex index, const _TIndex total)
{
	assert(0 <= index && index < total * 2);
	const std::string xy = index % 2 ? "y" : "x";
	return (boost::format("limb %d/%d ") % (index / 2 + 1) % total).str() + xy;

}

template <typename _TIndex>
std::string get_title_parts(const _TIndex index, const _TIndex total)
{
	assert(0 <= index && index <= total);
	if (index < total)
		return (boost::format("part %d/%d") % (index + 1) % total).str();
	else
		return "background";
}

template <typename _T, typename _TIndex>
void test(
	const std::string &path_image, const std::string &path_keypoints, const std::string &path_limbs_index,
	const std::pair<_TIndex, _TIndex> downsample, const _T sigma_parts, const _T sigma_limbs
)
{
	typedef Eigen::Tensor<_T, 3, Eigen::RowMajor, _TIndex> _TTensor;
	typedef Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex> _TConstTensor;
	typedef std::pair<_TIndex, _TIndex> _TLimbIndex;
	typedef std::vector<_TLimbIndex> _TLimbsIndex;

	const cv::Mat image = cv::imread(path_image, CV_LOAD_IMAGE_COLOR);
	const _TTensor keypoints = openpose::load_npy3<float, _TTensor>(path_keypoints);
	const auto _limbs_index = openpose::load_tsv_paired<_TIndex>(path_limbs_index);
	const _TLimbsIndex limbs_index(_limbs_index.begin(), _limbs_index.end());
	_TTensor _parts(keypoints.dimension(1) + 1, image.rows / downsample.first, image.cols / downsample.second);
	_TTensor _limbs((_TIndex)limbs_index.size() * 2, image.rows / downsample.first, image.cols / downsample.second);
	Eigen::TensorMap<_TConstTensor, Eigen::Aligned> _keypoints(keypoints.data(), keypoints.dimensions());
#if 1
	openpose::data::label_parts(
		_keypoints,
		sigma_parts,
		(_TIndex)image.rows, (_TIndex)image.cols,
		Eigen::TensorMap<_TTensor, Eigen::Aligned>(_parts.data(), _parts.dimensions())
	);
	for (_TIndex index = 0; index < _parts.dimension(0); ++index)
	{
		const cv::Mat canvas = openpose::render(image,
			Eigen::TensorMap<_TConstTensor, Eigen::Aligned>(_parts.data(), _parts.dimensions()),
			index
		);
		cv::imshow(get_title_parts(index, keypoints.dimension(1)), canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
#if 1
	openpose::data::label_limbs(
		_keypoints,
		limbs_index,
		sigma_limbs,
		(_TIndex)image.rows, (_TIndex)image.cols,
		Eigen::TensorMap<_TTensor, Eigen::Aligned>(_limbs.data(), _limbs.dimensions())
	);
	for (_TIndex index = 0; index < _limbs.dimension(0); ++index)
	{
		const cv::Mat canvas = openpose::render(image,
			Eigen::TensorMap<_TConstTensor, Eigen::Aligned>(_limbs.data(), _limbs.dimensions()),
			index
		);
		cv::imshow(get_title_limbs<_TIndex>(index, limbs_index.size()), canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
#endif
}

int main(void)
{
#define IMAGE_EXT ".jpg"
	typedef float _T;
	typedef Eigen::Index _TIndex;
	boost::property_tree::ptree pt;
	boost::property_tree::ini_parser::read_ini(DUMP_DIR "/config.ini", pt);
	const _T sigma_parts = pt.get<_T>("label.sigma_parts");
	const _T sigma_limbs = pt.get<_T>("label.sigma_limbs");
	const std::pair<_TIndex, _TIndex> downsample(8, 8);
	{
		const std::string prefix = std::string(DUMP_DIR) + "/data/COCO_train2014_000000000077";
		test(prefix + IMAGE_EXT, prefix + ".keypoints.npy", DUMP_DIR "/data.tsv", downsample, sigma_parts, sigma_limbs);
	}
	return 0;
}
