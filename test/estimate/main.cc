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

#include <boost/format.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <openpose/postprocess/nms.hpp>
#include <openpose/postprocess/hungarian.hpp>
#include <openpose/npy.hpp>
#include <openpose/tsv.hpp>
#include <openpose/render.hpp>

template <typename _T>
std::vector<_T> split_numbers(const std::string &str)
{
	boost::tokenizer<> tok(str);
	std::vector<_T> numbers;
	std::transform(tok.begin(), tok.end(), std::back_inserter(numbers), &boost::lexical_cast<_T, std::string>);
	return numbers;
}

int main(void)
{
	typedef uchar _TPixel;
	typedef float _TReal;
	typedef Eigen::Index _TIndex;
	typedef Eigen::Tensor<_TReal, 3, Eigen::RowMajor, _TIndex> _TTensorReal;
	typedef Eigen::Tensor<const _TReal, 3, Eigen::RowMajor, _TIndex> _TConstTensorReal;
	typedef std::pair<_TIndex, _TIndex> _TLimbIndex;
	typedef std::vector<_TLimbIndex> _TLimbsIndex;

	const cv::Mat image = cv::imread(DUMP_DIR "/featuremap/image.jpg", CV_LOAD_IMAGE_COLOR);
	const auto _limbs_index = openpose::load_tsv_paired<_TIndex>(DUMP_DIR "/featuremap/limbs_index.tsv");
	const _TLimbsIndex limbs_index(_limbs_index.begin(), _limbs_index.end());
	const _TTensorReal limbs = openpose::load_npy3<_TReal, _TTensorReal>(DUMP_DIR "/featuremap/limbs.npy");
	const _TTensorReal parts = openpose::load_npy3<_TReal, _TTensorReal>(DUMP_DIR "/featuremap/parts.npy");
	boost::property_tree::ptree pt;
	boost::property_tree::ini_parser::read_ini(DUMP_DIR "/config.ini", pt);
	const _TReal threshold = pt.get<_TReal>("nms.threshold");
	const _TReal step = pt.get<_TReal>("integration.step");
	const std::vector<size_t> _step_limits = split_numbers<size_t>(pt.get<std::string>("integration.step_limits"));
	const std::pair<size_t, size_t> step_limits(_step_limits[0], _step_limits[1]);
	const _TReal min_score = pt.get<_TReal>("integration.min_score");
	const size_t min_count = pt.get<size_t>("integration.min_count");
#ifdef DEBUG_SHOW
	openpose::image_ = image;
	openpose::parts_ = parts;
#endif
	auto peaks = openpose::postprocess::parts_peaks(
		Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(parts.data(), parts.dimensions()),
		threshold
	);
	const auto clusters = openpose::postprocess::clustering(
		peaks,
		Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(limbs.data(), limbs.dimensions()), limbs_index,
		step, step_limits, min_score, min_count
	);
	{
		const auto canvas = openpose::render(
			image, limbs_index,
			Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(parts.data(), parts.dimensions()),
			peaks, clusters
		);
		cv::imshow("", canvas);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	return 0;
}
