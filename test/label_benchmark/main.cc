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
#include <unsupported/Eigen/CXX11/Tensor>
#include <benchmark/benchmark.h>
#include <openpose/data/label.hpp>
#include <openpose/npy.hpp>
#include <openpose/tsv.hpp>
#include <openpose/render.hpp>

typedef uchar _TPixel;
typedef int32_t _TInteger;
typedef float _TReal;
typedef Eigen::Index _TIndex;
typedef Eigen::Tensor<_TReal, 3, Eigen::RowMajor, _TIndex> _TTensor;
typedef Eigen::Tensor<const _TReal, 3, Eigen::RowMajor, _TIndex> _TConstTensor;
typedef std::pair<_TIndex, _TIndex> _TLimbIndex;
typedef std::vector<_TLimbIndex> _TLimbsIndex;

const _TReal sigma_parts = 7;
const _TReal sigma_limbs = 7;
const std::pair<_TIndex, _TIndex> downsample(8, 8);
const std::string prefix = std::string(DUMP_DIR) + "/data/COCO_train2014_000000000077";

const cv::Mat image = cv::imread(prefix + ".jpg", CV_LOAD_IMAGE_COLOR);
const _TTensor keypoints = openpose::load_npy3<_TReal, _TTensor>(prefix + ".keypoints.npy");
const auto _limbs_index = openpose::load_tsv_paired<_TIndex>(DUMP_DIR "/data.tsv");
const _TLimbsIndex limbs_index(_limbs_index.begin(), _limbs_index.end());

_TTensor _parts(keypoints.dimension(1) + 1, image.rows / downsample.first, image.cols / downsample.second);
_TTensor _limbs((_TIndex)limbs_index.size() * 2, image.rows / downsample.first, image.cols / downsample.second);
Eigen::TensorMap<_TConstTensor, Eigen::Aligned> _keypoints(keypoints.data(), keypoints.dimensions());

static void label_parts(benchmark::State &state)
{
	while (state.KeepRunning())
		openpose::data::label_parts(_keypoints,
			sigma_parts,
			(_TIndex)image.rows, (_TIndex)image.cols,
			Eigen::TensorMap<_TTensor, Eigen::Aligned>(_parts.data(), _parts.dimensions())
		);
}

BENCHMARK(label_parts);

static void label_limbs(benchmark::State &state)
{
	while (state.KeepRunning())
		openpose::data::label_limbs(
			_keypoints,
			limbs_index,
			sigma_limbs,
			(_TIndex)image.rows, (_TIndex)image.cols,
			Eigen::TensorMap<_TTensor, Eigen::Aligned>(_limbs.data(), _limbs.dimensions())
		);
}

BENCHMARK(label_limbs);

BENCHMARK_MAIN();
