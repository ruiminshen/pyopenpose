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
#include <unsupported/Eigen/CXX11/Tensor>
#include <benchmark/benchmark.h>
#include <openpose/postprocess/nms.hpp>
#include <openpose/postprocess/hungarian.hpp>
#include <openpose/postprocess/estimate.hpp>
#include <openpose/npy.hpp>
#include <openpose/tsv.hpp>
#include <openpose/render.hpp>

typedef uchar _TPixel;
typedef float _TReal;
typedef Eigen::Index _TIndex;
typedef Eigen::Tensor<_TReal, 3, Eigen::RowMajor, _TIndex> _TTensorReal;
typedef Eigen::Tensor<const _TReal, 3, Eigen::RowMajor, _TIndex> _TConstTensorReal;
typedef Eigen::Tensor<const _TReal, 2, Eigen::RowMajor, _TIndex> _TConstTensorReal2;
typedef std::pair<_TIndex, _TIndex> _TLimbIndex;
typedef std::vector<_TLimbIndex> _TLimbsIndex;

const cv::Mat image = cv::imread(DUMP_DIR "/featuremap/image.jpg", CV_LOAD_IMAGE_COLOR);
const auto _limbs_index = openpose::load_tsv_paired<_TIndex>(DUMP_DIR "/featuremap/limbs_index.tsv");
const _TLimbsIndex limbs_index(_limbs_index.begin(), _limbs_index.end());
const _TTensorReal limbs = openpose::load_npy3<_TReal, _TTensorReal>(DUMP_DIR "/featuremap/limbs.npy");
const _TTensorReal parts = openpose::load_npy3<_TReal, _TTensorReal>(DUMP_DIR "/featuremap/parts.npy");
const _TReal threshold = 0.05;
const _TReal step = 5;
const std::pair<size_t, size_t> step_limits(5, 25);
const _TReal min_score = 0.05;
const size_t min_count = 6;
const _TReal cluster_min_score = 0.4;
const size_t cluster_min_count = 3;

static void part_peaks(benchmark::State &state)
{
	while (state.KeepRunning())
		openpose::postprocess::part_peaks(Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(parts.data(), parts.dimension(1), parts.dimension(2)), threshold);
}

BENCHMARK(part_peaks);

static void sort_peaks(benchmark::State &state)
{
	const auto peaks_ = openpose::postprocess::part_peaks(Eigen::TensorMap<_TConstTensorReal2, Eigen::Aligned>(parts.data(), parts.dimension(1), parts.dimension(2)), threshold);
	while (state.KeepRunning())
	{
		auto _peaks = peaks_;
		openpose::postprocess::sort_peaks(_peaks);
	}
}

BENCHMARK(sort_peaks);

static void parts_peaks(benchmark::State &state)
{
	while (state.KeepRunning())
		openpose::postprocess::parts_peaks(Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(parts.data(), parts.dimensions()), threshold);
}

BENCHMARK(parts_peaks);

static void parts_peaks_mt(benchmark::State &state)
{
	while (state.KeepRunning())
		openpose::postprocess::parts_peaks_mt(Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(parts.data(), parts.dimensions()), threshold);
}

BENCHMARK(parts_peaks_mt);

static void clustering(benchmark::State &state)
{
	const auto peaks_ = openpose::postprocess::parts_peaks(Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(parts.data(), parts.dimensions()), threshold);
	while (state.KeepRunning())
	{
		auto peaks = peaks_;
		openpose::postprocess::clustering(peaks, Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(limbs.data(), limbs.dimensions()), limbs_index, step, step_limits, min_score, min_count);
	}
}

BENCHMARK(clustering);

static void estimate(benchmark::State &state)
{
	while (state.KeepRunning())
		openpose::postprocess::estimate(Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(parts.data(), parts.dimensions()), Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(limbs.data(), limbs.dimensions()), limbs_index, threshold, step, step_limits, min_score, min_count, cluster_min_score, cluster_min_count);
}

BENCHMARK(estimate);

static void estimate_mt(benchmark::State &state)
{
	while (state.KeepRunning())
		openpose::postprocess::estimate(Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(parts.data(), parts.dimensions()), Eigen::TensorMap<_TConstTensorReal, Eigen::Aligned>(limbs.data(), limbs.dimensions()), limbs_index, threshold, step, step_limits, min_score, min_count, cluster_min_score, cluster_min_count);
}

BENCHMARK(estimate_mt);

BENCHMARK_MAIN();
