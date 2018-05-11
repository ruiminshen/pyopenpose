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
#include <vector>
#include <list>
#include <tuple>
#include <set>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <boost/format.hpp>
#ifdef DEBUG_SHOW
#include <boost/filesystem.hpp>
#include <openpose/debug_global.hpp>
#include <openpose/render.hpp>
#endif

namespace openpose
{
namespace postprocess
{
template <typename _TIndex>
bool check_unique(std::vector<std::pair<_TIndex, _TIndex> > limbs_index)
{
	std::sort(limbs_index.begin(), limbs_index.end());
	return std::unique(limbs_index.begin(), limbs_index.end()) == limbs_index.end();
}

template <typename _TIndex>
size_t limbs_points(const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index)
{
	if (limbs_index.empty())
		throw std::runtime_error("limbs_index is empty");
	if (!check_unique(limbs_index))
		throw std::runtime_error("duplicated limbs found");
	std::set<_TIndex> set;
	set.insert(limbs_index.front().first);
	set.insert(limbs_index.front().second);
	for (size_t i = 1; i < limbs_index.size(); ++i)
	{
		const auto &limb_index = limbs_index[i];
		if (set.find(limb_index.first) == set.end())
			throw std::runtime_error((boost::format("first limb part %d not found in part set {%d}") % limb_index.first % boost::algorithm::join(set | boost::adaptors::transformed(static_cast<std::string(*)(_TIndex)>(std::to_string)), ", ")).str());
		set.insert(limb_index.first);
		set.insert(limb_index.second);
	}
	return set.size();
}

template <typename _TIndex>
size_t points_count(const std::vector<_TIndex> &points)
{
	size_t count = 0;
	for (size_t i = 0; i < points.size(); ++i)
		if (points[i] >= 0)
			++count;
	return count;
}

template <typename _T, typename _TIndex>
bool check_connections(const std::list<std::tuple<_TIndex, _TIndex, _T> > &connections)
{
	if (connections.empty())
		return true;
	for (auto i = ++connections.begin(); i != connections.end(); ++i)
	{
		const auto &connection = *i;
		const _TIndex p1 = std::get<0>(connection);
		const _TIndex p2 = std::get<1>(connection);
		for (auto j = connections.begin(); j != i; ++j)
		{
			const auto &_connection = *j;
			if (std::get<0>(_connection) == p1 || std::get<1>(_connection) == p2)
				return false;
		}
	}
	return true;
}

template <typename _T, typename _TIndex, int Options>
std::list<std::tuple<_TIndex, _TIndex, _T> > calc_limb_score(
	Eigen::TensorMap<Eigen::Tensor<const _T, 2, Eigen::RowMajor, _TIndex>, Options> limb1, Eigen::TensorMap<Eigen::Tensor<const _T, 2, Eigen::RowMajor, _TIndex>, Options> limb2,
	std::vector<std::tuple<_T, _T, _T> > &peaks1, std::vector<std::tuple<_T, _T, _T> > &peaks2,
	const _T step, const std::pair<size_t, size_t> &step_limits, const _T min_score, const size_t min_count, const _T epsilon = 1e-6)
{
	typedef std::tuple<_TIndex, _TIndex, _T> _TConnection;
	typedef std::list<_TConnection> _TConnections;

	assert(min_count > 0);
	assert(epsilon > 0);
	assert(step_limits.first <= step_limits.second);
	_TConnections connections;
	for (size_t i = 0; i < peaks1.size(); ++i)
	{
		const auto &peak1 = peaks1[i];
		if (std::get<2>(peak1) <= 0)
			continue;
		for (size_t j = 0; j < peaks2.size(); ++j)
		{
			const auto &peak2 = peaks2[j];
			if (std::get<2>(peak2) <= 0)
				continue;
			const _T y1 = std::get<0>(peak1), x1 = std::get<1>(peak1);
			const _T y2 = std::get<0>(peak2), x2 = std::get<1>(peak2);
			const _T dy = y2 - y1, dx = x2 - x1; //diff
			const _T dist = sqrt(dx * dx + dy * dy);
			if (dist < epsilon)
				continue;
			const _T nx = dx / dist, ny = dy / dist; //norm
			_T score = 0;
			size_t count = 0;
			const size_t steps = std::max(step_limits.first, std::min(step_limits.second, (size_t)std::sqrt(step * std::max(std::abs(dy), std::abs(dx)))));
			for (size_t s = 0; s < steps; ++s)
			{
				const _T prog = (_T)s / steps;
				const _TIndex y = round(y1 + dy * prog);
				const _TIndex x = round(x1 + dx * prog);
				const _T _score = (nx * limb1(y, x) + ny * limb2(y, x));
				if (_score > min_score)
				{
					score += _score;
					++count;
				}
			}
			if (count >= min_count)
				connections.push_back(std::make_tuple(i, j, score / count));
		}
	}
	return connections;
}

template <typename _T, typename _TIndex>
void filter_connections(
	std::list<std::tuple<_TIndex, _TIndex, _T> > &connections,
	const std::vector<std::tuple<_T, _T, _T> > &peaks1, const std::vector<std::tuple<_T, _T, _T> > &peaks2,
	std::vector<std::vector<std::tuple<_T, _T, _T> > > &peaks
)
{
	typedef std::tuple<_TIndex, _TIndex, _T> _TConnection;
	typedef std::vector<_TConnection> _TConnections;
	_TConnections _connections(connections.begin(), connections.end());
	std::sort(_connections.begin(), _connections.end(), [](const _TConnection &c1, const _TConnection &c2)->bool{return std::get<2>(c1) > std::get<2>(c2);});
	std::vector<bool> occur1(peaks1.size(), false), occur2(peaks2.size(), false);
	connections.clear();
	const size_t num = std::min<_TIndex>(peaks1.size(), peaks2.size());
	size_t cnt = 0;
	for (_TIndex i = 0; i < _connections.size(); ++i)
	{
		const _TConnection &connection = _connections[i];
		const _TIndex p1 = std::get<0>(connection);
		const _TIndex p2 = std::get<1>(connection);
		const auto &peak1 = peaks1[p1];
		const auto &peak2 = peaks2[p2];
		if (!occur1[p1] && !occur2[p2] && std::get<2>(peak1) > 0 && std::get<2>(peak2) > 0)
		{
			connections.push_back(connection);
			occur1[p1] = true;
			occur2[p2] = true;
			++cnt;
			if (cnt >= num)
				break;
		}
	}
	assert(check_connections(connections));
}

template <typename _T, typename _TIndex, int Options>
std::list<std::tuple<std::vector<_TIndex>, _T, _TIndex> > clustering(
	std::vector<std::vector<std::tuple<_T, _T, _T> > > &peaks,
	Eigen::TensorMap<Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex>, Options> limbs, const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index,
	const _T step, const std::pair<size_t, size_t> &step_limits, const _T min_score, const size_t min_count
)
{
	typedef Eigen::Tensor<const _T, 3, Eigen::RowMajor, _TIndex> _TConstTensor;
	typedef Eigen::TensorMap<_TConstTensor, Eigen::Aligned> _TConstTensorMap;
	typedef Eigen::Tensor<_T, 2, Eigen::RowMajor, _TIndex> _TLimb;
	typedef std::vector<_TIndex> _TPoints;
	typedef std::tuple<_TPoints, _T, _TIndex> _TCluster;
	std::list<_TCluster> clusters;
	assert(limbs.dimension(0) == limbs_index.size() * 2);
	assert(limbs_points(limbs_index) == peaks.size());
	const _TIndex size = limbs.dimension(1) * limbs.dimension(2);
	for (_TIndex index = 0; index < limbs_index.size(); ++index)
	{
		const std::pair<_TIndex, _TIndex> &limb_index = limbs_index[index];
		assert(0 <= limb_index.first && limb_index.first < peaks.size());
		assert(0 <= limb_index.second && limb_index.second < peaks.size());
		auto &peaks1 = peaks[limb_index.first];
		auto &peaks2 = peaks[limb_index.second];
		const _TIndex channel = index * 2;
		Eigen::TensorMap<Eigen::Tensor<const _T, 2, Eigen::RowMajor, _TIndex>, Options> limb1(
			limbs.data() + channel * size,
			limbs.dimension(1), limbs.dimension(2)
		);
		Eigen::TensorMap<Eigen::Tensor<const _T, 2, Eigen::RowMajor, _TIndex>, Options> limb2(
			limb1.data() + size,
			limbs.dimension(1), limbs.dimension(2)
		);
		auto connections = calc_limb_score(
			limb1, limb2,
			peaks1, peaks2,
			step, step_limits, min_score, min_count
		);
#if 1
#ifdef DEBUG_SHOW
		{
			_TConstTensorMap _parts_(parts_.data(), parts_.dimensions());
			auto canvas = render(image_, limbs_index, _parts_, peaks, clusters);
			canvas = render(canvas, limb_index, _parts_, peaks, connections);
			const std::string title = (boost::format("connections: limb%d (%d-%d)") % index % limb_index.first % limb_index.second).str();
			std::cout << title << std::endl;
			const std::string root = DUMP_DIR "/debug_show/clustering/connections";
			boost::filesystem::create_directories(root);
			cv::imwrite((boost::format("%s/%02d.jpg") % root % index).str(), canvas);
		}
#endif
#endif
		filter_connections(connections, peaks1, peaks2, peaks);
#if 1
#ifdef DEBUG_SHOW
		{
			_TConstTensorMap _parts_(parts_.data(), parts_.dimensions());
			auto canvas = render(image_, limbs_index, _parts_, peaks, clusters, false);
			canvas = render(canvas, limb_index, _parts_, peaks, connections, false);
			canvas = render(canvas, peaks);
			const std::string title = (boost::format("suppress: limb%d (%d-%d)") % index % limb_index.first % limb_index.second).str();
			std::cout << title << std::endl;
			const std::string root = DUMP_DIR "/debug_show/clustering/suppress";
			boost::filesystem::create_directories(root);
			cv::imwrite((boost::format("%s/%02d.jpg") % root % index).str(), canvas);
		}
		{
			_TConstTensorMap _parts_(parts_.data(), parts_.dimensions());
			auto canvas = render(image_, limbs_index, _parts_, peaks, clusters);
			canvas = render(canvas, limb_index, _parts_, peaks, connections);
			const std::string title = (boost::format("filter_connections: limb%d (%d-%d)") % index % limb_index.first % limb_index.second).str();
			std::cout << title << std::endl;
			const std::string root = DUMP_DIR "/debug_show/clustering/filter_connections";
			boost::filesystem::create_directories(root);
			cv::imwrite((boost::format("%s/%02d.jpg") % root % index).str(), canvas);
		}
#endif
#endif
		if (index == 0)
		{
			for (auto i = connections.begin(); i != connections.end(); ++i)
			{
				const auto &connection = *i;
				const _TIndex p1 = std::get<0>(connection), p2 = std::get<1>(connection);
				const auto &peak1 = peaks1[p1];
				const auto &peak2 = peaks2[p2];
#if 1
#ifdef DEBUG_SHOW
				std::cout << boost::format("cluster%d (%d-%d)") % clusters.size() % p1 % p2 << std::endl;
#endif
#endif
				_TPoints points(peaks.size(), -1);
				points[limb_index.first] = p1;
				points[limb_index.second] = p2;
				const _T score = std::get<2>(connection) + std::get<2>(peak1) + std::get<2>(peak2);
				clusters.push_back(std::make_tuple(points, score, 2));
			}
		}
		else
		{
			for (auto i = connections.begin(); i != connections.end(); ++i)
			{
				const auto &connection = *i;
				const _TIndex p1 = std::get<0>(connection), p2 = std::get<1>(connection);
				const auto &peak1 = peaks1[p1];
				const auto &peak2 = peaks2[p2];
				size_t num = 0;
				for (auto c = clusters.begin(); c != clusters.end(); ++c)
				{
					auto &cluster = *c;
					auto &points = std::get<0>(cluster);
					if (points[limb_index.first] == p1)
					{
#if 1
#ifdef DEBUG_SHOW
						std::cout << boost::format("cluster%d (%d->%d)") % std::distance(clusters.begin(), c) % p1 % p2 << std::endl;
#endif
#endif
						assert(points_count(points) == std::get<2>(cluster));
						if (points[limb_index.second] == -1)
						{
							points[limb_index.second] = p2;
							const _T score = std::get<2>(connection) + std::get<2>(peak2);
							std::get<1>(cluster) += score;
							std::get<2>(cluster) += 1;
						}
						assert(points_count(points) == std::get<2>(cluster));
						++num;
					}
				}
				if (!num)
				{
#if 1
#ifdef DEBUG_SHOW
					std::cout << boost::format("cluster%d (%d-%d)") % clusters.size() % p1 % p2 << std::endl;
#endif
#endif
					_TPoints points(peaks.size(), -1);
					points[limb_index.first] = p1;
					points[limb_index.second] = p2;
					const _T score = std::get<2>(connection) + std::get<2>(peak1) + std::get<2>(peak2);
					clusters.push_back(std::make_tuple(points, score, 2));
				}
			}
		}
#if 1
#ifdef DEBUG_SHOW
		{
			auto canvas = render(image_, limbs_index, _TConstTensorMap(parts_.data(), parts_.dimensions()), peaks, clusters);
			const std::string title = (boost::format("clusters: limb%d (%d-%d)") % index % limb_index.first % limb_index.second).str();
			std::cout << title << std::endl;
			const std::string root = DUMP_DIR "/debug_show/clustering/clusters";
			boost::filesystem::create_directories(root);
			cv::imwrite((boost::format("%s/%02d.jpg") % root % index).str(), canvas);
		}
#endif
#endif
	}
	return clusters;
}

template <typename _T, typename _TIndex>
std::list<std::list<std::pair<std::tuple<_TIndex, _T, _T>, std::tuple<_TIndex, _T, _T> > > > filter_cluster(
	const std::vector<std::vector<std::tuple<_T, _T, _T> > > &peaks,
	const std::vector<std::pair<_TIndex, _TIndex> > &limbs_index,
	const std::list<std::tuple<std::vector<_TIndex>, _T, _TIndex> > &clusters,
	const _T min_score, const size_t min_count
)
{
	typedef std::tuple<_TIndex, _T, _T> _TPoint;
	typedef std::pair<_TPoint, _TPoint> _TEdge;
	typedef std::list<_TEdge> _TKeypoints;
	std::list<_TKeypoints> results;
	assert(min_count > 0);
	for (auto c = clusters.begin(); c != clusters.end(); ++c)
	{
		const auto &cluster = *c;
		const _TIndex count = std::get<2>(cluster);
		if (count >= min_count && std::get<1>(cluster) / count > min_score)
		{
			results.push_back(_TKeypoints());
			auto &keypoints = results.back();
			const std::vector<_TIndex> &points = std::get<0>(cluster);
			for (size_t l = 0; l < limbs_index.size(); ++l)
			{
				const std::pair<_TIndex, _TIndex> &limb_index = limbs_index[l];
				const _TIndex p1 = points[limb_index.first], p2 = points[limb_index.second];
				if (p1 >= 0 && p2 >= 0)
				{
					const auto &_p1 = peaks[limb_index.first][p1];
					const auto &_p2 = peaks[limb_index.second][p2];
					keypoints.push_back(std::make_pair(std::make_tuple(limb_index.first, std::get<0>(_p1), std::get<1>(_p1)), std::make_tuple(limb_index.second, std::get<0>(_p2), std::get<1>(_p2))));
				}
			}
		}
	}
	return results;
}
}
}
