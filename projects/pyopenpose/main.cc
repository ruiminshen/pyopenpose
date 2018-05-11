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
#include "label.hpp"
#include "nms.hpp"
#include "hungarian.hpp"
#include "estimate.hpp"
#include "visualize.hpp"

#define QUOTE(x) #x

typedef Eigen::Index TIndex;

PYBIND11_PLUGIN(PROJECT_NAME) {
	pybind11::module m(QUOTE(PROJECT_NAME));
	//label
	m.def("label_parts", &label_parts<float, TIndex>);
	m.def("label_parts", &label_parts<double, TIndex>);
	m.def("label_limbs", &label_limbs<float, TIndex>);
	m.def("label_limbs", &label_limbs<double, TIndex>);
	//nms
	m.def("part_peaks", &part_peaks<float, TIndex>);
	m.def("part_peaks", &part_peaks<double, TIndex>);
	m.def("parts_peaks", &parts_peaks<float, TIndex>);
	m.def("parts_peaks", &parts_peaks<double, TIndex>);
	//hungarian
	m.def("limbs_points", &openpose::postprocess::limbs_points<TIndex>);
	m.def("calc_limb_score", &calc_limb_score<float, TIndex>);
	m.def("calc_limb_score", &calc_limb_score<double, TIndex>);
	m.def("filter_connections", &openpose::postprocess::filter_connections<float, TIndex>);
	m.def("filter_connections", &openpose::postprocess::filter_connections<double, TIndex>);
	m.def("clustering", &clustering<float, TIndex>);
	m.def("clustering", &clustering<double, TIndex>);
	m.def("filter_cluster", &openpose::postprocess::filter_cluster<float, TIndex>);
	m.def("filter_cluster", &openpose::postprocess::filter_cluster<double, TIndex>);
	//estimate
	m.def("estimate", &estimate<float, TIndex>);
	m.def("estimate", &estimate<double, TIndex>);
	m.def("estimate_mt", &estimate_mt<float, TIndex>);
	m.def("estimate_mt", &estimate_mt<double, TIndex>);
	//visualize
	m.def("draw_mask", &draw_mask<uint8_t, uint8_t>);
	return m.ptr();
}
