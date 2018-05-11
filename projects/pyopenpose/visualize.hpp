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
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

template <typename _T, typename _TMask>
void draw_mask(pybind11::array_t<_T> image, pybind11::array_t<_TMask> mask, const _TMask threshold = 128)
{
	auto _image = image.template mutable_unchecked<3>();
	const auto _mask = mask.template unchecked<2>();
	assert(_image.shape(0) == _mask.shape(0));
	assert(_image.shape(1) == _mask.shape(1));
	for (size_t y = 0; y < _mask.shape(0); ++y)
		for (size_t x = 0; x < _mask.shape(1); ++x)
			if (_mask(y, x) < threshold)
				for (size_t i = 0; i < _image.shape(2); ++i)
					_image(y, x, i) = 0;
}
