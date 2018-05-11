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
#include <memory>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

template <typename _TTensor, typename _T>
_TTensor numpy_tensor(pybind11::array_t<_T> array, typename std::enable_if<_TTensor::NumIndices == 2>::type* = nullptr)
{
	typedef typename _TTensor::Index _TIndex;
	const auto r = array.template unchecked<2>();
	_TTensor tensor((_TIndex)r.shape(0), (_TIndex)r.shape(1));
	for (_TIndex i = 0; i < r.shape(0); ++i)
		for (_TIndex j = 0; j < r.shape(1); ++j)
			tensor(i, j) = r(i, j);
	return tensor;
}

template <typename _TTensor, typename _T>
_TTensor numpy_tensor(pybind11::array_t<_T> array, typename std::enable_if<_TTensor::NumIndices == 3>::type* = nullptr)
{
	typedef typename _TTensor::Index _TIndex;
	const auto r = array.template unchecked<3>();
	_TTensor tensor((_TIndex)r.shape(0), (_TIndex)r.shape(1), (_TIndex)r.shape(2));
	for (_TIndex i = 0; i < r.shape(0); ++i)
		for (_TIndex j = 0; j < r.shape(1); ++j)
			for (_TIndex k = 0; k < r.shape(2); ++k)
				tensor(i, j, k) = r(i, j, k);
	return tensor;
}
