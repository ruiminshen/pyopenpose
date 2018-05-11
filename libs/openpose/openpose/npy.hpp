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

#include <vector>
#include <string>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cnpy.h>

namespace openpose
{
template <typename _T, typename _TTensor, int Options>
void save_npy(Eigen::TensorMap<_TTensor, Options> tensor, const std::string &path)
{
	std::vector<unsigned int> shape(tensor.rank());
	for (size_t i = 0; i < shape.size(); ++i)
		shape[i] = tensor.dimension(i);
	std::vector<_T> data(tensor.size());
	for (size_t i = 0; i < data.size(); ++i)
		data[i] = tensor(i);
	cnpy::npy_save(path, &data.front(), &shape.front(), shape.size());
}

template <typename _T, typename _TTensor>
_TTensor load_npy2(const std::string &path, typename std::enable_if<_TTensor::NumIndices == 2>::type* = nullptr)
{
	typedef typename _TTensor::Index _TIndex;
	cnpy::NpyArray arr = cnpy::npy_load(path);
	assert(arr.shape.size() == 2);
	_TTensor tensor((_TIndex)arr.shape[0], (_TIndex)arr.shape[1]);
	const _T *data = arr.data<_T>();
	for (size_t i = 0; i < tensor.size(); ++i)
		tensor(i) = data[i];
	assert(tensor.rank() == arr.shape.size());
	return tensor;
}

template <typename _T, typename _TTensor>
_TTensor load_npy3(const std::string &path, typename std::enable_if<_TTensor::NumIndices == 3>::type* = nullptr)
{
	typedef typename _TTensor::Index _TIndex;
	cnpy::NpyArray arr = cnpy::npy_load(path);
	assert(arr.shape.size() == 3);
	_TTensor tensor((_TIndex)arr.shape[0], (_TIndex)arr.shape[1], (_TIndex)arr.shape[2]);
	const _T *data = arr.data<_T>();
	for (size_t i = 0; i < tensor.size(); ++i)
		tensor(i) = data[i];
	assert(tensor.rank() == arr.shape.size());
	return tensor;
}
}
