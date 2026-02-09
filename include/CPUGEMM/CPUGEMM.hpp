#pragma once

#include <CPUGEMM/Matrix.hpp> // for CPUGEMM::Matrix<>
#include <functional> // for std::functional<>

namespace CPUGEMM
{
	template<typename T>
	using Activation = std::function<T(T)>;

	template<typename T>
	Matrix<T> Gemm(const Activation<T>& actfn, T alpha, T beta, T bias, const Matrix<T>& a, const Matrix<T>& b, const Matrix<T>& c)
	{

	}
}
