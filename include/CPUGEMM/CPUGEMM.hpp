#pragma once

#include <CPUGEMM/Matrix.hpp> // for CPUGEMM::Matrix<>
#include <functional> // for std::functional<>
#include <chrono> // for 
#include <thread> // for std::thread

namespace CPUGEMM
{
	template<typename T>
	using Activation = std::function<T(T)>;

	template<typename T>
	std::pair<Matrix<T>, float> GEMM(const Matrix<T>& a, const Matrix<T>& b, const Matrix<T>& c, T alpha = 1, T beta = 1, T bias = 0, const Activation<T>& actfn = [](T _) { return _; })
	{
		assert(a.numColumns() == b.numRows());
		assert(b.numColumns() == c.numColumns());
		assert(a.numRows() == c.numRows());

		Matrix<T> d(c.numRows(), c.numColumns());

		auto beginTime = std::chrono::steady_clock::now();

		for(std::size_t i = 0; i < d.numRows(); ++i)
		{
			for(std::size_t j = 0; j < d.numColumns(); ++j)
			{
				T t = 0;
				for(std::size_t k = 0; k < a.numColumns(); ++k)
				{
					t += a[i][k] * b[k][j] * alpha;
				}
				d[i][j] = actfn(t + c[i][j] * beta + bias);
			}
		}

		auto endTime = std::chrono::steady_clock::now();
		float timeTaken = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(endTime - beginTime).count();

		return { std::move(d), timeTaken };
	}

	template<typename T>
	std::pair<Matrix<T>, float> GEMM2(const Matrix<T>& a, const Matrix<T>& b, const Matrix<T>& c, T alpha = 1, T beta = 1, T bias = 0, const Activation<T>& actfn = [](T _) { return _; })
	{
		assert(a.numColumns() == b.numRows());
		assert(b.numColumns() == c.numColumns());
		assert(a.numRows() == c.numRows());

		Matrix<T> d(c.numRows(), c.numColumns());

		std::vector<std::thread> threads;

		const std::size_t numThreads = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4;
		threads.reserve(numThreads);
		auto numRowsPerThread = d.numRows() / numThreads;
		auto numRowsForLastThread = d.numRows() % numThreads;

		auto beginTime = std::chrono::steady_clock::now();

		for(std::size_t n = 0; n < numThreads; ++n)
		{
			std::thread thread([numRowsPerThread, n, alpha, beta, bias, &actfn](Matrix<T>& d, const Matrix<T>& a, const Matrix<T>& b, const Matrix<T>& c)
			{
				for(std::size_t i = n * numRowsPerThread; i < ((n + 1) * numRowsPerThread); ++i)
				{
					for(std::size_t j = 0; j < d.numColumns(); ++j)
					{
							T t = 0;
							for(std::size_t k = 0; k < a.numColumns(); ++k)
							{
								t += a[i][k] * b[k][j] * alpha;
							}
							d[i][j] = actfn(t + c[i][j] * beta + bias);
		
					}
				}
			}, std::ref(d), std::ref(a), std::ref(b), std::ref(c));
			threads.push_back(std::move(thread));
		}

		if(numRowsForLastThread)
		{
			std::thread thread([numThreads, numRowsPerThread, alpha, beta, bias, &actfn](Matrix<T>& d, const Matrix<T>& a, const Matrix<T>& b, const Matrix<T>& c)
			{
				for(std::size_t i = numThreads * numRowsPerThread; i < d.numRows(); ++i)
				{
					for(std::size_t j = 0; j < d.numColumns(); ++j)
					{
							T t = 0;
							for(std::size_t k = 0; k < a.numColumns(); ++k)
							{
								t += a[i][k] * b[k][j] * alpha;
							}
							d[i][j] = actfn(t + c[i][j] * beta + bias);
		
					}
				}
			}, std::ref(d), std::ref(a), std::ref(b), std::ref(c));
			threads.push_back(std::move(thread));
		}

		for(auto& thread : threads)
			thread.join();

		auto endTime = std::chrono::steady_clock::now();
		float timeTaken = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(endTime - beginTime).count();

		return { std::move(d), timeTaken };
	}
}
