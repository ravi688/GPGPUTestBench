#include <iostream>
#include <CUDAMatMul/CUDAMatMul.hpp>
#include <CPUGEMM/CPUGEMM.hpp>
#include <Eigen/Dense>

using Eigen::MatrixXd;

void TestCPUGEMM();

int main()
{
	MatrixXd m(2, 2);
	m(0, 0) = 3;
	m(1, 0) = 2.5;
	m(0, 1) = -1;
	m(1, 1) = m(1, 0) + m(0, 1);
	std::cout << m << std::endl;

	CPUGEMM::Matrix<float> matrix(4, 4);
	matrix[0][0] = 1;
	matrix[0][1] = 2;
	matrix[0][2] = 3;
	matrix[0][3] = 4;

	matrix[1][0] = 11;
	matrix[1][1] = 12;
	matrix[1][2] = 13;
	matrix[1][3] = 14;

	matrix[2][0] = 21;
	matrix[2][1] = 22;
	matrix[2][2] = 23;
	matrix[2][3] = 24;

	matrix[3][0] = 31;
	matrix[3][1] = 32;
	matrix[3][2] = 33;
	matrix[3][3] = 34;	

	std::cout << "Matrix: \n";
	std::cout << matrix << std::endl;

	CPUGEMM::Matrix<float> randMatrix = CPUGEMM::GenerateRandomMatrix<float>(16, 16, 0, 30);
	std::cout << "Random Matrix: \n";
	std::cout << randMatrix << std::endl;


	TestCPUGEMM();


	return 0;
}

template<typename T>
MatrixXd ConvertToEigenMatrix(const CPUGEMM::Matrix<T>& m)
{
	MatrixXd em(m.numRows(), m.numColumns());
	for(std::size_t i = 0; i < m.numRows(); ++i)
	{
		for(std::size_t j = 0; j < m.numColumns(); ++j)
			em(i, j) = m[i][j];
	}
	return em;
}

template<typename T>
CPUGEMM::Matrix<T> ConvertToCPUGEMMMatrix(const MatrixXd& m)
{
	CPUGEMM::Matrix<T> _m(m.rows(), m.cols());
	for(std::size_t i = 0; i < _m.numRows(); ++i)
	{
		for(std::size_t j = 0; j < _m.numColumns(); ++j)
		{
			_m[i][j] = m(i, j);
		}
	}
	return _m;
}

bool EigenEqualApprox(const MatrixXd& a, const MatrixXd& b)
{
	assert(a.rows() == b.rows());
	assert(a.cols() == b.cols());

	for(Eigen::Index i = 0; i < a.rows(); ++i)
	{
		for(Eigen::Index j = 0; j < a.cols(); ++j)
		{
			if(std::fabs(a(i, j) - b(i, j)) > 0.1f)
			{
				std::cout << a(i, j) << " != " << b(i, j) << "\n";
				return false;
			}
		}
	}
	return true;
}

template<typename T>
void CheckGEMM(const CPUGEMM::Matrix<T>& a, const CPUGEMM::Matrix<T>& b, const CPUGEMM::Matrix<T>& c, const CPUGEMM::Matrix<T>& d)
{
	MatrixXd ea = ConvertToEigenMatrix(a);
	MatrixXd eb = ConvertToEigenMatrix(b);
	MatrixXd ec = ConvertToEigenMatrix(c);
	MatrixXd ed = ConvertToEigenMatrix(d);


	MatrixXd t = ea * eb + ec;
	CPUGEMM::Matrix<T> _t = ConvertToCPUGEMMMatrix<T>(t);
	// std::cout << "---Eigen Matrix (D)---\n" << _t << "\n";
	if(!EigenEqualApprox(t, ed))
		std::cerr << "***Doesn't match with eigen's calculation***\n";
}

void TestCPUGEMM()
{
	std::cout << "---------------TestCPUGEMM-----------------\n";
	auto a = CPUGEMM::GenerateRandomMatrix<float>(512, 1024, 0, 10.0);
	auto b = CPUGEMM::GenerateRandomMatrix<float>(1024, 512, 0, 10.0);
	auto c = CPUGEMM::GenerateRandomMatrix<float>(512, 512, 0, 10.0);

	// std::cout << "---Matrix (A)---\n" << a << "\n";
	// std::cout << "---Matrix (B)---\n" << b << "\n";
	// std::cout << "---Matrix (C)---\n" << c << "\n";

	std::cout << "---CASE 1----\n";
	auto d = CPUGEMM::GEMM(a, b, c);
	std::cout << "Time taken: " << d.second << " milliseconds\n";
	std::cout << "Verifying\n";
	CheckGEMM(a, b, c, d.first);

	std::cout << "---CASE 2----\n";
	auto d2 = CPUGEMM::GEMM2(a, b, c);
	std::cout << "Time taken: " << d2.second << " milliseconds\n";
	std::cout << "Verifying\n";
	CheckGEMM(a, b, c, d2.first);
}