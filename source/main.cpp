#include <iostream>
#include <CUDAMatMul/CUDAMatMul.hpp>
#include <CPUGEMM/CPUGEMM.hpp>
#include <Eigen/Dense>

using Eigen::MatrixXd;

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

	return 0;
}
