# GPGPUTestBench
General Purpose GPU Programming Test Bench

## Single threaded matrix multiplication in CPU
*Defined in `<CPUGEMM/CPUGEMM.hpp>`*, namespace `CPUGEMM`
```cpp
template<typename T>
	std::pair<Matrix<T>, float> GEMM(const Matrix<T>& a, const Matrix<T>& b, const Matrix<T>& c, T alpha = 1, T beta = 1, T bias = 0, const Activation<T>& actfn = [](T _) { return _; })
```
