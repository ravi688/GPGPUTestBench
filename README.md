# GPGPUTestBench
General Purpose GPU Programming Test Bench

## Building
```
git clone https://github.com/ravi688/GPGPUTestBench.git
cd GPGPUTestBench
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
## Running examples
```
./build/CUDAMatmul
```

## API
### Single threaded matrix multiplication on CPU
*Defined in `<CPUGEMM/CPUGEMM.hpp>`, namespace `CPUGEMM`*
```cpp
template<typename T>
	std::pair<Matrix<T>, float> GEMM(const Matrix<T>& a, const Matrix<T>& b, const Matrix<T>& c, T alpha = 1, T beta = 1, T bias = 0, const Activation<T>& actfn = [](T _) { return _; })
```

### Multi threaded matrix multiplication on CPU
*Defined in `<CPUGEMM/CPUGEMM.hpp>`, namespace `CPUGEMM`*
```cpp
template<typename T>
	std::pair<Matrix<T>, float> GEMM2(const Matrix<T>& a, const Matrix<T>& b, const Matrix<T>& c, T alpha = 1, T beta = 1, T bias = 0, const Activation<T>& actfn = [](T _) { return _; })
```

## License
MIT
