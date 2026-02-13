#pragma once

#include <vector> // for std::vector
#include <concepts> // for std::floating_point concept
#include <cassert> // for assert()

#include <random>

namespace CPUGEMM
{
	template<std::floating_point T>
	class Matrix
	{
		public:
			template<typename U>
			class RowIterator
			{
			private:
				std::vector<std::vector<U>>& m_data;
				std::size_t m_cursor;
			public:
				RowIterator(std::vector<std::vector<U>>& data, std::size_t cursor = 0) : m_data(data), m_cursor(cursor) { }
				RowIterator<U> operator++()
				{
					assert(m_cursor < m_data.size());
					++m_cursor;
					return RowIterator<U> { m_data, m_cursor };
				}

				std::vector<U>& operator*()
				{
					return m_data[m_cursor];
				}

				bool operator==(const RowIterator<U>& it) const
				{
					return m_cursor == it.m_cursor;
				}
				bool operator!=(const RowIterator<U>& it) const
				{
					return m_cursor != it.m_cursor;
				}
			};

			template<typename U>
			class ConstRowIterator
			{
			private:
				const std::vector<std::vector<U>>& m_data;
				std::size_t m_cursor;
			public:
				ConstRowIterator(const std::vector<std::vector<U>>& data, std::size_t cursor = 0) : m_data(data), m_cursor(cursor) { }
				ConstRowIterator<U> operator++()
				{
					assert(m_cursor < m_data.size());
					++m_cursor;
					return ConstRowIterator<U> { m_data, m_cursor };
				}

				const std::vector<U>& operator*()
				{
					return m_data[m_cursor];
				}

				bool operator==(const ConstRowIterator<U>& it) const
				{
					return m_cursor == it.m_cursor;
				}
				bool operator!=(const ConstRowIterator<U>& it) const
				{
					return m_cursor != it.m_cursor;
				}
			};

		private:
			std::vector<std::vector<T>> m_data;
		public:
			Matrix(std::size_t numRows, std::size_t numColumns)
			{
				std::vector<T> emptyRow(numColumns, 0);
				m_data = std::vector<std::vector<T>>(numRows, emptyRow);
			}
			Matrix(Matrix&& m) : m_data(std::move(m.m_data)) { }
			std::vector<T>& operator[](std::size_t index)
			{
				assert(index < m_data.size());
				return m_data[index];
			}

			const std::vector<T>& operator[](std::size_t index) const
			{
				assert(index < m_data.size());
				return m_data[index];
			}

			RowIterator<T> begin() { return RowIterator<T> { m_data }; }
			RowIterator<T> end() { return RowIterator<T> { m_data, m_data.size() }; }

			ConstRowIterator<T> cbegin() const { return ConstRowIterator<T> { m_data }; }
			ConstRowIterator<T> cend() const { return ConstRowIterator<T> { m_data, m_data.size() }; }

			ConstRowIterator<T> begin() const { return cbegin(); }
			ConstRowIterator<T> end() const { return cend(); }


			// getters
			std::size_t numRows() const noexcept { return m_data.size(); }
			std::size_t numColumns() const noexcept { return m_data[0].size();  }
	};

	template<typename T>
	std::ostream& operator<<(std::ostream& stream, const Matrix<T>& m)
	{
		for(const auto& row : m)
		{
			for(const auto& e : row)
			{
				stream << e << " ";
			}
			stream << "\n";
		}
		return stream;
	}

	template<typename T>
	Matrix<T> GenerateRandomMatrix(std::size_t numRows, std::size_t numColumns, T min, T max)
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		std::uniform_real_distribution<T> distrib(min, max);

		Matrix<T> m(numRows, numColumns);
		for(auto& row : m)
			for(auto& e : row)
				e = distrib(gen);

		return m;
	}
}
