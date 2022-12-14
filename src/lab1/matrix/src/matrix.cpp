#include "matrix.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <stdexcept>

namespace lab1 {

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, EntryGenerator entryGenerator) {
  if (rows == 0 || cols == 0) {
    throw std::invalid_argument("Matrix dimensions must be non-zero");
  }

  _matrix.resize(rows);

  if (entryGenerator) {
    std::generate(_matrix.begin(), _matrix.end(),
                  [=, rowNo = size_t(0)]() mutable -> std::vector<T> {
                    std::vector<T> row(cols);
                    std::generate(row.begin(), row.end(),
                                  [=, colNo = size_t(0)]() mutable -> T {
                                    return entryGenerator(rowNo, colNo++);
                                  });
                    ++rowNo;
                    return row;
                  });
  } else {
    std::generate(_matrix.begin(), _matrix.end(),
                  [cols]() { return std::vector<T>(cols); });
  }
}

template <typename T>
size_t Matrix<T>::rows() const {
  return _matrix.size();
}

template <typename T>
size_t Matrix<T>::cols() const {
  return _matrix[0].size();
}

template <typename T>
const T &Matrix<T>::operator()(size_t row, size_t col) const {
  return _matrix[row][col];
}

template <typename T>
T &Matrix<T>::operator()(size_t row, size_t col) {
  return _matrix[row][col];
}

template <typename T>
Matrix<T> Matrix<T>::multiply(const Matrix<T> &other) const {
  if (cols() != other.rows()) {
    throw std::invalid_argument("matrices are of invalid dimensions");
  }

  Matrix result(rows(), other.cols());

  for (size_t i = 0; i < rows(); ++i) {
    for (size_t j = 0; j < other.cols(); ++j) {
      T total{};
      for (size_t k = 0; k < cols(); ++k) {
        total += (*this)(i, k) * other(k, j);
      }
      result(i, j) = total;
    }
  }

  return result;
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const Matrix<T> &m) {
  for (size_t i = 0; i < m.rows(); ++i) {
    for (size_t j = 0; j < m.cols(); ++j) {
      out << std::setw(5) << m(i, j) << ' ';
    }
    out << '\n';
  }
  return out;
}

template <typename T>
std::pair<MatrixRowIterator<T>, MatrixRowIterator<T>> Matrix<T>::rowEntries(
    size_t index) const {
  return {{*this, index}, {*this, index, cols()}};
}

template <typename T>
std::pair<MatrixColumnIterator<T>, MatrixColumnIterator<T>>
Matrix<T>::columnEntries(size_t index) const {
  return {{*this, index}, {*this, index, rows()}};
}

template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int8_t>;
template class Matrix<int16_t>;
template class Matrix<int32_t>;
template class Matrix<int64_t>;

}  // namespace lab1
