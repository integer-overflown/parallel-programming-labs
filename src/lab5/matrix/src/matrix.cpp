#include "matrix.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace lab5 {

namespace {
template <typename T, typename EntryGenerator>
void generate(std::vector<std::vector<T>> &matrix, size_t cols,
              EntryGenerator entryGenerator) {
  if (entryGenerator) {
    std::generate(matrix.begin(), matrix.end(),
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
    std::generate(matrix.begin(), matrix.end(),
                  [cols]() { return std::vector<T>(cols); });
  }
}

template <typename T>
void from2DList(std::vector<std::vector<T>> &matrix,
                std::initializer_list<std::initializer_list<T>> init) {
  std::for_each(init.begin(), init.end(),
                [out = matrix.begin()](std::initializer_list<T> row) mutable {
                  *out = row;
                  ++out;
                });
}

}  // namespace

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, EntryGenerator entryGenerator)
    : _matrix(rows) {
  if (rows == 0 || cols == 0) {
    throw std::invalid_argument("Matrix dimensions must be non-zero");
  }

  generate(_matrix, cols, entryGenerator);
}

template <typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> init)
    : _matrix(init.size()) {
  from2DList(_matrix, init);
}

template <typename T>
size_t Matrix<T>::numRows() const {
  return _matrix.size();
}

template <typename T>
size_t Matrix<T>::numColumns() const {
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
Matrix<T> Matrix<T>::doMultiplySeq(const Matrix<T> &other) const {
  if (numColumns() != other.numRows()) {
    throw std::invalid_argument("matrices are of invalid dimensions");
  }

  Matrix result(numRows(), other.numColumns());

  for (size_t i = 0; i < numRows(); ++i) {
    for (size_t j = 0; j < other.numColumns(); ++j) {
      T total{};
      for (size_t k = 0; k < numColumns(); ++k) {
        total += (*this)(i, k) * other(k, j);
      }
      result(i, j) = total;
    }
  }

  return result;
}

template <typename T>
Matrix<T> Matrix<T>::doMultiplyPar(const Matrix<T> &other) const {
  return Matrix(0, 0);
}

template <typename T>
Matrix<T> Matrix<T>::doAddSeq(const Matrix &other) const {
  if (!(numRows() == other.numRows() && numColumns() == other.cols())) {
    throw std::invalid_argument(
        "Only matrices of the same dimensions can be added");
  }

  Matrix<T> result(*this);

  for (size_t i = 0; i < numRows(); ++i) {
    for (size_t j = 0; j < numColumns(); ++j) {
      result(i, j) += other(i, j);
    }
  }

  return result;
}

template <typename T>
Matrix<T> Matrix<T>::doAddPar(const Matrix &other) const {
  if (!(numRows() == other.numRows() && numColumns() == other.cols())) {
    throw std::invalid_argument(
        "Only matrices of the same dimensions can be added");
  }

  Matrix<T> result(*this);

#pragma omp parallel for
  for (size_t i = 0; i < numRows(); ++i) {
    for (size_t j = 0; j < numColumns(); ++j) {
      result(i, j) += other(i, j);
    }
  }

  return result;
}

template <typename T>
std::pair<MatrixRowIterator<T>, MatrixRowIterator<T>> Matrix<T>::rowEntries(
    size_t index) const {
  return {{*this, index}, {*this, index, numColumns()}};
}

template <typename T>
std::pair<MatrixColumnIterator<T>, MatrixColumnIterator<T>>
Matrix<T>::columnEntries(size_t index) const {
  return {{*this, index}, {*this, index, numRows()}};
}

template <typename T>
typename Matrix<T>::value_type Matrix<T>::valueAt(size_t i, size_t j) const {
  return _matrix[i][j];
}

template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int8_t>;
template class Matrix<int16_t>;
template class Matrix<int32_t>;
template class Matrix<int64_t>;

BitMatrix::BitMatrix(size_t rows, size_t cols,
                     BitMatrix::EntryGenerator entryGenerator)
    : _matrix(rows) {
  if (rows == 0 || cols == 0) {
    throw std::invalid_argument("Matrix dimensions must be non-zero");
  }
  generate(_matrix, cols, entryGenerator);
}

BitMatrix::value_type BitMatrix::valueAt(size_t i, size_t j) const {
  return _matrix[i][j];
}

std::vector<bool>::const_reference BitMatrix::operator()(size_t i,
                                                         size_t j) const {
  return _matrix[i][j];
}

std::vector<bool>::reference BitMatrix::operator()(size_t i, size_t j) {
  return _matrix[i][j];
}

size_t BitMatrix::numRows() const { return _matrix.size(); }

size_t BitMatrix::numColumns() const { return _matrix[0].size(); }

BitMatrix::BitMatrix(std::initializer_list<std::initializer_list<bool>> init)
    : _matrix(init.size()) {
  from2DList(_matrix, init);
}

BitMatrix BitMatrix::doMultiplySeq(const BitMatrix &other) const {
  if (numColumns() != other.numRows()) {
    throw std::invalid_argument("matrices are of invalid dimensions");
  }

  BitMatrix result(numRows(), other.numColumns());

  for (size_t i = 0; i < numRows(); ++i) {
    for (size_t j = 0; j < other.numColumns(); ++j) {
      bool total{};
      for (size_t k = 0; k < numColumns(); ++k) {
        total += valueAt(i, k) * other.valueAt(k, j);
      }
      result(i, j) = total;
    }
  }

  return result;
}

BitMatrix BitMatrix::doMultiplyPar(const BitMatrix &other) const {
  return BitMatrix(0, 0);
}

BitMatrix BitMatrix::doAddSeq(const BitMatrix &other) const {
  if (!(numRows() == other.numRows() && numColumns() == other.cols())) {
    throw std::invalid_argument(
        "Only matrices of the same dimensions can be added");
  }

  BitMatrix result(*this);

  for (size_t i = 0; i < numRows(); ++i) {
    for (size_t j = 0; j < numColumns(); ++j) {
      result(i, j) = result.valueAt(i, j) | other.valueAt(i, j);
    }
  }

  return result;
}

BitMatrix BitMatrix::doAddPar(const BitMatrix &other) const {
  if (!(numRows() == other.numRows() && numColumns() == other.cols())) {
    throw std::invalid_argument(
        "Only matrices of the same dimensions can be added");
  }

  BitMatrix result(*this);

  const size_t rows = numRows();
  const size_t cols = numColumns();

#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result(i, j) = result.valueAt(i, j) | other.valueAt(i, j);
    }
  }

  return result;
}

}  // namespace lab5
