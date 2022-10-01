#include "matrix.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <stdexcept>

namespace lab1 {

Matrix::Matrix(size_t rows, size_t cols, EntryGenerator entryGenerator) {
  if (rows == 0 || cols == 0) {
    throw std::invalid_argument("Matrix dimensions must be non-zero");
  }

  _matrix.resize(rows);

  if (entryGenerator) {
    std::generate(_matrix.begin(), _matrix.end(),
                  [=, rowNo = size_t(0)]() mutable -> std::vector<float> {
                    std::vector<float> row(cols);
                    std::generate(row.begin(), row.end(),
                                  [=, colNo = size_t(0)]() mutable -> float {
                                    return entryGenerator(rowNo, colNo++);
                                  });
                    ++rowNo;
                    return row;
                  });
  } else {
    std::generate(_matrix.begin(), _matrix.end(),
                  [cols]() { return std::vector<float>(cols); });
  }
}

size_t Matrix::rows() const { return _matrix.size(); }

size_t Matrix::cols() const { return _matrix[0].size(); }

const float &Matrix::operator()(size_t row, size_t col) const {
  return _matrix[row][col];
}

float &Matrix::operator()(size_t row, size_t col) { return _matrix[row][col]; }

Matrix Matrix::multiply(const Matrix &other) const {
  if (cols() != other.rows()) {
    throw std::invalid_argument("matrices are of invalid dimensions");
  }

  Matrix result(rows(), other.cols());

  for (size_t i = 0; i < rows(); ++i) {
    for (size_t j = 0; j < other.cols(); ++j) {
      float total{};
      for (size_t k = 0; k < cols(); ++k) {
        total += (*this)(i, k) * other(k, j);
      }
      result(i, j) = total;
    }
  }

  return result;
}

std::ostream &operator<<(std::ostream &out, const Matrix &m) {
  for (size_t i = 0; i < m.rows(); ++i) {
    for (size_t j = 0; j < m.cols(); ++j) {
      out << std::setw(5) << m(i, j) << ' ';
    }
    out << '\n';
  }
  return out;
}

std::pair<MatrixRowIterator, MatrixRowIterator> Matrix::rowEntries(
    size_t index) const {
  return {{*this, index}, {*this, index, cols()}};
}

std::pair<MatrixColumnIterator, MatrixColumnIterator> Matrix::columnEntries(
    size_t index) const {
  return {{*this, index}, {*this, index, rows()}};
}

}  // namespace lab1
