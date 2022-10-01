#ifndef OSLABS_MATRIX_H
#define OSLABS_MATRIX_H
#include <cstddef>
#include <execution>
#include <iostream>
#include <vector>

namespace lab1 {

template <typename T>
class Matrix;

template <typename T>
class MatrixRowIterator {
 public:
  MatrixRowIterator(const Matrix<T> &matrix, size_t row, size_t pos = 0)
      : _matrix(&matrix), _row(row), _pos(pos) {}

  const T &operator*() const { return _matrix->_matrix[_row][_pos]; }

  MatrixRowIterator &operator++() {
    ++_pos;
    return *this;
  }

  const MatrixRowIterator operator++(int) {
    MatrixRowIterator it(*this);
    ++_pos;
    return it;
  }

  friend bool operator==(const MatrixRowIterator &lhs,
                         const MatrixRowIterator &rhs) {
    return lhs._matrix == rhs._matrix && lhs._row == rhs._row &&
           lhs._pos == rhs._pos;
  }

  friend bool operator!=(const MatrixRowIterator &lhs,
                         const MatrixRowIterator &rhs) {
    return !(lhs == rhs);
  }

  MatrixRowIterator operator+(size_t n) const {
    return {*_matrix, _row, _pos + n};
  }

  MatrixRowIterator operator-(size_t n) const {
    return {*_matrix, _row, _pos - n};
  }

  MatrixRowIterator &operator+=(size_t n) { return *this = *this + n; }

  MatrixRowIterator &operator-=(size_t n) { return *this = *this - n; }

 private:
  const Matrix<T> *_matrix;
  size_t _row, _pos{};
};

template <typename T>
class MatrixColumnIterator {
 public:
  MatrixColumnIterator(const Matrix<T> &matrix, size_t column, size_t pos = 0)
      : _matrix(&matrix), _column(column), _pos(pos) {}

  const T &operator*() const { return _matrix->_matrix[_pos][_column]; }

  MatrixColumnIterator &operator++() {
    ++_pos;
    return *this;
  }

  const MatrixColumnIterator operator++(int) {
    MatrixColumnIterator it(*this);
    ++_pos;
    return it;
  }

  friend bool operator==(const MatrixColumnIterator &lhs,
                         const MatrixColumnIterator &rhs) {
    return lhs._matrix == rhs._matrix && lhs._column == rhs._column &&
           lhs._pos == rhs._pos;
  }

  friend bool operator!=(const MatrixColumnIterator &lhs,
                         const MatrixColumnIterator &rhs) {
    return !(lhs == rhs);
  }

  MatrixColumnIterator operator+(size_t n) const {
    return {*_matrix, _column, _pos + n};
  }

  MatrixColumnIterator operator-(size_t n) const {
    return {*_matrix, _column, _pos - n};
  }

  MatrixColumnIterator &operator+=(size_t n) { return *this = *this + n; }

  MatrixColumnIterator &operator-=(size_t n) { return *this = *this - n; }

 private:
  const Matrix<T> *_matrix;
  size_t _column, _pos{};
};

template <typename T>
class Matrix {
 public:
  template <typename>
  friend class MatrixRowIterator;
  template <typename>
  friend class MatrixColumnIterator;

  using EntryGenerator = T (*)(ptrdiff_t, ptrdiff_t);
  Matrix(size_t rows, size_t cols, EntryGenerator entryGenerator = nullptr);
  [[nodiscard]] size_t rows() const;
  [[nodiscard]] size_t cols() const;
  const T &operator()(size_t row, size_t col) const;
  T &operator()(size_t row, size_t col);

  [[nodiscard]] Matrix multiply(const Matrix &other) const;

  [[nodiscard]] std::pair<MatrixRowIterator<T>, MatrixRowIterator<T>>
  rowEntries(size_t index) const;
  [[nodiscard]] std::pair<MatrixColumnIterator<T>, MatrixColumnIterator<T>>
  columnEntries(size_t index) const;

  friend std::ostream &operator<<(std::ostream &out, const Matrix &m);

 private:
  std::vector<std::vector<T>> _matrix;
};

extern template class Matrix<float>;
extern template class Matrix<double>;
extern template class Matrix<int8_t>;
extern template class Matrix<int16_t>;
extern template class Matrix<int32_t>;
extern template class Matrix<int64_t>;

}  // namespace lab1

#endif  // OSLABS_MATRIX_H
