#ifndef OSLABS_MATRIX_H
#define OSLABS_MATRIX_H
#include <cstddef>
#include <execution>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <vector>

namespace lab5 {

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

template <typename Impl>
class MatrixBase {
 public:
  [[nodiscard]] Impl multiply(const Impl &other,
                              std::execution::sequenced_policy) const {
    return static_cast<const Impl *>(this)->doMultiplySeq(other);
  }

  [[nodiscard]] Impl multiply(
      const Impl &other, std::execution::parallel_unsequenced_policy) const {
    return static_cast<const Impl *>(this)->doMultiplyPar(other);
  }

  [[nodiscard]] Impl add(const Impl &other,
                         std::execution::sequenced_policy) const {
    return static_cast<const Impl *>(this)->doAddSeq(other);
  }

  [[nodiscard]] Impl add(const Impl &other,
                         std::execution::parallel_unsequenced_policy) const {
    return static_cast<const Impl *>(this)->doAddPar(other);
  }

  [[nodiscard]] size_t rows() const {
    return static_cast<const Impl *>(this)->numRows();
  }

  [[nodiscard]] size_t cols() const {
    return static_cast<const Impl *>(this)->numColumns();
  }

  [[nodiscard]] auto value(size_t i, size_t j) const {
    return static_cast<typename Impl::value_type>(
        static_cast<const Impl *>(this)->valueAt(i, j));
  }

  friend Impl operator+(const Impl &a, const Impl &b) {
    return a.add(b, std::execution::seq);
  }

  friend Impl operator*(const Impl &a, Impl &b) {
    return a.multiply(b, std::execution::seq);
  }
};

template <typename T>
class Matrix : public MatrixBase<Matrix<T>> {
  using Self = Matrix<T>;
  using Base = MatrixBase<Self>;

 public:
  template <typename>
  friend class MatrixRowIterator;
  template <typename>
  friend class MatrixColumnIterator;
  friend Base;

  using value_type = T;
  using EntryGenerator = T (*)(ptrdiff_t, ptrdiff_t);

  Matrix(size_t rows, size_t cols, EntryGenerator entryGenerator = nullptr);
  Matrix(std::initializer_list<std::initializer_list<T>> init);
  const T &operator()(size_t row, size_t col) const;
  T &operator()(size_t row, size_t col);

  [[nodiscard]] std::pair<MatrixRowIterator<T>, MatrixRowIterator<T>>
  rowEntries(size_t index) const;
  [[nodiscard]] std::pair<MatrixColumnIterator<T>, MatrixColumnIterator<T>>
  columnEntries(size_t index) const;

  template <typename U>
  friend std::ostream &operator<<(std::ostream &out, const Matrix<U> &m);

 private:
  [[nodiscard]] Matrix doMultiplySeq(const Matrix &other) const;

  [[nodiscard]] Matrix doMultiplyPar(const Matrix &other) const;

  [[nodiscard]] Matrix doAddSeq(const Matrix &other) const;

  [[nodiscard]] Matrix doAddPar(const Matrix &other) const;

  [[nodiscard]] size_t numRows() const;

  [[nodiscard]] size_t numColumns() const;

  [[nodiscard]] value_type valueAt(size_t i, size_t j) const;

  std::vector<std::vector<T>> _matrix;
};

template <typename U>
std::ostream &operator<<(std::ostream &out, const Matrix<U> &m) {
  for (size_t i = 0; i < m.rows(); ++i) {
    for (size_t j = 0; j < m.cols(); ++j) {
      out << std::setw(5) << m(i, j) << ' ';
    }
    out << '\n';
  }
  return out;
}

extern template class Matrix<float>;
extern template class Matrix<double>;
extern template class Matrix<int8_t>;
extern template class Matrix<int16_t>;
extern template class Matrix<int32_t>;
extern template class Matrix<int64_t>;

}  // namespace lab5

#endif  // OSLABS_MATRIX_H
