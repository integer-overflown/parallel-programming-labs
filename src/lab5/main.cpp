#include <iostream>

#include "matrix.h"

int main() {
  lab5::Matrix<float> m = {{1, 2, 3}, {4, 5, 6}};
  lab5::Matrix<float> n = {{9, 8, 7}, {6, 5, 4}};

  std::cout << m + n << '\n';
}
