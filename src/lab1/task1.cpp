#include <iostream>
#include <ctime>
#include <cstdint>
#include <limits>
#include <iomanip>

int main()
{
    time_t maxInt32Time(std::numeric_limits<int32_t>::max());
    std::cout << "local: " << std::put_time(std::localtime(&maxInt32Time), "%c %Z") << '\n';
}
