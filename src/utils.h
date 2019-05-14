// (c) 2017 Blai Bonet
// (c) 2019 Florent Teichteil-Koenigsbuch -> make read_time_in_seconds() cross platform

#ifndef UTILS_H
#define UTILS_H

#define BOOST_CHRONO_HEADER_ONLY
#define BOOST_CHRONO_DONT_PROVIDE_HYBRID_ERROR_HANDLING

#include <boost/chrono.hpp>
#include <string>

namespace Utils {

inline float read_time_in_seconds(bool add_stime = true) {
    double time = 0;
    time += double(boost::chrono::duration_cast<boost::chrono::microseconds>(boost::chrono::process_user_cpu_clock::now().time_since_epoch()).count()) / double(1e6);
    if (add_stime) {
        time += double(boost::chrono::duration_cast<boost::chrono::microseconds>(boost::chrono::process_system_cpu_clock::now().time_since_epoch()).count()) / double(1e6);
    }

    return time;
}

inline std::string normal() { return "\x1B[0m"; }
inline std::string red() { return "\x1B[31;1m"; }
inline std::string green() { return "\x1B[32;1m"; }
inline std::string yellow() { return "\x1B[33;1m"; }
inline std::string blue() { return "\x1B[34;1m"; }
inline std::string magenta() { return "\x1B[35;1m"; }
inline std::string cyan() { return "\x1B[36;1m"; }
inline std::string error() { return "\x1B[31;1merror: \x1B[0m"; }
inline std::string warning() { return "\x1B[35;1mwarning: \x1B[0m"; }
inline std::string internal_error() { return "\x1B[31;1minternal error: \x1B[0m"; }

inline std::string cmdline(int argc, const char *argv[]) {
    std::string cmd = argv[0];
    for( int j = 1; j < argc; ++j )
        cmd += std::string(" ") + argv[j];
    return cmd;
}

};

#endif

