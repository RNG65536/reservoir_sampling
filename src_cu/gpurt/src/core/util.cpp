#include "util.h"

#include <random>
using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <lodepng.cpp>
#include <tiny_obj_loader.cc>

default_random_engine            rng;
uniform_real_distribution<float> dist(0.0f, 1.0f);
float                            randf()
{
    return dist(rng);
}

Logger::Logger(std::string logfile, bool override /*= false*/)
{
    output.open(
        logfile.c_str(),
        /*std::ios::binary |*/ (override ? std::ios::ate : std::ios::app));
    if (false == output.is_open() || false == output.good())
    {
        printf("Cannot open log: %s\n", logfile.c_str());
    }
    else
    {
        printf("Log opened: %s\n", logfile.c_str());
    }
    output << "------------------------------------------\n";
}

Logger::~Logger()
{
    output.close();
}

template <class T>
Logger& Logger::operator<<(const T& v)
{
    output << v;
    output.flush();
    return *this;
}

void lprintf(const char* sformat, ...)
{
    static Logger vllog("glsl.txt", true);

    va_list ap;
    va_start(ap, sformat);
    char line[(1 << 16) + 1];
    vsprintf_s(line, sformat, ap);
    line[(1 << 16)] = '\0';
    va_end(ap);
    printf(line);

    vllog << line;
}

std::string printToString(const char* format, ...)
{
    char    buf[1024];
    va_list arguments;
    va_start(arguments, format);
    vsprintf_s(buf, format, arguments);
    va_end(arguments);

    return std::string(buf);
}

static unsigned char clamp(float x)
{
    return unsigned char(x < 0 ? 0 : x > 255 ? 255 : x);
}

void Abort(unsigned int err)
{
    if (err > 0)
    {
        lprintf("abort : ", err);
    }
    exit(err);
}


