#define _USE_MATH_DEFINES

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <random>
using std::cout;
using std::endl;

std::random_device rdev;

namespace
{
std::default_random_engine         rng1(rdev());
std::uniform_int_distribution<int> dist1;

int randi(int mod) { return dist1(rng1) % mod; }
}  // namespace

namespace
{
std::default_random_engine            rng2(rdev());
std::uniform_real_distribution<float> dist2(0.0f, 1.0f);

float randf() { return dist2(rng2); }
}  // namespace

class Sampler
{
public:
    virtual ~Sampler() {}
    virtual float next() = 0;
};

unsigned int Hash(unsigned int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

unsigned int RngNext(unsigned int& seed_x, unsigned int& seed_y)
{
    unsigned int result = seed_x * 0x9e3779bb;

    seed_y ^= seed_x;
    seed_x = ((seed_x << 26) | (seed_x >> (32 - 26))) ^ seed_y ^ (seed_y << 9);
    seed_y = (seed_x << 13) | (seed_x >> (32 - 13));

    return result;
}

float uint_as_float(unsigned int x) { return *reinterpret_cast<float*>(&x); }

float Rand(unsigned int& seed_x, unsigned int& seed_y)
{
    unsigned int u = 0x3f800000 | (RngNext(seed_x, seed_y) >> 9);
    return uint_as_float(u) - 1.0;
}

class FastSampler : public Sampler
{
    unsigned int seed_x;
    unsigned int seed_y;

public:
    FastSampler(unsigned int seed0, unsigned int seed1, unsigned int frame_idx = 0)
    {
        unsigned int s0 = (seed0 << 16) | seed1;
        unsigned int s1 = frame_idx;

        seed_x = Hash(s0);
        seed_y = Hash(s1);
        RngNext(seed_x, seed_y);
    }

    float next() { return Rand(seed_x, seed_y); }
};

//
// to sample from a stream of n_total numbers into a pool
// of size n_sample with equal probability
//
namespace reservoir_sampling
{
constexpr int n_total  = 1000;
constexpr int n_sample = 10;  // pool size M

int main()
{
    std::vector<int> count(n_total);
    count.assign(n_total, 0);

    // number of trials
    for (int t = 0; t < 10000; t++)
    {
        std::vector<int> pool(n_sample);

        for (int i = 0; i < n_total; i++)
        {
            if (i < n_sample)
            {
                pool[i] = i;
            }
            else
            {
                // with probability M / i
                if (randi(i) < n_sample)
                {
                    pool[randi(n_sample)] = i;
                }
            }
        }

        for (int i = 0; i < n_sample; i++)
        {
            count[pool[i]]++;
        }
    }

    // count the samples after all trials
    // the should follow the uniform distribution
    for (int i = 0; i < n_total; i++)
    {
        cout << count[i] << ", ";
    }

    return 0;
}
}  // namespace reservoir_sampling

//
// 1 sample reservoir for sampling from a stream of weighted samples
//
template <class T>
class ReservoirT
{
public:
    T     y;     // the output sample
    int   yi;    // output sample index
    float wsum;  // the sum of weights
    int   M;     // the number of samples seen so far
    float W;     // ?? for multi streaming ris

    ReservoirT() { reset(); }

    void reset()
    {
        // y    = T(0);
        yi   = -1;
        wsum = 0;
        M    = 0;
        W    = 0;
    }

    //
    // xi : the incoming sample
    // wi : weight of the sample
    //
    void update(const T& xi, float wi)
    {
        wsum += wi;
        M += 1;
        if (randf() <= wi / wsum)  // the first is accepted with probability 1
        {
            y  = xi;
            yi = M - 1;
        }
    }
};

using Reservoir  = ReservoirT<float>;
using Reservoir3 = ReservoirT<glm::vec3>;

//
// to sample from a stream of n_total numbers with
// weighted probability using 1 sample reservoir
//
namespace weighted_reservoir_sampling_one_sample
{
constexpr int n_total = 1000;

float weight(int i)
{
    // return 1.0f;
    return (i + 0.5f) / n_total;
}

int main()
{
    std::vector<int> count(n_total);
    count.assign(n_total, 0);

    // number of trials
    for (int t = 0; t < 100000; t++)
    {
        Reservoir r;
        // int   y    = 0;
        // float wsum = 0;
        // int   M    = 0;

        for (int i = 0; i < n_total; i++)
        {
            float val = 0.0f;  // unused
            r.update(val, weight(i));
            // float wi = weight(i);
            // wsum += wi;
            // M += 1;
            // if (randf() < wi / wsum)
            //{
            //    y = i;
            //}
        }

        count[r.yi]++;
    }

    // count the samples after all trials
    // they should follow a distribution given by the weights
    for (int i = 0; i < n_total; i++)
    {
        cout << count[i] << ", ";
    }

    return 0;
}
}  // namespace weighted_reservoir_sampling_one_sample

//
// comparison of different monte carlo sampling methods
// in mc integration for sin(x * pi) * x in [0, 1]
//
namespace monte_carlo
{
float fx(float x)
{
    if (x >= 0.0f && x <= 1.0f)
        return sin(x * (float)M_PI) * x;
    else
        return 0;
}

int g_sample_count = 0;

void sample_clear()
{
    g_sample_count = 0;
    cout << endl;
}

void sample_record() { ++g_sample_count; }

void sample_report()
{
    cout << ">> sample count = " << g_sample_count << endl;
    cout << endl;
}

//
// sample between a and b with uniform distribution
//
std::pair<float, float> sample_uniform(float a, float b)
{
    sample_record();

    float x   = a + randf() * (b - a);
    float pdf = 1.0f / (b - a);
    return std::make_pair(x, pdf);
}

//
// sample between a and b with linear distribution
//
std::pair<float, float> sample_linear(float a, float b)
{
    sample_record();

    float ksi = sqrt(randf());
    float x   = a + ksi * (b - a);
    // float pdf = (x - a) * 2.0f / ((b - a) * (b - a));
    float pdf = ksi * 2.0f / (b - a);
    return std::make_pair(x, pdf);
}

float pdf_uniform(float x, float a, float b) { return 1.0f / (b - a); }

float pdf_linear(float x, float a, float b) { return (x - a) * 2.0f / ((b - a) * (b - a)); }

float integration_by_summation(float a, float b)
{
    float sum         = 0;
    int   n_intervals = 10000;
    float dx          = (b - a) / n_intervals;
    for (int i = 0; i < n_intervals; i++)
    {
        sum += dx * fx(a + dx * (i + 0.5f));
    }
    return sum;
}

float integration_by_monte_carlo_uniform(float a, float b)
{
    int   n_samples = 10 * 40;
    float sum       = 0;
    for (int i = 0; i < n_samples; i++)
    {
        // float x   = a + randf() * (b - a);
        // float pdf = 1.0f / (b - a);
        auto [x, pdf]  = sample_uniform(a, b);
        float estimate = fx(x) / pdf;
        sum += estimate;
    }
    return sum / n_samples;
}

float integration_by_monte_carlo_linear(float a, float b)
{
    int   n_samples = 10 * 40;
    float sum       = 0;
    for (int i = 0; i < n_samples; i++)
    {
        auto [x, pdf]  = sample_linear(a, b);
        float estimate = fx(x) / pdf;
        sum += estimate;
    }
    return sum / n_samples;
}

float integration_by_monte_carlo_mis(float a, float b)
{
    int   n_samples = 10;
    float sum       = 0;
    for (int i = 0; i < n_samples; i++)
    {
        int n_count = 20;
        // balance heuristic
        float estimate = 0;
        for (int j = 0; j < n_count; j++)
        {
            auto [x1, pdf1]  = sample_uniform(a, b);
            float mis_weight = pdf1 / (pdf1 + pdf_linear(x1, a, b));
            estimate += mis_weight * fx(x1) / pdf1 / n_count;
        }
        for (int j = 0; j < n_count; j++)
        {
            auto [x2, pdf2]  = sample_linear(a, b);
            float mis_weight = pdf2 / (pdf_uniform(x2, a, b) + pdf2);
            estimate += mis_weight * fx(x2) / pdf2 / n_count;
        }
        sum += estimate;
    }
    return sum / n_samples;
}

float integration_by_monte_carlo_ris(
    float a, float b, const std::function<std::pair<float, float>(float, float)>& sampler)
{
    int   n_samples = 10;
    float sum       = 0;
    for (int i = 0; i < n_samples; i++)
    {
        std::vector<float> x;
        std::vector<float> w;
        int                M    = 2 * 20;  // sample pool size
        float              wsum = 0;

        for (int i = 0; i < M; i++)
        {
            // float xi = a + randf() * (b - a);
            // float pdf = 1.0f / (b - a);
            // auto [xi, pdf] = sample_uniform(a, b);
            auto [xi, pdf] = sampler(a, b);

            // p_hat(x) is replaced with f(x)
            // because p_hat(x) = k * f(x) where k is a normalizer
            // and k is cancelled in the estimator
            // besides, k is not known without integrating f(x)
            x.push_back(xi);
            float wi = fx(xi) / pdf;
            wsum += wi;
            w.push_back(wi);
        }

        // pdf for resampling
        for (int i = 0; i < w.size(); i++)
        {
            w[i] /= wsum;
        }
        // cdf for resampling
        for (int i = 1; i < w.size(); i++)
        {
            w[i] += w[i - 1];
        }
        // discrete sampling
        float sample = randf();
        int   idx    = 0;
        for (; idx < w.size(); idx++)
        {
            if (w[idx] > sample) break;
        }
        // float estimate = fx(x[idx]) / fx(x[idx]) * (wsum / M);
        float estimate = (wsum / M);
        sum += estimate;
    }
    return sum / n_samples;
}

float integration_by_monte_carlo_mis_ris(float a, float b)
{
    int   n_samples = 10;
    float sum       = 0;
    for (int i = 0; i < n_samples; i++)
    {
        // generate sample pool using mis
        int n_techniques = 2;
        int n_count      = 20;

        std::vector<float> x;
        std::vector<float> w;
        int                M    = n_techniques * n_count;  // sample pool size
        float              wsum = 0;

        // get effective pdf using balance heuristic
        for (int j = 0; j < n_count; j++)
        {
            auto [x1, pdf1]  = sample_uniform(a, b);
            float mis_weight = pdf1 / (pdf1 + pdf_linear(x1, a, b));
            float epdf1      = pdf1 / (mis_weight * n_techniques);

            x.push_back(x1);
            float wi = fx(x1) / epdf1;
            wsum += wi;
            w.push_back(wi);
        }
        for (int j = 0; j < n_count; j++)
        {
            auto [x2, pdf2]  = sample_linear(a, b);
            float mis_weight = pdf2 / (pdf_uniform(x2, a, b) + pdf2);
            float epdf2      = pdf2 / (mis_weight * n_techniques);

            x.push_back(x2);
            float wi = fx(x2) / epdf2;
            wsum += wi;
            w.push_back(wi);
        }

        // for (int i = 0; i < M; i++)
        //{
        //    // float xi = a + randf() * (b - a);
        //    // float pdf = 1.0f / (b - a);
        //    // auto [xi, pdf] = sample_uniform(a, b);
        //    // auto [xi, pdf] = sampler(a, b);

        //    x.push_back(xi);
        //    float wi = fx(xi) / pdf;
        //    wsum += wi;
        //    w.push_back(wi);
        //}

        // pdf
        for (int i = 0; i < w.size(); i++)
        {
            w[i] /= wsum;
        }
        // cdf
        for (int i = 1; i < w.size(); i++)
        {
            w[i] += w[i - 1];
        }
        // discrete sampling
        float sample = randf();
        int   idx    = 0;
        for (; idx < w.size(); idx++)
        {
            if (w[idx] > sample) break;
        }
        // float estimate = fx(x[idx]) / fx(x[idx]) * (wsum / M);
        float estimate = (wsum / M);
        sum += estimate;
    }
    return sum / n_samples;
}

//
// basically the same as the original ris, but use wrs for sampling
// from a streaming pool instead of generating the pool explicitly
//
float integration_by_monte_carlo_streaming_ris(
    float a, float b, const std::function<std::pair<float, float>(float, float)>& sampler)
{
    int   n_samples = 10;  // number of samples for shading
    float sum       = 0;
    for (int i = 0; i < n_samples; i++)
    {
        int M = 2 * 20;  // sample pool size, number of candidates

        Reservoir r;

        for (int j = 0; j < M; j++)
        {
            // target pdf is fx
            auto [xi, pdf] = sampler(a, b);

            // p_hat(x) is replaced with f(x)
            // because p_hat(x) = k * f(x) where k is a normalizer
            // and k is cancelled in the estimator
            // besides, k is not known without integrating f(x)
            r.update(xi, fx(xi) / pdf);
        }

        // target pdf is fx
        // float estimate = fx(r.y) / fx(r.y) * (wsum / r.M);
        float estimate = (r.wsum / r.M);
        sum += estimate;
    }
    return sum / n_samples;
}

float integration_by_monte_carlo_multi_streaming_ris(
    float a, float b, const std::function<std::pair<float, float>(float, float)>& sampler)
{
    int   n_samples = 10;
    float sum       = 0;
    for (int i = 0; i < n_samples; i++)
    {
        int num_reservoirs = 5;
        int M              = 2 * 20 / num_reservoirs;  // sample pool size

        std::vector<Reservoir> res(num_reservoirs);

        for (int k = 0; k < num_reservoirs; k++)
        {
            for (int j = 0; j < M; j++)
            {
                // target pdf is fx
                auto [xi, pdf] = sampler(a, b);
                res[k].update(xi, fx(xi) / pdf);
            }
            res[k].W = 1.0f / fx(res[k].y) * (res[k].wsum / res[k].M);
        }

        // combine reservoirs
        Reservoir s;
        for (int k = 0; k < num_reservoirs; k++)
        {
            s.update(res[k].y, fx(res[k].y) * res[k].W * res[k].M);
        }
        s.M = 0;
        for (int k = 0; k < num_reservoirs; k++)
        {
            s.M += res[k].M;
        }
        s.W = 1.0f / fx(s.y) * (s.wsum / s.M);

        // target pdf is fx
        // float estimate = fx(s.y) / fx(s.y) * (s.wsum / s.M);
        float estimate = (s.wsum / s.M);
        sum += estimate;
    }
    return sum / n_samples;
}

float variance(const std::vector<float>& data)
{
    float sum = 0;
    for (int i = 0; i < data.size(); i++)
    {
        sum += data[i];
    }
    float avg = sum / data.size();

    sum = 0;
    for (int i = 0; i < data.size(); i++)
    {
        sum += (data[i] - avg) * (data[i] - avg);
    }
    return sum / (data.size() - 1);
}

float rmse(const std::vector<float>& data, float ref)
{
    float sum = 0;
    for (int i = 0; i < data.size(); i++)
    {
        sum += (data[i] - ref) * (data[i] - ref);
    }
    return sqrt(sum / data.size());
}

struct Result
{
    std::string method;
    float       rmse_score;

    static bool compare(const Result& a, const Result& b) { return a.rmse_score < b.rmse_score; }
};

int main()
{
    // leaderboard
    std::vector<Result> results;

    float ref = (float)M_1_PI;
    cout << "ref integral of f(x) = " << ref << endl;

    float f1 = integration_by_summation(0.0f, 1.0f);
    cout << "integral by sum = " << f1 << endl;

    constexpr int num_trials = 10000;

    // mc uniform
    {
        sample_clear();

        std::vector<float> mc;
        for (int i = 0; i < num_trials; i++)
        {
            float f = integration_by_monte_carlo_uniform(0.0f, 1.0f);
            mc.push_back(f);
        }
        // cout << fmt::format("mc std = {}", sqrt(variance(mc))) << endl;

        float rmse_score = rmse(mc, ref);
        cout << "mc uniform rmse = " << rmse_score << endl;

        sample_report();

        results.push_back(Result{"mc uniform", rmse_score});
    }

    // mc linear
    {
        sample_clear();

        std::vector<float> mc;
        for (int i = 0; i < num_trials; i++)
        {
            float f = integration_by_monte_carlo_linear(0.0f, 1.0f);
            mc.push_back(f);
        }

        float rmse_score = rmse(mc, ref);
        cout << "mc linear rmse = " << rmse_score << endl;

        sample_report();

        results.push_back(Result{"mc linear", rmse_score});
    }

    // mis is better than ris with uniform pdf,
    // but slightly worse than ris with linear pdf,
    // using the same number of samples
    {
        sample_clear();

        // mis
        std::vector<float> mc_mis;
        for (int i = 0; i < num_trials; i++)
        {
            float f = integration_by_monte_carlo_mis(0.0f, 1.0f);
            mc_mis.push_back(f);
        }

        float rmse_score = rmse(mc_mis, ref);
        cout << "mc mis rmse = " << rmse_score << endl;

        sample_report();

        results.push_back(Result{"mc mis", rmse_score});
    }

    // ris uniform
    {
        sample_clear();

        std::vector<float> mc_ris;
        for (int i = 0; i < num_trials; i++)
        {
            float f = integration_by_monte_carlo_ris(0.0f, 1.0f, sample_uniform);
            mc_ris.push_back(f);
        }

        float rmse_score = rmse(mc_ris, ref);
        cout << "mc ris uniform rmse = " << rmse_score << endl;

        sample_report();

        results.push_back(Result{"ris uniform", rmse_score});
    }

    // ris linear
    {
        sample_clear();

        std::vector<float> mc_ris;
        for (int i = 0; i < num_trials; i++)
        {
            float f = integration_by_monte_carlo_ris(0.0f, 1.0f, sample_linear);
            mc_ris.push_back(f);
        }

        float rmse_score = rmse(mc_ris, ref);
        cout << "mc ris linear rmse = " << rmse_score << endl;

        sample_report();

        results.push_back(Result{"ris linear", rmse_score});
    }

    // mis ris (slightly better than mis with high n_count, but worse with low
    // n_count)
    {
        sample_clear();

        std::vector<float> mc_mis_ris;
        for (int i = 0; i < num_trials; i++)
        {
            float f = integration_by_monte_carlo_mis_ris(0.0f, 1.0f);
            mc_mis_ris.push_back(f);
        }

        float rmse_score = rmse(mc_mis_ris, ref);
        cout << "mc mis ris rmse = " << rmse_score << endl;

        sample_report();

        results.push_back(Result{"mis ris", rmse_score});
    }

    // streaming ris (similar rmse as original ris)
    // uniform
    {
        sample_clear();

        std::vector<float> mc_streaming_ris;
        for (int i = 0; i < num_trials; i++)
        {
            float f = integration_by_monte_carlo_streaming_ris(0.0f, 1.0f, sample_uniform);
            mc_streaming_ris.push_back(f);
        }

        float rmse_score = rmse(mc_streaming_ris, ref);
        cout << "mc streaming ris uniform rmse = " << rmse_score << endl;

        sample_report();

        results.push_back(Result{"streaming ris uniform", rmse_score});
    }

    // linear
    {
        sample_clear();

        std::vector<float> mc_streaming_ris;
        for (int i = 0; i < num_trials; i++)
        {
            float f = integration_by_monte_carlo_streaming_ris(0.0f, 1.0f, sample_linear);
            mc_streaming_ris.push_back(f);
        }

        float rmse_score = rmse(mc_streaming_ris, ref);
        cout << "mc streaming ris linear rmse = " << rmse_score << endl;

        sample_report();

        results.push_back(Result{"streaming ris linear", rmse_score});
    }

    // multi streaming ris (the same as ris at the same sample count)
    // uniform
    {
        sample_clear();

        std::vector<float> mc_multi_streaming_ris;
        for (int i = 0; i < num_trials; i++)
        {
            float f = integration_by_monte_carlo_multi_streaming_ris(0.0f, 1.0f, sample_uniform);
            mc_multi_streaming_ris.push_back(f);
        }

        float rmse_score = rmse(mc_multi_streaming_ris, ref);
        cout << "mc multi streaming ris uniform rmse = " << rmse_score << endl;

        sample_report();

        results.push_back(Result{"multi streaming ris uniform", rmse_score});
    }

    // linear
    {
        sample_clear();

        std::vector<float> mc_multi_streaming_ris;
        for (int i = 0; i < num_trials; i++)
        {
            float f = integration_by_monte_carlo_multi_streaming_ris(0.0f, 1.0f, sample_linear);
            mc_multi_streaming_ris.push_back(f);
        }

        float rmse_score = rmse(mc_multi_streaming_ris, ref);
        cout << "mc multi streaming ris linear rmse = " << rmse_score << endl;

        sample_report();

        results.push_back(Result{"multi streaming ris linear", rmse_score});
    }

    std::sort(results.begin(), results.end(), Result::compare);
    for (auto& c : results)
    {
        cout << c.rmse_score << " \t : \t " << c.method << endl;
    }

    return 0;
}
}  // namespace monte_carlo

// MC rendering using RESTIR
namespace restir
{
class Image
{
public:
    int                width, height;
    std::vector<float> buffer;

    Image(int width, int height) : width(width), height(height)
    {
        buffer.resize(width * height * 3);
    }

    void set_pixel(int row, int col, float r, float g, float b)
    {
        int offset         = (col + row * width) * 3;
        buffer[offset + 0] = r;
        buffer[offset + 1] = g;
        buffer[offset + 2] = b;
    }

    void get_pixel(int row, int col, float* r, float* g, float* b) const
    {
        int offset = (col + row * width) * 3;
        *r         = buffer[offset + 0];
        *g         = buffer[offset + 1];
        *b         = buffer[offset + 2];
    }

    void clear(float r, float g, float b)
    {
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                this->set_pixel(row, col, r, g, b);
            }
        }
    }
};

void dump_ppm(const char* filename, const Image& image)
{
    // Save result to a PPM image (keep these flags if you compile under
    // Windows) NOTE::especially std:ios::binary which is equivalent to "wb" in
    // fprintf()
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    if (!ofs.is_open())
    {
        std::cout << "cannot write to " << filename << std::endl;
        return;
    }

    float r, g, b;

    ofs << "P6\n" << image.width << " " << image.height << "\n255\n";
    for (int j = image.height - 1; j >= 0; --j)
    {
        for (int i = 0; i < image.width; ++i)
        {
            image.get_pixel(j, i, &r, &g, &b);
            ofs << (unsigned char)(std::max(0.0f, std::min(1.0f, r)) * 255)
                << (unsigned char)(std::max(0.0f, std::min(1.0f, g)) * 255)
                << (unsigned char)(std::max(0.0f, std::min(1.0f, b)) * 255);
        }
    }
    ofs.close();
}

class Ray
{
public:
    glm::vec3 orig, dir;
};

class PointLight
{
public:
    glm::vec3 position;
    glm::vec3 emission;
};

class QuadLight
{
public:
    glm::mat4 xform;
    glm::mat4 xform_n;
    float     pdf_A;
    glm::vec3 emission;

    QuadLight(const glm::mat4& xform_, const glm::vec3& emission_)
        : xform(xform_), emission(emission_)
    {
        xform_n        = glm::transpose(glm::inverse(xform));
        glm::vec3 v0   = sample(0.0f, 0.0f);
        glm::vec3 v1   = sample(1.0f, 0.0f);
        glm::vec3 v2   = sample(0.0f, 1.0f);
        float     area = abs(glm::length(glm::cross(v1 - v0, v2 - v0)));
        pdf_A          = 1.0f / area;
    }

    glm::vec3 sample(float x, float y) const
    {
        glm::vec4 p(x * 2 - 1, 0.0f, y * 2 - 1, 1.0f);
        return glm::vec3(xform * p);
    }
    glm::vec3 normal() const
    {
        glm::vec4 n(0.0f, 1.0f, 0.0f, 0.0f);
        return glm::normalize(glm::vec3(xform_n * n));
    }
    float pdf_area() const { return pdf_A; }
};

class Material
{
public:
    glm::vec3 albedo   = glm::vec3(1, 1, 1);
    glm::vec3 emission = glm::vec3(0, 0, 0);
};

struct Sphere
{
    float     rad;  // radius
    glm::vec3 p;
    Material  mat;

    Sphere(float rad_, glm::vec3 p_, const Material& mat)
    {
        this->rad = rad_;
        this->p   = p_;
        this->mat = mat;
    }
    float intersect(const Ray& r, float ray_eps) const
    {
        glm::vec3 op = p - r.orig;
        float t, eps = ray_eps, b = glm::dot(op, r.dir), det = b * b - glm::dot(op, op) + rad * rad;
        if (det < 0)
            return 0;
        else
            det = sqrt(det);
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    }
};
class Scene
{
public:
    Scene()
    {
        Material white_mat;
        white_mat.albedo   = glm::vec3(0.75, 0.75, 0.75);
        white_mat.emission = glm::vec3(0, 0, 0);

        Material green_mat;
        green_mat.albedo   = glm::vec3(0.25, 0.75, 0.25);
        green_mat.emission = glm::vec3(0, 0, 0);

        spheres.push_back(Sphere(10, glm::vec3(20, 0, 0), green_mat));
        spheres.push_back(Sphere(1e3 - 1, glm::vec3(0, -1e3, 0), white_mat));
        ;

        // point_lights.push_back(PointLight{glm::vec3(20, 20, 20), glm::vec3(20000, 20000,
        // 20000)});
        if (0)
        {
            glm::mat4 xform = glm::mat4(1.0f);
            xform           = glm::scale(glm::mat4(1.0f), glm::vec3(0.02f, 0.02f, 0.02f)) * xform;
            xform =
                glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)) *
                xform;
            xform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 15.0f, 0.0f)) * xform;
            quad_lights.push_back(QuadLight(xform, glm::vec3(20, 20, 20) * 500.0f));
        }

        if (0)
        {
            glm::vec3 e(1, 1, 1);
            glm::mat4 xform = glm::mat4(1.0f);
            xform           = glm::scale(glm::mat4(1.0f), glm::vec3(25.0f, 25.0f, 25.0f)) * xform;
            xform =
                glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)) *
                xform;
            xform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 15.0f, 0.0f)) * xform;
            quad_lights.push_back(QuadLight(xform, e));
        }
        else
        {
            glm::vec3 e(1, 1, 1);
            {
                glm::mat4 xform = glm::mat4(1.0f);
                xform = glm::scale(glm::mat4(1.0f), glm::vec3(25.0f, 25.0f, 25.0f) * 0.5f) * xform;
                xform = glm::rotate(
                            glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)) *
                        xform;
                xform = glm::translate(glm::mat4(1.0f), glm::vec3(12.5f, 15.0f, 12.5f)) * xform;
                quad_lights.push_back(QuadLight(xform, e));
            }
            {
                glm::mat4 xform = glm::mat4(1.0f);
                xform = glm::scale(glm::mat4(1.0f), glm::vec3(25.0f, 25.0f, 25.0f) * 0.5f) * xform;
                xform = glm::rotate(
                            glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)) *
                        xform;
                xform = glm::translate(glm::mat4(1.0f), glm::vec3(12.5f, 15.0f, -12.5f)) * xform;
                quad_lights.push_back(QuadLight(xform, e));
            }
            {
                glm::mat4 xform = glm::mat4(1.0f);
                xform = glm::scale(glm::mat4(1.0f), glm::vec3(25.0f, 25.0f, 25.0f) * 0.5f) * xform;
                xform = glm::rotate(
                            glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)) *
                        xform;
                xform = glm::translate(glm::mat4(1.0f), glm::vec3(-12.5f, 15.0f, 12.5f)) * xform;
                quad_lights.push_back(QuadLight(xform, e));
            }
            {
                glm::mat4 xform = glm::mat4(1.0f);
                xform = glm::scale(glm::mat4(1.0f), glm::vec3(25.0f, 25.0f, 25.0f) * 0.5f) * xform;
                xform = glm::rotate(
                            glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)) *
                        xform;
                xform = glm::translate(glm::mat4(1.0f), glm::vec3(-12.5f, 15.0f, -12.5f)) * xform;
                quad_lights.push_back(QuadLight(xform, e));
            }
        }
    }

    bool intersect(
        const Ray& ray, float ray_eps, float* hit_t, glm::vec3* hit_nl, Material* hit_mat) const
    {
        float closest = 1e10f;

        for (auto& s : spheres)
        {
            float t = s.intersect(ray, ray_eps);
            if (t > ray_eps && t < closest)
            {
                closest  = t;
                *hit_t   = closest;
                *hit_nl  = glm::normalize(ray.orig + ray.dir * t - s.p);
                *hit_mat = s.mat;
            }
        }

        return closest < 1e10f;
    }

    bool occluded(const Ray& ray, float ray_eps, float max_t) const
    {
        float closest = 1e10f;

        for (auto& s : spheres)
        {
            float t = s.intersect(ray, ray_eps);
            if (t > ray_eps && t < max_t)
            {
                closest = t;
                break;
            }
        }

        return closest < 1e10f;
    }

    std::vector<PointLight> point_lights;
    std::vector<QuadLight>  quad_lights;
    std::vector<Sphere>     spheres;
};

class Camera
{
public:
    glm::vec3 orig;
    glm::vec3 dir;
    glm::vec3 up;
    glm::vec3 right;
    float     fovy = 60.0f;

    Camera(const glm::vec3& orig, const glm::vec3& target, const glm::vec3& up)
    {
        this->orig  = orig;
        this->dir   = glm::normalize(target - orig);
        this->up    = glm::normalize(up);
        this->right = glm::cross(this->dir, this->up);
        this->up    = glm::cross(this->right, this->dir);
    }
};

Ray cast_primary(int row, int col, const Camera& camera, const Image& image)
{
    Ray ray;

    float     sy  = tan(camera.fovy * (M_PI / 360.0f));
    float     sx  = float(image.width) / float(image.height) * sy;
    float     x   = (col + 0.5) / image.width * 2.0f - 1.0f;
    float     y   = (row + 0.5) / image.height * 2.0f - 1.0f;
    glm::vec3 dir = camera.right * x * sx + camera.up * y * sy + camera.dir;
    ray.dir       = glm::normalize(dir);
    ray.orig      = camera.orig;

    return ray;
}

void dump_points(const std::vector<glm::vec3>& pts, const std::string& filename)
{
    std::ofstream ofs(filename);
    for (auto& p : pts)
    {
        ofs << p.x << ";" << p.y << ";" << p.z << std::endl;
    }
    ofs.close();
}

class Renderer
{
public:
    class PathSample
    {
    public:
        glm::vec3 hit_p;
        glm::vec3 hit_nl;
        Material  hit_mat;
        glm::vec3 light_p;
        glm::vec3 light_nl;
        glm::vec3 light_e;
    };

    static float to_gray(const glm::vec3& v) { return v.x * 0.3f + v.y * 0.6f + v.z * 0.1f; }

    // unshadowed path contribution
    float p_hat(const Scene& scene, const PathSample& path, bool clamp_zero)
    {
        float w = 0;

        glm::vec3 dist      = path.light_p - path.hit_p;
        glm::vec3 light_dir = glm::normalize(dist);

        float cos_hit   = glm::dot(path.hit_nl, light_dir);
        float cos_light = glm::dot(path.light_nl, -light_dir);
        float G         = cos_hit * cos_light / glm::dot(dist, dist);
        if (cos_hit > 0 && cos_light > 0)
        {
            glm::vec3 R = path.hit_mat.albedo / float(M_PI);
            glm::vec3 E = path.light_e;
            w           = to_gray(G * R * E);
        }

        // p_hat can be arbitrary
        // so work around zero p_hat by clamping
        if (clamp_zero)
        {
            return std::max(w, 1e-7f);
        }
        // or ignore the invalid sample when shading
        else
        {
            return w;
        }
    }

    glm::vec3 shade_path(const Scene& scene, const PathSample& path)
    {
        glm::vec3 c(0.0f, 0.0f, 0.0f);

        glm::vec3 dist      = path.light_p - path.hit_p;
        glm::vec3 light_dir = glm::normalize(dist);

        float cos_hit   = glm::dot(path.hit_nl, light_dir);
        float cos_light = glm::dot(path.light_nl, -light_dir);
        float G         = cos_hit * cos_light / glm::dot(dist, dist);
        if (cos_hit > 0 && cos_light > 0)
        {
            glm::vec3 R = path.hit_mat.albedo / float(M_PI);
            glm::vec3 E = path.light_e;
            float V = scene.occluded(Ray{path.hit_p, light_dir}, 0.005f, glm::length(dist) - 0.005f)
                          ? 0.0f
                          : 1.0f;
            c       = G * R * E * V;
        }
        return c;
    }

    void render_restir_alg3(const Scene&  scene,
                            const Camera& camera,
                            Image*        image,
                            const char*   filename)
    {
        image->clear(0.0f, 0.0f, 0.0f);

        omp_set_num_threads(omp_get_max_threads() - 1);

        for (int row = 0; row < image->height; row++)
        {
            printf("%d / %d\n", row, image->height);

#pragma omp parallel for
            for (int col = 0; col < image->width; col++)
            {
                Ray cam_ray = cast_primary(row, col, camera, *image);

                float     hit_t;
                glm::vec3 hit_nl;
                Material  hit_mat;
                if (!scene.intersect(cam_ray, 0.001f, &hit_t, &hit_nl, &hit_mat))
                {
                    continue;
                }

                // do lighting
                glm::vec3 hit_p = cam_ray.orig + cam_ray.dir * hit_t;
                glm::vec3 I(0, 0, 0);

                int  nsamples   = 100;
                int  M          = 10;
                bool clamp_zero = false;

                for (int sample = 0; sample < nsamples; sample++)
                {
                    FastSampler rng(row, col, sample);

                    ReservoirT<PathSample> r;

                    // sample lights
                    for (int i = 0; i < M; i++)
                    {
                        int num_lights = scene.quad_lights.size();

                        // pick light to sample from
                        Reservoir r0;
                        for (int i = 0; i < num_lights; i++) r0.update(1.0f, 1.0f);
                        int   ii = r0.yi;
                        auto& pl = scene.quad_lights[ii];

                        float pdf_L = 1.0f / num_lights;
                        float pdf_A = pl.pdf_area();
                        float pdf   = pdf_A * pdf_L;

                        glm::vec3 light_nl = pl.normal();
                        glm::vec3 light_p  = pl.sample(rng.next(), rng.next());

                        PathSample p;
                        p.hit_p    = hit_p;
                        p.hit_nl   = hit_nl;
                        p.hit_mat  = hit_mat;
                        p.light_p  = light_p;
                        p.light_nl = light_nl;
                        p.light_e  = pl.emission;

                        float w = p_hat(scene, p, clamp_zero);
                        // if (w > 0) // whether or not update for zero p_hat
                        {
                            r.update(p, w / pdf);
                        }
                    }

                    float w = p_hat(scene, r.y, clamp_zero);

                    // work around zero p_hat
                    // by ignoring invalid sample when shading
                    // otherwise clamp p_hat
                    if (w > 0)
                    {
                        // if update for zero p_hat
                        r.W = 1.0f / w * (r.wsum / r.M);

                        // if not update for zero p_hat
                        // r.W = 1.0f / p_hat(scene, r.y) * (r.wsum / M);

                        I += shade_path(scene, r.y) * r.W;
                    }
                }

                I /= nsamples;

                image->set_pixel(row, col, I.x, I.y, I.z);
            }
        }

        dump_ppm(filename, *image);
    }

    void render_restir_alg4(const Scene&  scene,
                            const Camera& camera,
                            Image*        image,
                            const char*   filename)
    {
        image->clear(0.0f, 0.0f, 0.0f);

        omp_set_num_threads(omp_get_max_threads() - 1);

        for (int row = 0; row < image->height; row++)
        {
            printf("%d / %d\n", row, image->height);

#pragma omp parallel for
            for (int col = 0; col < image->width; col++)
            {
                Ray cam_ray = cast_primary(row, col, camera, *image);

                float     hit_t;
                glm::vec3 hit_nl;
                Material  hit_mat;
                if (!scene.intersect(cam_ray, 0.001f, &hit_t, &hit_nl, &hit_mat))
                {
                    continue;
                }

                // do lighting
                glm::vec3 hit_p = cam_ray.orig + cam_ray.dir * hit_t;
                glm::vec3 I(0, 0, 0);

                int  nsamples   = 1000;
                int  M          = 3;
                bool clamp_zero = false;

                for (int sample = 0; sample < nsamples; sample++)
                {
                    FastSampler rng(row, col, sample);

                    std::vector<ReservoirT<PathSample>> rr(3);

                    for (auto& r : rr)
                    {
                        // sample lights
                        for (int i = 0; i < M; i++)
                        {
                            int num_lights = scene.quad_lights.size();

                            // pick light to sample from
                            Reservoir r0;
                            for (int i = 0; i < num_lights; i++) r0.update(1.0f, 1.0f);
                            int   ii = r0.yi;
                            auto& pl = scene.quad_lights[ii];

                            float pdf_L = 1.0f / num_lights;
                            float pdf_A = pl.pdf_area();
                            float pdf   = pdf_A * pdf_L;

                            glm::vec3 light_nl = pl.normal();
                            glm::vec3 light_p  = pl.sample(rng.next(), rng.next());

                            PathSample p;
                            p.hit_p    = hit_p;
                            p.hit_nl   = hit_nl;
                            p.hit_mat  = hit_mat;
                            p.light_p  = light_p;
                            p.light_nl = light_nl;
                            p.light_e  = pl.emission;

                            float w = p_hat(scene, p, clamp_zero);
                            // if (w > 0) // whether or not update for zero p_hat
                            {
                                r.update(p, w / pdf);
                            }
                        }

                        float w = p_hat(scene, r.y, clamp_zero);

                        // work around zero p_hat
                        // by ignoring invalid sample when shading
                        // otherwise clamp p_hat
                        if (w > 0)
                        {
                            // if update for zero p_hat
                            r.W = 1.0f / w * (r.wsum / r.M);

                            // if not update for zero p_hat
                            // r.W = 1.0f / p_hat(scene, r.y) * (r.wsum / M);
                        }
                    }

                    ReservoirT<PathSample> s;
                    for (auto& r : rr)
                    {
                        float w = p_hat(scene, r.y, clamp_zero);
                        if (w > 0)
                        {
                            s.update(r.y, w * r.W * r.M);
                        }
                    }
                    s.M = 0;
                    for (auto& r : rr)
                    {
                        s.M += r.M;
                    }

                    float w = p_hat(scene, s.y, clamp_zero);

                    // work around zero p_hat
                    // by ignoring invalid sample when shading
                    // otherwise clamp p_hat
                    if (w > 0)
                    {
                        // if update for zero p_hat
                        s.W = 1.0f / w * (s.wsum / s.M);

                        // if not update for zero p_hat
                        // r.W = 1.0f / p_hat(scene, r.y) * (r.wsum / M);

                        I += shade_path(scene, s.y) * s.W;
                    }
                }

                I /= nsamples;

                image->set_pixel(row, col, I.x, I.y, I.z);
            }
        }

        dump_ppm(filename, *image);
    }

    void render_restir_alg4_spatial_reuse(const Scene&  scene,
                                          const Camera& camera,
                                          Image*        image,
                                          const char*   filename)
    {
        int  nsamples   = 10;
        bool clamp_zero = false;
        int  M          = 32;

        image->clear(0.0f, 0.0f, 0.0f);

        omp_set_num_threads(omp_get_max_threads() - 1);

        for (int sample = 0; sample < nsamples; sample++)
        {
            auto r_image   = new std::vector<ReservoirT<PathSample>>(image->width * image->height);
            auto r_image_2 = new std::vector<ReservoirT<PathSample>>(image->width * image->height);
            std::vector<glm::vec3> nl_image(image->width * image->height);
            std::vector<glm::vec3> pos_image(image->width * image->height);

            for (int row = 0; row < image->height; row++)
            {
                printf("[s %d] %d / %d\n", sample, row, image->height);

#pragma omp parallel for
                for (int col = 0; col < image->width; col++)
                {
                    auto& r = (*r_image)[col + row * image->width];
                    r.reset();

                    Ray cam_ray = cast_primary(row, col, camera, *image);

                    float     hit_t;
                    glm::vec3 hit_nl;
                    Material  hit_mat;
                    if (!scene.intersect(cam_ray, 0.001f, &hit_t, &hit_nl, &hit_mat))
                    {
                        continue;
                    }

                    // do lighting
                    glm::vec3 hit_p = cam_ray.orig + cam_ray.dir * hit_t;

                    nl_image[col + row * image->width]  = hit_nl;
                    pos_image[col + row * image->width] = hit_p;

                    FastSampler rng(row, col, sample);

                    // sample lights
                    for (int i = 0; i < M; i++)
                    {
                        int num_lights = scene.quad_lights.size();

                        // pick light to sample from
                        Reservoir r0;
                        for (int i = 0; i < num_lights; i++) r0.update(1.0f, 1.0f);
                        int   ii = r0.yi;
                        auto& pl = scene.quad_lights[ii];

                        float pdf_L = 1.0f / num_lights;
                        float pdf_A = pl.pdf_area();
                        float pdf   = pdf_A * pdf_L;

                        glm::vec3 light_nl = pl.normal();
                        glm::vec3 light_p  = pl.sample(rng.next(), rng.next());

                        PathSample p;
                        p.hit_p    = hit_p;
                        p.hit_nl   = hit_nl;
                        p.hit_mat  = hit_mat;
                        p.light_p  = light_p;
                        p.light_nl = light_nl;
                        p.light_e  = pl.emission;

                        float w = p_hat(scene, p, clamp_zero);
                        // if (w > 0) // whether or not update for zero p_hat
                        {
                            r.update(p, w / pdf);
                        }
                    }

                    float w = p_hat(scene, r.y, clamp_zero);

                    // work around zero p_hat
                    // by ignoring invalid sample when shading
                    // otherwise clamp p_hat
                    if (w > 0)
                    {
                        // if update for zero p_hat
                        r.W = 1.0f / w * (r.wsum / r.M);

                        // if not update for zero p_hat
                        // r.W = 1.0f / p_hat(scene, r.y) * (r.wsum / M);
                    }
                }
            }

            // spatial reuse
            for (int iter = 0; iter < 4; iter++)
            {
                for (int row = 0; row < image->height; row++)
                {
                    printf("[%d] %d / %d\n", iter, row, image->height);

#pragma omp parallel for
                    for (int col = 0; col < image->width; col++)
                    {
                        FastSampler rng(row, col, nsamples + iter);

                        // if ((*r_image)[col + row * image->width].W > 0)
                        {
                            auto& s = (*r_image_2)[col + row * image->width];
                            s.reset();

                            int                     M_ = 0;
                            int                     K  = 5;
                            std::vector<glm::ivec2> nn_pixels(K);
                            for (int k = 0; k < K; k++)
                            {
                                float radius = sqrt(rng.next()) * 10;
                                float angle  = M_PI * 2 * rng.next();
                                nn_pixels[k] =
                                    glm::ivec2(int(radius * sin(angle)), int(radius * cos(angle)));
                            }
                            nn_pixels.push_back(glm::ivec2(0, 0));

                            for (auto& pixel : nn_pixels)
                            {
                                int row_ = std::max(0, std::min(image->height - 1, row + pixel.x));
                                int col_ = std::max(0, std::min(image->width - 1, col + pixel.y));

                                int n  = col + row * image->width;
                                int nn = col_ + row_ * image->width;

                                auto& r = (*r_image)[nn];  // neighbor

                                bool close_nl = glm::dot(nl_image[n], nl_image[nn]) > 0.95f;
                                // bool close_pos = glm::length(pos_image[n] - pos_image[nn]) <
                                // 0.05f;
                                bool close_pos   = glm::length(pos_image[n] - pos_image[nn]) < 1.0f;
                                bool close_valid = r.W > 0;
                                bool use_neighbor = close_nl && close_pos && close_valid;
                                // bool use_neighbor = close_nl && close_valid;
                                // bool use_neighbor = close_valid;

                                if (use_neighbor)
                                {
                                    float w = p_hat(scene, r.y, clamp_zero);
                                    if (w > 0)
                                    {
                                        s.update(r.y, w * r.W * r.M);
                                        M_ += r.M;
                                    }
                                }
                            }
                            s.M = M_;

                            if (s.M > 0)
                            {
                                float w = p_hat(scene, s.y, clamp_zero);

                                // work around zero p_hat
                                // by ignoring invalid sample when shading
                                // otherwise clamp p_hat
                                if (w > 0)
                                {
                                    // if update for zero p_hat
                                    s.W = 1.0f / w * (s.wsum / s.M);

                                    // if not update for zero p_hat
                                    // r.W = 1.0f / p_hat(scene, r.y) * (r.wsum / M);
                                }
                            }
                        }
                    }
                }

                std::swap(r_image, r_image_2);
            }

            for (int row = 0; row < image->height; row++)
            {
                printf("[c] %d / %d\n", row, image->height);

#pragma omp parallel for
                for (int col = 0; col < image->width; col++)
                {
                    glm::vec3 I(0, 0, 0);
                    image->get_pixel(row, col, &I.x, &I.y, &I.z);

                    auto& s = (*r_image)[col + row * image->width];

                    I += shade_path(scene, s.y) * (s.W / nsamples);

                    image->set_pixel(row, col, I.x, I.y, I.z);
                }
            }

            delete r_image;
            delete r_image_2;
        }

        dump_ppm(filename, *image);
    }

//    void render_restir_alg5(const Scene&  scene,
//                            const Camera& camera,
//                            Image*        image,
//                            const char*   filename)
//    {
//        int  nsamples   = 1;
//        bool clamp_zero = false;
//        int  M          = 32;
//
//        image->clear(0.0f, 0.0f, 0.0f);
//
//        auto r_image   = new std::vector<ReservoirT<PathSample>>(image->width * image->height);
//        auto r_image_2 = new std::vector<ReservoirT<PathSample>>(image->width * image->height);
//
//        omp_set_num_threads(omp_get_max_threads() - 1);
//
//        for (int sample = 0; sample < nsamples; sample++)
//        {
//            std::vector<glm::vec3> nl_image(image->width * image->height);
//            std::vector<glm::vec3> pos_image(image->width * image->height);
//
//            for (int row = 0; row < image->height; row++)
//            {
//                printf("[s %d] %d / %d\n", sample, row, image->height);
//
//#pragma omp parallel for
//                for (int col = 0; col < image->width; col++)
//                {
//                    auto& r = (*r_image)[col + row * image->width];
//                    r.reset();
//
//                    Ray cam_ray = cast_primary(row, col, camera, *image);
//
//                    float     hit_t;
//                    glm::vec3 hit_nl;
//                    Material  hit_mat;
//                    if (!scene.intersect(cam_ray, 0.001f, &hit_t, &hit_nl, &hit_mat))
//                    {
//                        continue;
//                    }
//
//                    // do lighting
//                    glm::vec3 hit_p = cam_ray.orig + cam_ray.dir * hit_t;
//
//                    nl_image[col + row * image->width]  = hit_nl;
//                    pos_image[col + row * image->width] = hit_p;
//
//                    FastSampler rng(row, col, sample);
//
//                    // sample lights
//                    for (int i = 0; i < M; i++)
//                    {
//                        int num_lights = scene.quad_lights.size();
//
//                        // pick light to sample from
//                        Reservoir r0;
//                        for (int i = 0; i < num_lights; i++) r0.update(1.0f, 1.0f);
//                        int   ii = r0.yi;
//                        auto& pl = scene.quad_lights[ii];
//
//                        float pdf_L = 1.0f / num_lights;
//                        float pdf_A = pl.pdf_area();
//                        float pdf   = pdf_A * pdf_L;
//
//                        glm::vec3 light_nl = pl.normal();
//                        glm::vec3 light_p  = pl.sample(rng.next(), rng.next());
//
//                        PathSample p;
//                        p.hit_p    = hit_p;
//                        p.hit_nl   = hit_nl;
//                        p.hit_mat  = hit_mat;
//                        p.light_p  = light_p;
//                        p.light_nl = light_nl;
//                        p.light_e  = pl.emission;
//
//                        float w = p_hat(scene, p, clamp_zero);
//                        // if (w > 0) // whether or not update for zero p_hat
//                        {
//                            r.update(p, w / pdf);
//                        }
//                    }
//
//                    float w = p_hat(scene, r.y, clamp_zero);
//
//                    // work around zero p_hat
//                    // by ignoring invalid sample when shading
//                    // otherwise clamp p_hat
//                    if (w > 0)
//                    {
//                        // if update for zero p_hat
//                        r.W = 1.0f / w * (r.wsum / r.M);
//
//                        // if not update for zero p_hat
//                        // r.W = 1.0f / p_hat(scene, r.y) * (r.wsum / M);
//                    }
//                }
//            }
//
//            for (int row = 0; row < image->height; row++)
//            {
//#pragma omp parallel for
//                for (int col = 0; col < image->width; col++)
//                {
//                    auto& r    = (*r_image)[col + row * image->width];
//                    auto& path = r.y;
//
//                    glm::vec3 dist      = path.light_p - path.hit_p;
//                    glm::vec3 light_dir = glm::normalize(dist);
//
//                    if (scene.occluded(
//                            Ray{path.hit_p, light_dir}, 0.005f, glm::length(dist) - 0.005f))
//                    {
//                        r.reset();
//                    }
//                }
//            }
//
//            // spatial reuse
//            for (int iter = 0; iter < 4; iter++)
//            {
//                for (int row = 0; row < image->height; row++)
//                {
//                    printf("[%d] %d / %d\n", iter, row, image->height);
//
//#pragma omp parallel for
//                    for (int col = 0; col < image->width; col++)
//                    {
//                        // if ((*r_image)[col + row * image->width].W > 0)
//                        {
//                            auto& s = (*r_image_2)[col + row * image->width];
//                            s.reset();
//
//                            int M_ = 0;
//                            for (int r_ = -1; r_ <= 1; r_++)
//                            {
//                                for (int c_ = -1; c_ <= 1; c_++)
//                                {
//                                    int row_ = std::max(0, std::min(image->height - 1, row + r_));
//                                    int col_ = std::max(0, std::min(image->width - 1, col + c_));
//
//                                    int n  = col + row * image->width;
//                                    int nn = col_ + row_ * image->width;
//
//                                    auto& r = (*r_image)[nn];  // neighbor
//
//                                    bool close_nl = glm::dot(nl_image[n], nl_image[nn]) > 0.95f;
//                                    // bool close_pos = glm::length(pos_image[n] - pos_image[nn]) <
//                                    // 0.05f;
//                                    bool close_pos =
//                                        glm::length(pos_image[n] - pos_image[nn]) < 1.0f;
//                                    bool close_valid  = r.W > 0;
//                                    bool use_neighbor = close_nl && close_pos && close_valid;
//                                    // bool use_neighbor = close_valid;
//
//                                    if (use_neighbor)
//                                    {
//                                        float w = p_hat(scene, r.y, clamp_zero);
//                                        if (w > 0)
//                                        {
//                                            s.update(r.y, w * r.W * r.M);
//                                            M_ += r.M;
//                                        }
//                                    }
//                                }
//                            }
//                            s.M = M_;
//
//                            if (s.M > 0)
//                            {
//                                float w = p_hat(scene, s.y, clamp_zero);
//
//                                // work around zero p_hat
//                                // by ignoring invalid sample when shading
//                                // otherwise clamp p_hat
//                                if (w > 0)
//                                {
//                                    // if update for zero p_hat
//                                    s.W = 1.0f / w * (s.wsum / s.M);
//
//                                    // if not update for zero p_hat
//                                    // r.W = 1.0f / p_hat(scene, r.y) * (r.wsum / M);
//                                }
//                            }
//                        }
//                    }
//                }
//
//                std::swap(r_image, r_image_2);
//            }
//
//            for (int row = 0; row < image->height; row++)
//            {
//                printf("[c] %d / %d\n", row, image->height);
//
//#pragma omp parallel for
//                for (int col = 0; col < image->width; col++)
//                {
//                    glm::vec3 I(0, 0, 0);
//                    image->get_pixel(row, col, &I.x, &I.y, &I.z);
//
//                    auto& s = (*r_image)[col + row * image->width];
//
//                    I += shade_path(scene, s.y) * (s.W / nsamples);
//
//                    image->set_pixel(row, col, I.x, I.y, I.z);
//                }
//            }
//        }
//
//        delete r_image;
//        delete r_image_2;
//
//        dump_ppm(filename, *image);
//    }

    void render_gt(const Scene& scene, const Camera& camera, Image* image, const char* filename)
    {
        image->clear(0.0f, 0.0f, 0.0f);

        omp_set_num_threads(omp_get_max_threads() - 1);

        for (int row = 0; row < image->height; row++)
        {
            printf("%d / %d\n", row, image->height);

#pragma omp parallel for
            for (int col = 0; col < image->width; col++)
            {
                Ray cam_ray = cast_primary(row, col, camera, *image);

                float     hit_t;
                glm::vec3 hit_nl;
                Material  hit_mat;
                if (!scene.intersect(cam_ray, 0.01f, &hit_t, &hit_nl, &hit_mat))
                {
                    continue;
                }

                // do lighting
                glm::vec3 hit_p = cam_ray.orig + cam_ray.dir * hit_t;
                glm::vec3 I(0, 0, 0);

                int nsamples = 10000;
                for (int sample = 0; sample < nsamples; sample++)
                {
                    FastSampler rng(row, col, sample);

#if 1
                    Reservoir res;
                    for (int i = 0; i < scene.quad_lights.size(); i++) res.update(1.0f, 1.0f);
                    int   ii    = res.yi;
                    float pdf_L = 1.0f / scene.quad_lights.size();

                    auto& pl    = scene.quad_lights[ii];
                    float pdf_A = pl.pdf_area();
                    float pdf   = pdf_A * pdf_L;
#else
                    float pdf = pdf_A;
                    for (auto& pl : scene.quad_lights)
#endif
                    {
                        glm::vec3 light_p = pl.sample(rng.next(), rng.next());
                        // glm::vec3 light_p  = pl.sample(randf(), randf());
                        glm::vec3 light_nl = pl.normal();
                        float     pdf_A    = pl.pdf_area();

                        PathSample p;
                        p.hit_p    = hit_p;
                        p.hit_nl   = hit_nl;
                        p.hit_mat  = hit_mat;
                        p.light_p  = light_p;
                        p.light_nl = light_nl;
                        p.light_e  = pl.emission;

                        I += shade_path(scene, p) / pdf;
                    }
                }

                I /= nsamples;

                image->set_pixel(row, col, I.x, I.y, I.z);
            }
        }

        dump_ppm(filename, *image);
    }
};

int main()
{
    Image image(640, 480);

    Scene    scene;
    Camera   camera(glm::vec3(0, 20, 50), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    Renderer renderer;

    // renderer.render_gt(scene, camera, &image, "test_gt.ppm");
    // renderer.render_restir_alg3(scene, camera, &image, "test.ppm");
    // renderer.render_restir_alg4(scene, camera, &image, "test.ppm");
    renderer.render_restir_alg4_spatial_reuse(scene, camera, &image, "test.ppm");
    // renderer.render_restir_alg5(scene, camera, &image, "test.ppm");

    return 0;
    ;
}
}  // namespace restir

int main()
{
    // return reservoir_sampling::main();
    // return weighted_reservoir_sampling_one_sample::main();
    // return monte_carlo::main();

    return restir::main();
}