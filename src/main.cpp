#define _USE_MATH_DEFINES

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
using std::cout;
using std::endl;

std::random_device rdev;

namespace
{
std::default_random_engine         rng1(rdev());
std::uniform_int_distribution<int> dist1;

int randi(int mod)
{
    return dist1(rng1) % mod;
}
}  // namespace

namespace
{
std::default_random_engine            rng2(rdev());
std::uniform_real_distribution<float> dist2(0.0f, 1.0f);

float randf()
{
    return dist2(rng2);
}
}  // namespace

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
class Reservoir
{
public:
    float y;     // the output sample
    int   yi;    // output sample index
    float wsum;  // the sum of weights
    int   M;     // the number of samples seen so far
    float W;     // ?? for multi streaming ris

    Reservoir()
    {
        y    = 0;
        yi   = -1;
        wsum = 0;
        M    = 0;
        W    = 0;
    }

    //
    // xi : the incoming sample
    // wi : weight of the sample
    //
    void update(float xi, float wi)
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

void sample_record()
{
    ++g_sample_count;
}

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

float pdf_uniform(float x, float a, float b)
{
    return 1.0f / (b - a);
}

float pdf_linear(float x, float a, float b)
{
    return (x - a) * 2.0f / ((b - a) * (b - a));
}

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
    float                                                       a,
    float                                                       b,
    const std::function<std::pair<float, float>(float, float)>& sampler)
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
    float                                                       a,
    float                                                       b,
    const std::function<std::pair<float, float>(float, float)>& sampler)
{
    int   n_samples = 10; // number of samples for shading
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
    float                                                       a,
    float                                                       b,
    const std::function<std::pair<float, float>(float, float)>& sampler)
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

    static bool compare(const Result& a, const Result& b)
    {
        return a.rmse_score < b.rmse_score;
    }
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
            float f =
                integration_by_monte_carlo_ris(0.0f, 1.0f, sample_uniform);
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
            float f = integration_by_monte_carlo_streaming_ris(
                0.0f, 1.0f, sample_uniform);
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
            float f = integration_by_monte_carlo_streaming_ris(
                0.0f, 1.0f, sample_linear);
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
            float f = integration_by_monte_carlo_multi_streaming_ris(
                0.0f, 1.0f, sample_uniform);
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
            float f = integration_by_monte_carlo_multi_streaming_ris(
                0.0f, 1.0f, sample_linear);
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

int main()
{
    // return reservoir_sampling::main();
    // return weighted_reservoir_sampling_one_sample::main();
    return monte_carlo::main();
}