#pragma once

template <typename T>
class RenderTarget
{
public:
    T*   mBuf;
    T*   mOffset;
    int  mTotal;
    int  mOffsetN;
    int* mSampCount;

    RenderTarget() : mBuf(0), mSampCount(0)
    {
    }
    ~RenderTarget()
    {
        if (mBuf) delete[] mBuf;
        if (mSampCount) delete[] mSampCount;
    }
    void init(int _mTotal)
    {
        mTotal     = _mTotal;
        mBuf       = new T[mTotal];
        mSampCount = new int[mTotal];
        mOffset    = mBuf;
        mOffsetN   = 0;
        reset();
    }
    void rewind()  // return pointer to head
    {
        mOffset  = mBuf;
        mOffsetN = 0;
    }
    void reset()
    {
        rewind();
        memset(mBuf, 0, sizeof(T) * mTotal);
        memset(mSampCount, 0, sizeof(int) * mTotal);
        //         for(size_t n =0; n<mTotal; n++){
        //             mBuf[n] = 0;
        //             mSampCount[n] = 0;
        //         }
    }
    RenderTarget& operator<<(T _Val)
    {
        if (!(mOffsetN < mTotal)) rewind();
        //         if(mOffsetN < mTotal)
        //         {
        *(mOffset++) = _Val;
        mSampCount[mOffsetN]++;
        mOffsetN++;
        //         }
        return *this;
    }
    inline T fetch()
    {
        return *mOffset;
    }
    inline T fetch(int n)
    {
        return mBuf[n];
    }
    //     inline void set(size_t x, size_t y, _ColorChannnel _color)
    //     {
    //         mBuf[]
    //     }
    void acc(int spp, T _Val)
    {
        if (mOffsetN < mTotal)
        {
            int accSamps = mSampCount[mOffsetN];
            *(mOffset++) =
                (_Val * spp + mBuf[mOffsetN] * accSamps) / (accSamps + spp);
            mSampCount[mOffsetN] += spp;
            mOffsetN++;
        }
    }
    void acc(int spp, T _Val, int _OffsetN)
    {
        if (_OffsetN < mTotal)
        {
            int accSamps = mSampCount[_OffsetN];
            mBuf[_OffsetN] =
                (_Val * spp + mBuf[_OffsetN] * accSamps) / (accSamps + spp);
            mSampCount[_OffsetN] += spp;
        }
    }
};

void toSingle(RenderTarget<float>& out, RenderTarget<float>& in)
{
    int mTotal = out.mTotal;
    for (int n = 0; n < mTotal; n++)
    {
        out.mBuf[n] = (float)in.mBuf[n];
    }
}

void toDouble(RenderTarget<float>& out, RenderTarget<float>& in)
{
    int mTotal = out.mTotal;
    for (int n = 0; n < mTotal; n++)
    {
        out.mBuf[n] = (float)in.mBuf[n];
    }
}

template <typename T>
static inline T clamp(T x)
{
    return x < 0 ? 0 : x > 1 ? 1 : x;
}

void toSingle_with_correction(RenderTarget<float>& out,
                              RenderTarget<float>& in,
                              float                gamma)
{
    int mTotal = out.mTotal;
    for (int n = 0; n < mTotal; n++)
    {
        //         out.mBuf[n] = (float)pow(clamp(in.mBuf[n]),1.0/gamma);//REF
        out.mBuf[n] = (float)pow(f_max(in.mBuf[n], 0.0f), 1.0f / gamma);
    }
}

void toDouble_with_correction(RenderTarget<float>& out,
                              RenderTarget<float>& in,
                              float                gamma)
{
    int mTotal = out.mTotal;
    for (int n = 0; n < mTotal; n++)
    {
        out.mBuf[n] = (float)powf(clamp(in.mBuf[n]), 1.0f / gamma);
    }
}

void writeTexBuffer(const RenderTarget<float>& texBuf,
                    int                        mWidth,
                    int                        mHeight,
                    const char*                filename)
{
    // Save result to a PPM image (keep these flags if you compile under
    // Windows)       NOTE::especially std:ios::binary which is equivalent to
    // "wb" in fprintf()
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    ofs << "P6\n" << mWidth << " " << mHeight << "\n255\n";
    unsigned char* bufByte = new unsigned char[mWidth * mHeight * 3];
    for (int n = 0; n < mWidth * mHeight * 3; ++n)
    {
        bufByte[n] = (unsigned char)(f_min(1.0f, texBuf.mBuf[n]) * 255);
    }
    ofs.write(reinterpret_cast<char*>(bufByte), mWidth * mHeight * 3);
    ofs.close();
    delete[] bufByte;
}

static void tonemap(float* dst_fb, float* fb, int width, int height)
{
    int                      pixel_count = width * height;
    std::unique_ptr<float[]> lum = std::make_unique<float[]>(pixel_count);

    float lum_eps = 1e-7f;

    for (int n = 0; n < pixel_count; n++)
    {
        lum[n] = 0.299f * fb[n * 3] + 0.587f * fb[n * 3 + 1] +
                 0.114f * fb[n * 3 + 2];
        if (lum[n] < lum_eps)
        {
            lum[n] = lum_eps;
        }
    }

    float lum_min = FLT_MAX;
    float lum_max = -FLT_MAX;
    for (int n = 0; n < pixel_count; n++)
    {
        lum_min = lum_min < lum[n] ? lum_min : lum[n];
        lum_max = lum_max > lum[n] ? lum_max : lum[n];
    }

    float l_logmean = 0;
    float l_mean    = 0;
    float r_mean    = 0;
    float g_mean    = 0;
    float b_mean    = 0;
    for (int n = 0; n < pixel_count; n++)
    {
        l_logmean += logf(lum[n]);
        l_mean += lum[n];
        r_mean += fb[n * 3];
        g_mean += fb[n * 3 + 1];
        b_mean += fb[n * 3 + 2];
    }

    l_logmean /= pixel_count;
    l_mean /= pixel_count;
    r_mean /= pixel_count;
    g_mean /= pixel_count;
    b_mean /= pixel_count;

    float lmin = logf(lum_min);
    float lmax = logf(lum_max);
    float k    = (lmax - l_logmean) / (lmax - lmin);
    float m0   = 0.3f + 0.7f * powf(k, 1.4f);  // %contrast
    m0         = 0.77f;                        //%hdrsee default

    float m = m0;  // %Contrast [0.3f, 1.f]
    //     printf("contrast: %f\n", m);

    float c = 0.5f;  // %Chromatic Adaptation  [0.f, 1.f]
    float a = 0;     // %Light Adaptation  [0.f, 1.f]
    float f = 1.5;  //%Intensity  [-35.f, 10.f] (void*)func = intuitiveintensity
                    ////specify by log scale

    f = expf(-f);

    for (int n = 0; n < pixel_count; n++)
    {
        float r(fb[n * 3]), g(fb[n * 3 + 1]), b(fb[n * 3 + 2]);

        float r_lc = c * r + (1.0f - c) * lum[n];       //%local adaptation
        float r_gc = c * r_mean + (1.0f - c) * l_mean;  //%global adaptation
        float r_ca = a * r_lc + (1.0f - a) * r_gc;      // %pixel adaptation

        float g_lc = c * g + (1.0f - c) * lum[n];       // %local adaptation
        float g_gc = c * g_mean + (1.0f - c) * l_mean;  //  %global adaptation
        float g_ca = a * g_lc + (1.0f - a) * g_gc;      // %pixel adaptation

        float b_lc = c * b + (1.0f - c) * lum[n];       // %local adaptation
        float b_gc = c * b_mean + (1.0f - c) * l_mean;  // %global adaptation
        float b_ca = a * b_lc + (1.0f - a) * b_gc;      //  %pixel adaptation

        r = r / (r + powf(f * r_ca, m));
        g = g / (g + powf(f * g_ca, m));
        b = b / (b + powf(f * b_ca, m));

        dst_fb[n * 3]     = r;
        dst_fb[n * 3 + 1] = g;
        dst_fb[n * 3 + 2] = b;
    }
}

class RenderRecord
{
    float* fb;

    struct Header
    {
        int width, height;
    };

    Header header;

public:
    int spp;

    RenderRecord(int width, int height)
    {
        header.width  = width;
        header.height = height;
        spp           = 0;
        checkCudaErrors(
            cudaMalloc((void**)&fb, width * height * 3 * sizeof(float)));
        cudaMemset(fb, 0, sizeof(float) * 3 * header.width * header.height);
    }

    ~RenderRecord()
    {
        checkCudaErrors(cudaFree(fb));
    }

    void reset()
    {
        cudaMemset(fb, 0, sizeof(float) * 3 * header.width * header.height);
        spp = 0;
    }

    float* data()
    {
        return fb;
    }
};
