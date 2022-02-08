#pragma once

class StepCalculator
{
private:
    int spp = 1;

public:
    void reset(int spp_0 = 1)
    {
        spp = spp_0;
    }

    void update_spp(int total_spp, float last_frame_time)
    {
#if 0
        const int spp = 1;  // samples per pass per pixel
#elif 0
        spp = 1;
        if ((total_spp & (total_spp - 1) == 0) && total_spp <= 64)
            spp = spp * 2;
#elif 0
        if (total_spp >= 64)
            spp = 64;
        else if (total_spp >= 32)
            spp = 32;
        else if (total_spp >= 16)
            spp = 16;
        else if (total_spp >= 8)
            spp = 8;
        else if (total_spp >= 4)
            spp = 4;
        else if (total_spp >= 2)
            spp = 2;
#else
        if (last_frame_time < 0.2)
        {
            spp += spp;
        }
        else if (last_frame_time > 0.22)
        {
            if (spp > 1) spp -= 1;
        }
#endif
    }

    int get_spp() const
    {
        return spp;
    }
};
