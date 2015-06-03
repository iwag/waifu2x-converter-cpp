/* -*- mode: c -*- */

float
get_data(__global const float *p, int hsz, int wsz, int step, int yi, int xi, int num_plane, int plane)
{
    xi = min(wsz-1, xi);
    xi = max(0, xi);

    return p[xi * num_plane];
}

__kernel void
filter(__global const float * __restrict__ packed_input,
       unsigned int nInputPlanes,
       __global float * __restrict__ packed_output,
       unsigned int nOutputPlanes,
       __global float * __restrict__ biases,
       unsigned int hsz,
       unsigned int wsz,
       __global float * __restrict__ weight,
       __local float * __restrict__ local_mem)
{
    unsigned int yi = get_group_id(0);
    unsigned int lid = get_local_id(0);

    __global const float * __restrict__ in = packed_input;
    size_t in_step = wsz * sizeof(float) * nInputPlanes;

    __global char *inp = (__global char*)packed_input;

    inp += in_step*yi;
    __global char *in0p = inp - in_step;
    if (yi == 0) {
        in0p = inp;
    }

    __global char *in1p = inp;
    __global char *in2p = inp + in_step;

    if (yi == hsz-1) {
        in2p = inp;
    }

    __local float *intermediate = local_mem;
    local_mem += sizeof(float) * nOutputPlanes;

    unsigned int vec_width = min((int)VEC_WIDTH, (int)nOutputPlanes);
    unsigned int nOutputBlock = nOutputPlanes / vec_width;
    int inputBlockSize = 4;
    unsigned int nInputBlock = (nInputPlanes+3U)/4U;

    /* local_size = 16KB - arguments = 14KB?
     *
     * weight size = output block size(vec_width) * input block_size * 9 * sizeof(float)
     *  vec_width = 64,
     *  input block_size = 4
     *  64*4*9*4 = 9216 byte
     *
     * intermediate size = output block size * x block size * sizeof(float)
     *
     */

    for (int ibi=0; ibi<nInputBlock; ibi++) {
        for (int obi=0; obi<nOutputBlock; obi++) {
            __global float *out = packed_output + (yi*wsz)*nOutputPlanes;

            for (int xi=0; xi<wsz; xi+=2) {
                float intermediate0 = 0;
                float intermediate1 = 0;

                __global float *in01 = (__global float*)in0p;
                __global float *in11 = (__global float*)in1p;
                __global float *in21 = (__global float*)in2p;

                in01 += xi * nInputPlanes;
                in11 += xi * nInputPlanes;
                in21 += xi * nInputPlanes;

                __global float *w = weight + lid + obi*vec_width*9;

                unsigned int ipBegin = ibi * inputBlockSize;
                unsigned int ipEnd = min(ipBegin + inputBlockSize, nInputPlanes);

                for (unsigned int ipIndex = ipBegin;
                     ipIndex < ipEnd; ipIndex++)
                {
                    float i00, i01, i02, i03;
                    float i10, i11, i12, i13;
                    float i20, i21, i22, i23;

                    i01 = in01[0];
                    i11 = in11[0];
                    i21 = in21[0];
                    i02 = in01[+nInputPlanes];
                    i12 = in11[+nInputPlanes];
                    i22 = in21[+nInputPlanes];

                    if (xi == 0) {
                        i00 = i01;
                        i10 = i11;
                        i20 = i21;
                    } else {
                        i00 = in01[-nInputPlanes];
                        i10 = in11[-nInputPlanes];
                        i20 = in21[-nInputPlanes];
                    }

                    if (xi+1 == wsz-1) {
                        i03 = i02;
                        i13 = i12;
                        i23 = i22;
                    } else {
                        i03 = in01[+nInputPlanes*2];
                        i13 = in11[+nInputPlanes*2];
                        i23 = in21[+nInputPlanes*2];
                    }

                    in01 ++;
                    in11 ++;
                    in21 ++;

                    float v0 = 0, v1 = 0;

                    v0 += w[0*vec_width] * i00;
                    v1 += w[0*vec_width] * i01;

                    v0 += w[1*vec_width] * i01;
                    v1 += w[1*vec_width] * i02;

                    v0 += w[2*vec_width] * i02;
                    v1 += w[2*vec_width] * i03;


                    v0 += w[3*vec_width] * i10;
                    v1 += w[3*vec_width] * i11;

                    v0 += w[4*vec_width] * i11;
                    v1 += w[4*vec_width] * i12;

                    v0 += w[5*vec_width] * i12;
                    v1 += w[5*vec_width] * i13;


                    v0 += w[6*vec_width] * i20;
                    v1 += w[6*vec_width] * i21;

                    v0 += w[7*vec_width] * i21;
                    v1 += w[7*vec_width] * i22;

                    v0 += w[8*vec_width] * i22;
                    v1 += w[8*vec_width] * i23;

                    w += nOutputPlanes*9;

                    intermediate0 += v0;
                    intermediate1 += v1;
                }

                int opIndex = obi*vec_width + lid;

                if (ibi == nInputBlock-1) {
                    float bv = biases[opIndex];

                    float v, mtz, ltz;

                    v = intermediate0 + out[opIndex];
                    v += bv;
                    mtz = max(v, 0.0f);
                    ltz = min(v, 0.0f);
                    v = ltz * 0.1f + mtz;

                    out[opIndex] = v;
                    out += nOutputPlanes;

                    v = intermediate1 + out[opIndex];
                    v += bv;
                    mtz = max(v, 0.0f);
                    ltz = min(v, 0.0f);
                    v = ltz * 0.1f + mtz;

                    out[opIndex] = v;
                    out += nOutputPlanes;
                } else if (ibi == 0) {
                    out[opIndex] = intermediate0;
                    out += nOutputPlanes;
                    out[opIndex] = intermediate1;
                    out += nOutputPlanes;
                } else {
                    out[opIndex] += intermediate0;
                    out += nOutputPlanes;
                    out[opIndex] += intermediate1;
                    out += nOutputPlanes;
                }
            }
        }
    }
}

