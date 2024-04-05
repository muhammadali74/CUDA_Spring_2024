#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include <random>

#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "device_launch_parameters.h"
#include "vecutils.h" // from http://www.icmc.usp.br/~castelo/CUDA/common/inc/cutil_math.h
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline cudaError_t checkCudaErr(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
    }
    return err;
}

struct Ray
{
    float3 o, d;
    __device__ Ray(float3 o_, float3 d_) : o(o_), d(d_) {}
};
enum Refl_t
{
    DIFF,
    SPEC,
    REFR
}; // material types, used in radiance()

struct Sphere
{
    double rad;     // radius
    float3 p, e, c; // position, emission, color
    Refl_t refl;    // reflection type (DIFFuse, SPECular, REFRactive)
    // __device__ Sphere(double rad_, float3 p_, float3 e_, float3 c_, Refl_t refl_) : rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
    __device__ double intersect(const Ray &r) const
    {                        // returns distance, 0 if nohit
        float3 op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t, eps = 1e-4, b = dot(op, r.d), det = b * b - dot(op, op) + rad * rad;
        if (det < 0)
            return 0;
        else
            det = sqrt(det);
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    }
};

__constant__ Sphere spheres[] = {
    {1e5f, {1e5f + 1.0f, 40.8f, 81.6f}, {0.0f, 0.0f, 0.0f}, {0.75f, 0.25f, 0.25f}, DIFF},    // Left
    {1e5f, {-1e5f + 99.0f, 40.8f, 81.6f}, {0.0f, 0.0f, 0.0f}, {.25f, .25f, .75f}, DIFF},     // Rght
    {1e5f, {50.0f, 40.8f, 1e5f}, {0.0f, 0.0f, 0.0f}, {.75f, .75f, .75f}, DIFF},              // Back
    {1e5f, {50.0f, 40.8f, -1e5f + 600.0f}, {0.0f, 0.0f, 0.0f}, {1.00f, 1.00f, 1.00f}, DIFF}, // Frnt
    {1e5f, {50.0f, 1e5f, 81.6f}, {0.0f, 0.0f, 0.0f}, {.75f, .75f, .75f}, DIFF},              // Botm
    {1e5f, {50.0f, -1e5f + 81.6f, 81.6f}, {0.0f, 0.0f, 0.0f}, {.75f, .75f, .75f}, DIFF},     // Top
    {16.5f, {27.0f, 16.5f, 47.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, SPEC},            // small sphere 1
    {16.5f, {73.0f, 16.5f, 78.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, REFR},            // small sphere 2
    {600.0f, {50.0f, 681.6f - .27f, 81.6f}, {12.0f, 12.0f, 12.0f}, {0.0f, 0.0f, 0.0f}, DIFF} // Light
};

// __constant__ Sphere spheres[] = {
//     {1e5f, {50.0f, 1e5f - 4.0f, 81.6f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, DIFF},  // Botm
//     {12.0f, {48.0f, 32.0f, 24.0f}, {3.0f, 3.0f, 3.0f}, {0.0f, 0.0f, 0.0f}, DIFF},       // light
//     {12.0f, {24.0f, 8.0f, 40.0f}, {0.0f, 0.0f, 0.0f}, {0.408f, 0.741f, 0.467f}, DIFF},  // small sphere 2
//     {12.0f, {24.0f, 8.0f, -8.0f}, {0.0f, 0.0f, 0.0f}, {0.392f, 0.584f, 0.929f}, DIFF},  // 3
//     {12.0f, {20.0f, 52.0f, 40.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.498f, 0.314f}, DIFF},   // 5
//     {12.0f, {24.0f, 48.0f, -8.0f}, {0.0f, 0.0f, 0.0f}, {0.95f, 0.95f, 0.95f}, SPEC},    // 5
//     {12.0f, {72.0f, 8.0f, 40.0f}, {0.0f, 0.0f, 0.0f}, {0.95f, 0.95f, 0.95f}, SPEC},     // 3
//     {12.0f, {72.0f, 8.0f, -8.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.498f, 0.314f}, DIFF},    // 2
//     {12.0f, {76.0f, 52.0f, 40.0f}, {0.0f, 0.0f, 0.0f}, {0.392f, 0.584f, 0.929f}, DIFF}, // 1
//     {12.0f, {72.0f, 48.0f, -8.0f}, {0.0f, 0.0f, 0.0f}, {0.408f, 0.741f, 0.467f}, DIFF},
// };

// __constant__ Sphere spheres[] = {
//     {1e3f, {50.0f, 1000.0f, -500}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, DIFF}, // Botm
//     {600.0f, {50.0f, -600.0f, 12}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, DIFF}, // Botm
//     {26.0f, {0.0f, 30.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, DIFF},   // Botm
//     {26.0f, {100.0f, 30.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, DIFF},
//     {26.0f, {50.0f, 30.0f, 0.0f}, {12.0f, 12.0f, 12.0f}, {0.0f, 0.0f, 0.0f}, DIFF},
// }; // Botm

inline __host__ __device__ double
clamp(double x)
{
    return x < 0 ? 0 : x > 1 ? 1
                             : x;
}

inline __host__ __device__ int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

inline __device__ bool intersect(const Ray &r, double &t, int &id) // all args device frinedly
{
    double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
    for (int i = int(n); i--;)
        if ((d = spheres[i].intersect(r)) && d < t) // data dependency so cant paralleize. divergence ka boht khtra
        {
            t = d;
            id = i;
        }
    return t < inf;
}

__device__ float3 radiance(Ray &r, curandState state, int d = 0)
{ // returns ray color

    float3 pixelcolor = make_float3(0.0f, 0.0f, 0.0f);
    // float3 pixelcolor = make_float3();
    float3 mask = make_float3(1.0f, 1.0f, 1.0f);

    for (int depth = d; depth < 10; depth++)
    { // iteration up to 4 bounces (replaces recursion in CPU code)

        double t;   // distance to closest intersection
        int id = 0; // index of closest intersected sphere

        // test ray for intersection with scene
        if (!intersect(r, t, id))
        {
            // pixelcolor += make_float3(0.846f, 0.933f, 0.949f); // if miss, return black
            // return pixelcolor;
            return make_float3(0.0f, 0.0f, 0.0f);
        } // if miss, return black

        // else, we've got a hit!
        // compute hitpoint and normal
        const Sphere &obj = spheres[id];          // hitobject
        float3 x = r.o + r.d * t;                 // hitpoint
        float3 n = normalize(x - obj.p);          // normal
        float3 nl = dot(n, r.d) < 0 ? n : n * -1; // front facing normal
        float3 f = obj.c;
        double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y
                                                            : f.z; // max refl
        // if (depth > 5)
        // {
        //     if (curand_uniform(&state) < p)
        //     {
        //         f = f * (1 / p);
        //     }
        //     else
        //     {
        //         pixelcolor += obj.e * mask;
        //         break;
        //     }
        // }

        // create 2 random numbers
        float r1 = 2 * M_PI * curand_uniform(&state); // pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
        float r2 = curand_uniform(&state);            // pick random number for elevation
        float r2s = sqrtf(r2);

        // compute local orthonormal basis uvw at hitpoint to use for calculation random ray direction
        // first vector = normal at hitpoint, second vector is orthogonal to first, third vector is orthogonal to first two vectors
        if (obj.refl == DIFF)
        {
            float3 w = nl;
            float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
            float3 v = cross(w, u);
            float3 d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));
            pixelcolor += mask * obj.e;
            // new ray origin is intersection point of previous ray with scene
            r.o = x + nl * 0.05f;
            r.d = d;
            // mask *= dot(d, nl); // weigh light contribution using cosine of angle between incident light and normal
            // mask *= 2;          // fudge factor
        }

        else if (obj.refl == SPEC)
        {
            r.o = x + nl * 0.05f;
            r.d = r.d - n * 2 * dot(n, r.d);
            pixelcolor += mask * obj.e;
        }
        else if (obj.refl == REFR)
        {
            Ray reflRay(x + nl * 0.05f, r.d - n * 2 * dot(n, r.d)); // Ideal dielectric REFRACTION
            bool into = dot(n, nl) > 0;                             // Ray from outside going in?
            double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = dot(r.d, nl), cos2t;
            if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0) // Total internal reflection
            {
                pixelcolor += mask * obj.e;
                r.o = reflRay.o;
                r.d = reflRay.d;
            }
            else
            {
                float3 tdir = normalize((r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))));
                double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : dot(tdir, n));
                double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
                if (depth > 2)
                {
                    if (curand_uniform(&state) < P)
                    {
                        pixelcolor += mask * obj.e * RP;
                        r.o = reflRay.o;
                        r.d = reflRay.d;
                        // mask *= RP;
                    }
                    else
                    {
                        pixelcolor += mask * obj.e * TP;
                        r.o = x + nl * 0.05f;
                        r.d = tdir;
                        // mask *= TP;
                    }
                }
                else
                {
                    pixelcolor += mask * obj.e * Re;
                    pixelcolor += radiance(reflRay, state, depth + 1) * Re;
                    // r.o = reflRay.o;
                    // r.d = reflRay.d;
                    r.o = x;
                    r.d = tdir;
                    pixelcolor += mask * obj.e * Tr;

                    // mask *= Re;
                }
            }
        }

        // multiply with colour of object
        mask *= f;
        // mask *= 2; // fudge factor
    }

    return pixelcolor;
}

__global__ void raytracer(float3 *image, int w, int h, int samps)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int threadid = x + y * w;

    curandState state;
    curand_init(2045 + threadid, 0, 0, &state);
    int i;
    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // cam pos, dir
    float3 cx = make_float3(1024 * .5135 / 768, 0, 0);
    float3 cy = normalize(cross(cx, cam.d)) * .5135;

    if (x < w && y < h)
    {
        // printf("Ye hai aik thread   ");
        printf("\rRendering (%d spp) %5.2f%%", samps * 4, 100. * y / (h - 1));
        float3 temp = make_float3(0);
        for (int sy = 0; sy < 2; sy++)
        {
            i = (h - y - 1) * w + x;
            for (int sx = 0; sx < 2; sx++)
            {
                float3 r = make_float3(0);
                for (int s = 0; s < samps; s++)
                {
                    double r1 = 2 * curand_uniform(&state), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                    double r2 = 2 * curand_uniform(&state), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                    float3 d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) + cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
                    d = normalize(d);
                    Ray pp(cam.o + d * 140, d);
                    r = r + radiance(pp, state) * (1. / samps);
                } // Camera rays are pushed ^^^^^ forward to start in interior
                temp = temp + make_float3(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
            }
        }
        image[i] = image[i] + temp;
    }
}

// __constant__ *c = new float3[w * h];

int main(int argc, char *argv[])
{
    int w = 1024, h = 768, samps = argc == 2 ? atoi(argv[1]) / 4 : 1; // # samples
    float3 *c = new float3[w * h];                                    // To store the image
    float3 *d_c;
    checkCudaErr(cudaMalloc(&d_c, w * h * sizeof(float3)), "malloc");
    // checkCudaErr(cudaMemcpyToSymbol(spheres, spheres_h, sizeof(spheres_h)), "memcpy");
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks(w / threadsPerBlock.x, h / threadsPerBlock.y, 1);
    raytracer<<<numBlocks, threadsPerBlock>>>(d_c, w, h, samps);
    checkCudaErr(cudaGetLastError(), "raytracer kernel");
    cudaMemcpy(c, d_c, w * h * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaFree(d_c);

    FILE *f = fopen("image.ppm", "w"); // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w * h; i++)
        fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}