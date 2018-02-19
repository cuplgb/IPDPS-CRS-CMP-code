////////////////////////////////////////////////////////////////////////////////
/**
 * @file main.cpp
 * @date 2017-03-04
 * @author Tiago Lobato Gimenes    (tlgimenes@gmail.com)
 *
 * @copyright
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
////////////////////////////////////////////////////////////////////////////////

#include "log.hpp"
#include "utils.hpp"
#include "parser.hpp"
#include "su_gather.hpp"
#include "cuda.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>

#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <string>

#include <chrono>

////////////////////////////////////////////////////////////////////////////////

#define MAX_W 16

#define EPSILON 1e-13

#define FACTOR 1e6

#define NTHREADS 128

////////////////////////////////////////////////////////////////////////////////

std::chrono::high_resolution_clock::time_point beg, end;

double kernel_execution_time = 0.0;

////////////////////////////////////////////////////////////////////////////////

__global__ void
init_c(real *c, real inc, real c0) {
  int i = blockIdx.x;

  c[i] = c0 + inc*i;
}

////////////////////////////////////////////////////////////////////////////////

__global__ void
init_half(real* scalco, real* gx, real* gy, real* sx, real* sy, real* h) {
  int i = blockIdx.x;

  real _s = scalco[i];

  if(-EPSILON < _s && _s < EPSILON) _s = 1.0f;
  else if(_s < 0) _s = 1.0f / _s;

  real hx = (gx[i] - sx[i]) * _s;
  real hy = (gy[i] - sy[i]) * _s;

  h[i] = 0.25 * (hx * hx + hy * hy) / FACTOR;
}

////////////////////////////////////////////////////////////////////////////////

__global__ void
compute_semblances(real *h, real* c, real* samples, real* num, real* stt,
    int t_id0, int t_idf, real _idt, real _dt, int _tau, int _w, int nc, int ns) {
  real _den = 0.0f, _ac_linear = 0.0f, _ac_squared = 0.0f;
  real _num[MAX_W],  m = 0.0f;
  int err = 0;

  int i = blockIdx.x * NTHREADS + threadIdx.x;

  int t0 = i / nc;
  int c_id = i % nc;

  if(i < ns * nc)
  {
    real _c = c[c_id];
    real _t0 = _dt * t0;
    _t0 *= _t0;

    // start _num with zeros
    for(int j=0; j < _w; j++) _num[j] = 0.0f;

    for(int t_id=t_id0; t_id < t_idf; t_id++) {
      // Evaluate t
      real t = sqrt(_t0 + _c * h[t_id]) * _idt;

      int it = (int)( t );
      int ittau = it - _tau;
      real x = t - (real)it;

      if(ittau >= 0 && it + _tau + 1 < ns) {
        int k1 = ittau + (t_id-t_id0)*ns;
        real sk1p1 = samples[k1], sk1;

        for(int j=0; j < _w; j++) {
          k1++;
          sk1 = sk1p1;
          sk1p1 = samples[k1];
          // linear interpolation optmized for this problema
          real v = (sk1p1 - sk1) * x + sk1;

          _num[j] += v;
          _den += v * v;
          _ac_linear += v;
        }
        m += 1;
      } else { err++; }
    }

    // Reduction for num
    for(int j=0; j < _w; j++) _ac_squared += _num[j] * _num[j];

    // Evaluate semblances
    if(_den > EPSILON && m > EPSILON && _w > EPSILON && err < 2) {
      num[i] = _ac_squared / (_den * m);
      stt[i] = _ac_linear  / (_w   * m);
    }
    else {
      num[i] = -1.0f;
      stt[i] = -1.0f;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

__global__ void
redux_semblances(const real *num, const real* stt, int* ctr, real* str, real* stk, const int nc, const int cdp_id, const int ns) {
  int t0 = blockIdx.x * NTHREADS + threadIdx.x;

  if(t0 < ns)
  {
    real max_sem = 0.0f;
    int max_c = -1;

    for(int it=t0*nc; it < (t0+1)*nc ; it++) {
      real _num = num[it];
      if(_num > max_sem) {
        max_sem = _num;
        max_c = it;
      }
    }

    ctr[cdp_id*ns + t0] = max_c % nc;
    str[cdp_id*ns + t0] = max_sem;
    stk[cdp_id*ns + t0] = max_c > -1 ? stt[max_c] : 0;
  }
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv) {
  std::ofstream c_out("cmp.c.su", std::ofstream::out | std::ios::binary);
  std::ofstream s_out("cmp.coher.su", std::ofstream::out | std::ios::binary);
  std::ofstream stack("cmp.stack.su", std::ofstream::out | std::ios::binary);

  // Parse command line and read arguments
  parser::add_argument("-c0", "C0 constant");
  parser::add_argument("-c1", "C1 constant");
  parser::add_argument("-nc", "NC constant");
  parser::add_argument("-aph", "APH constant");
  parser::add_argument("-tau", "Tau constant");
  parser::add_argument("-i", "Data path");
  parser::add_argument("-v", "Verbosity Level 0-3");

  parser::parse(argc, argv);

  // Read parameters and input
  const real c0 = std::stof(parser::get("-c0", true)) * FACTOR;
  const real c1 = std::stof(parser::get("-c1", true)) * FACTOR;
  const real itau = std::stof(parser::get("-tau", true));
  const int nc = std::stoi(parser::get("-nc", true));
  const int aph = std::stoi(parser::get("-aph", true));
  std::string path = parser::get("-i", true);
  logger::verbosity_level(std::stoi(parser::get("-v", false)));

  // Reads *.su data and starts gather
  su_gather gather(path, aph, nc);

  real *h_gx, *h_gy, *h_sx, *h_sy, *h_scalco, *h_samples, *h_str, *h_stk, dt;
  int *ntraces_by_cdp_id, *h_ctr;

  // Linearize gather data in order to improove data coalescence in GPU
  gather.linearize(ntraces_by_cdp_id, h_samples, dt, h_gx, h_gy, h_sx, h_sy, h_scalco, nc);
  const int  ttraces = gather.ttraces(); // Total traces -> Total amount of traces read
  const int  ncdps = gather().size();    // Number of cdps -> Total number of cdps read
  const int  ns = gather.ns();           // Number of samples
  const int  ntrs = gather.ntrs();       // Max number of traces by cdp
  const real inc = (c1-c0) * (1.0f / (real)nc);

  real *h, *gx, *gy, *sx, *sy, *scalco, *cdpsmpl;
  real *c, *num, *stt, *str, *stk; // nc stts per sample
  int  *ctr; // ns Cs per cdp

  dt = dt / 1000000.0f;
  real idt = 1.0f / dt;
  int tau = ((int)( itau * idt) > 0) ? ((int)( itau * idt)) : 0;
  int w = (2 * tau) + 1;

  int number_of_semblances = 0;

  LOG(DEBUG, "Starting CUDA devices");

  cuda runtime;

  // Alloc memory
  cudaSafe(cudaMalloc((void**)&gx, sizeof(real)*ttraces));
  cudaSafe(cudaMalloc((void**)&gy, sizeof(real)*ttraces));
  cudaSafe(cudaMalloc((void**)&sx, sizeof(real)*ttraces));
  cudaSafe(cudaMalloc((void**)&sy, sizeof(real)*ttraces));
  cudaSafe(cudaMalloc((void**)&scalco, sizeof(real)*ttraces));
  cudaSafe(cudaMalloc((void**)&cdpsmpl, sizeof(real)*ntrs*ns));

  cudaSafe(cudaMemcpy(gx    , h_gx    , sizeof(real)*ttraces, cudaMemcpyHostToDevice));
  cudaSafe(cudaMemcpy(gy    , h_gy    , sizeof(real)*ttraces, cudaMemcpyHostToDevice));
  cudaSafe(cudaMemcpy(sx    , h_sx    , sizeof(real)*ttraces, cudaMemcpyHostToDevice));
  cudaSafe(cudaMemcpy(sy    , h_sy    , sizeof(real)*ttraces, cudaMemcpyHostToDevice));
  cudaSafe(cudaMemcpy(scalco, h_scalco, sizeof(real)*ttraces, cudaMemcpyHostToDevice));

  cudaSafe(cudaMalloc((void** ) &c  , sizeof(real)*nc      ));
  cudaSafe(cudaMalloc((void** ) &h  , sizeof(real)*ttraces ));
  cudaSafe(cudaMalloc((void** ) &num, sizeof(real)*ns*nc   ));
  cudaSafe(cudaMalloc((void** ) &stt, sizeof(real)*ns*nc   ));
  cudaSafe(cudaMalloc((void** ) &ctr, sizeof(int )*ncdps*ns));
  cudaSafe(cudaMalloc((void** ) &str, sizeof(real)*ncdps*ns));
  cudaSafe(cudaMalloc((void** ) &stk, sizeof(real)*ncdps*ns));

  cudaSafe(cudaMallocHost((void**)&h_ctr, sizeof(int )*ncdps*ns));
  cudaSafe(cudaMallocHost((void**)&h_str, sizeof(real)*ncdps*ns));
  cudaSafe(cudaMallocHost((void**)&h_stk, sizeof(real)*ncdps*ns));

  std::vector<std::pair<cudaEvent_t, cudaEvent_t>> events(ncdps*2);
  for(auto& e: events) {
    cudaEventCreate(&e.first);
    cudaEventCreate(&e.second);
  }

  // Chronometer
  beg = std::chrono::high_resolution_clock::now();

  //
  // CUDA REGION
  //

  // Evaluate Cs - linspace
  init_c<<<nc, 1>>>(c, inc, c0);

  // Evaluate halfoffset points in x and y coordinates
  init_half<<<ttraces, 1>>>(scalco, gx, gy, sx, sy, h);

  for(int cdp_id = 0; cdp_id < ncdps; cdp_id++) {
    int t_id0 = cdp_id > 0 ? ntraces_by_cdp_id[cdp_id-1] : 0;
    int t_idf = ntraces_by_cdp_id[cdp_id];
    int stride = t_idf - t_id0;

    cudaSafe(cudaMemcpyAsync(cdpsmpl, h_samples + t_id0*ns , sizeof(real)*stride*ns , cudaMemcpyHostToDevice));

    // Compute semblances for each c for each sample
    cudaSafe(cudaEventRecord(events[2*cdp_id].first));
    compute_semblances<<<(ns*nc+NTHREADS-1)/NTHREADS, NTHREADS>>>(h, c, cdpsmpl, num, stt, t_id0, t_idf, idt, dt, tau, w, nc, ns);
    cudaSafe(cudaEventRecord(events[2*cdp_id].second));

    // Get max C for max semblance for each sample on this cdp
    cudaSafe(cudaEventRecord(events[2*cdp_id+1].first));
    redux_semblances<<<(ns+NTHREADS-1)/NTHREADS, NTHREADS>>>(num, stt, ctr, str, stk, nc, cdp_id, ns);
    cudaSafe(cudaEventRecord(events[2*cdp_id+1].second));

    number_of_semblances += stride;

    LOG(DEBUG, "CUDA Progress: " + std::to_string(cdp_id) + "/" + std::to_string(ncdps));
  }
  // Gets time at end of computation
  cudaSafe(cudaDeviceSynchronize());
  end = std::chrono::high_resolution_clock::now();

  // Copy results back to host
  cudaSafe(cudaMemcpy(h_ctr, ctr, sizeof(int ) * ncdps * ns, cudaMemcpyDeviceToHost));
  cudaSafe(cudaMemcpy(h_str, str, sizeof(real) * ncdps * ns, cudaMemcpyDeviceToHost));
  cudaSafe(cudaMemcpy(h_stk, stk, sizeof(real) * ncdps * ns, cudaMemcpyDeviceToHost));

  for(auto& e: events) {
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, e.first, e.second);

    kernel_execution_time += milliseconds/1e3;
  }

  //
  // END CUDA REGION
  //

  // Logs stats (exec time and semblance-traces per second)
  double total_exec_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - beg).count();
  double stps = (number_of_semblances / 1e9 ) * (ns * nc / total_exec_time);
  double kernel_stps = (number_of_semblances / 1e9 ) * (ns * nc / kernel_execution_time);
  std::string stats = "Total Execution Time: " + std::to_string(total_exec_time);
  stats += ": Giga-Semblances-Trace/s: " + std::to_string(stps);
  stats += ": Kernel Execution Time: " + std::to_string(kernel_execution_time);
  stats += ": Kernel Giga-Semblances-Trace/s: " + std::to_string(kernel_stps);
  LOG(INFO, stats);

  // Delinearizes data and save it into a *.su file
  for(int i=0; i < ncdps; i++) {
    su_trace ctr_t = gather[i].traces()[0];
    su_trace str_t = gather[i].traces()[0];
    su_trace stk_t = gather[i].traces()[0];

    ctr_t.offset() = 0;
    ctr_t.sx() = ctr_t.gx() = (gather[i].traces()[0].sx() + gather[i].traces()[0].gx()) >> 1;
    ctr_t.sy() = ctr_t.gy() = (gather[i].traces()[0].sy() + gather[i].traces()[0].gy()) >> 1;

    for(int k=0; k < ns; k++) ctr_t.data()[k] = h_ctr[i*ns+k] < 0 ? 0.0f: (c0 + inc * h_ctr[i*ns+k]) / FACTOR;
    str_t.data().assign(h_str + i*ns, h_str + (i+1)*ns);
    stk_t.data().assign(h_stk + i*ns, h_stk + (i+1)*ns);

    ctr_t.fputtr(c_out);
    str_t.fputtr(s_out);
    stk_t.fputtr(stack);
  }

  cudaSafe(cudaFree(gx     ));
  cudaSafe(cudaFree(gy     ));
  cudaSafe(cudaFree(sx     ));
  cudaSafe(cudaFree(sy     ));
  cudaSafe(cudaFree(scalco ));
  cudaSafe(cudaFree(cdpsmpl));
  cudaSafe(cudaFree(h      ));
  cudaSafe(cudaFree(c      ));
  cudaSafe(cudaFree(num    ));
  cudaSafe(cudaFree(stt    ));
  cudaSafe(cudaFree(ctr    ));
  cudaSafe(cudaFree(str    ));
  cudaSafe(cudaFree(stk    ));

  cudaSafe(cudaFreeHost(h_ctr));
  cudaSafe(cudaFreeHost(h_str));
  cudaSafe(cudaFreeHost(h_stk));

  delete [] h_gx              ;
  delete [] h_gy              ;
  delete [] h_sx              ;
  delete [] h_sy              ;
  delete [] h_scalco          ;
  delete [] h_samples         ;
  delete [] ntraces_by_cdp_id ;

  return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
