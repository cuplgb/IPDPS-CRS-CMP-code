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

#define MAX_W 5

#define EPSILON 1e-13

#define FACTOR 1e6

#define NTHREADS 128

////////////////////////////////////////////////////////////////////////////////

std::chrono::high_resolution_clock::time_point beg, end;

double kernel_total_exec_time = 0.0;

////////////////////////////////////////////////////////////////////////////////

  __global__ void
init_par(real4 *par,
    real a0,
    real b0,
    real c0,
    real inc_a,
    real inc_b,
    real inc_c,
    int na,
    int nb,
    int nc)
{
  int i = blockIdx.x;

  int ida = i/(nc*nb);
  int idb = (i/nc)%nb;
  int idc = i%nc;

  par[i].a = (a0+ida*inc_a);
  par[i].b = (b0+idb*inc_b);
  par[i].c = (c0+idc*inc_c);
}

////////////////////////////////////////////////////////////////////////////////

  __global__ void
init_mid(real *scalco,
    real *gx,
    real *gy,
    real *sx,
    real *sy,
    real *m0x,
    real *m0y,
    real *h0)
{
  int i = blockIdx.x;

  real _s = scalco[i];

  if(-EPSILON < _s && _s < EPSILON) _s = 1.0f;
  else if(_s < 0) _s = 1.0f / _s;

  m0x[i] = (gx[i] + sx[i]) * _s * 0.5;
  m0y[i] = (gy[i] + sy[i]) * _s * 0.5;

  real hx = (gx[i] - sx[i]) * _s;
  real hy = (gy[i] - sy[i]) * _s;

  h0[i] = 0.25 * (hx * hx + hy * hy) / FACTOR;
}

////////////////////////////////////////////////////////////////////////////////

  __global__ void
compute_points_for_gather(
    real* m0x,
    real* m0y,
    real* h0,
    real* h,
    real* m2,
    real* m,
    int * ntraces_by_cdp_id,
    real m0x_cdp_id,
    real m0y_cdp_id,
    int cdp0,
    int cdpf
    )
{
  real dx, dy, _m2;
  int cdp = cdp0 + blockIdx.x;

  if(cdp0 <= cdp && cdp <= cdpf)
  {
    int t_id00 = cdp0 > 0 ? ntraces_by_cdp_id[cdp0-1] : 0;
    int t_id0 = cdp > 0 ? ntraces_by_cdp_id[cdp-1] : 0;
    int t_idf = ntraces_by_cdp_id[cdp];
    int sz = t_id0-t_id00;

    for(int it=0; it < t_idf-t_id0; it++)
    {
      dx = m0x[t_id0 + it] - m0x_cdp_id;
      dy = m0y[t_id0 + it] - m0y_cdp_id;
      _m2 = dx*dx + dy*dy;

      m2[sz + it] = _m2;
      m [sz + it] = sqrt(_m2);
      h [sz + it] = h0[t_id0 + it];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

  __global__ void
compute_semblances(
    real * h,
    real * m2,
    real * m,
    real4* par,
    real * samples,
    real * num,
    real * stt,
    int size,
    real idt,
    real dt,
    int tau,
    int w,
    int npar,
    int ns )
{
  int i = blockIdx.x * NTHREADS + threadIdx.x;
  int t0 = i / npar;
  int par_id = i % npar;

  if(i < ns*npar)
  {
    real _den = 0.0f, _ac_linear = 0.0f, _ac_squared = 0.0f;
    real _num[MAX_W],  mm = 0.0f;
    int err = 0;

    real4 _p = par[par_id];
    real _t0 = dt * t0;

    // start _num with zeros
    for(int j=0; j < w; j++) _num[j] = 0.0f;

    for(int k=0; k < size; k++) {
      // Evaluate t
      real _m2 = m2[k];
      real t = _t0 + _p.a * m[k];
      t = t*t + _p.b*_m2 + _p.c*h[k];
      t = t < 0.0 ? -1 : (sqrt(t) * idt);

      int it = (int)( t );
      int ittau = it - tau;
      real x = t - (real)it;

      if(ittau >= 0 && it + tau + 1 < ns) {
        int k1 = ittau + k*ns;
        real sk1p1=samples[k1], sk1;

        for(int j=0; j < w; j++) {
          k1++;
          sk1 = sk1p1;
          sk1p1 = samples[k1];
          // linear interpolation optmized for this problem
          real v = (sk1p1 - sk1) * x + sk1;

          _num[j] += v;
          _den += v * v;
          _ac_linear += v;
        }
        mm += 1;
      } else { err++; }
    }

    // Reduction for num
    for(int j=0; j < w; j++) _ac_squared += _num[j] * _num[j];

    // Evaluate semblances
    if(_den > EPSILON && mm > EPSILON && w > EPSILON && err < 2) {
      num[i] = _ac_squared / (_den * mm);
      stt[i] = _ac_linear  / (w   * mm);
    }
    else {
      num[i] = 0.0f;
      stt[i] = 0.0f;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

  __global__ void
redux_semblances(
    real *num,
    real *stt,
    int  *ctr,
    real *str,
    real *stk,
    int npar,
    int cdp_id,
    int ns)
{
  int t0 = blockIdx.x * NTHREADS + threadIdx.x;

  if(t0 < ns)
  {
    real max_sem = 0.0f, _num;
    int max_par = 0;

    for(int it=t0*npar; it < (t0+1)*npar; it++) {
      _num = num[it];
      if(_num > max_sem) {
        max_sem = _num;
        max_par = it;
      }
    }

    ctr[cdp_id*ns + t0] = max_par % npar;
    str[cdp_id*ns + t0] = max_sem;
    stk[cdp_id*ns + t0] = stt[max_par];
  }
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv) {
  std::ofstream a_out("crs.a.su", std::ofstream::out | std::ios::binary);
  std::ofstream b_out("crs.b.su", std::ofstream::out | std::ios::binary);
  std::ofstream c_out("crs.c.su", std::ofstream::out | std::ios::binary);
  std::ofstream s_out("crs.coher.su", std::ofstream::out | std::ios::binary);
  std::ofstream stack("crs.stack.su", std::ofstream::out | std::ios::binary);

  // Parse commanb line anb read arguments
  parser::add_argument("-a0", "A0 constant");
  parser::add_argument("-a1", "A1 constant");
  parser::add_argument("-na", "NA constant");
  parser::add_argument("-b0", "B0 constant");
  parser::add_argument("-b1", "B1 constant");
  parser::add_argument("-nb", "NB constant");
  parser::add_argument("-c0", "C0 constant");
  parser::add_argument("-c1", "C1 constant");
  parser::add_argument("-nc", "NC constant");
  parser::add_argument("-aph", "APH constant");
  parser::add_argument("-apm", "APM constant");
  parser::add_argument("-tau", "Tau constant");
  parser::add_argument("-i", "Data path");
  parser::add_argument("-v", "Verbosity Level 0-3");

  parser::parse(argc, argv);

  // Read parameters anb input
  const real a0 = std::stod(parser::get("-a0", true));
  const real a1 = std::stod(parser::get("-a1", true));
  const real b0 = std::stod(parser::get("-b0", true));
  const real b1 = std::stod(parser::get("-b1", true));
  const real c0 = std::stod(parser::get("-c0", true)) * FACTOR;
  const real c1 = std::stod(parser::get("-c1", true)) * FACTOR;
  const real itau = std::stod(parser::get("-tau", true));
  const int na = std::stoi(parser::get("-na", true));
  const int nb = std::stoi(parser::get("-nb", true));
  const int nc = std::stoi(parser::get("-nc", true));
  const int aph = std::stoi(parser::get("-aph", true));
  const int apm = std::stoi(parser::get("-apm", true));
  const int ng = 1;
  std::string path = parser::get("-i", true);
  logger::verbosity_level(std::stoi(parser::get("-v", false)));

  // Reads *.su data anb starts gather
  su_gather gather(path, aph, apm, nc);

  real *h_gx, *h_gy, *h_sx, *h_sy, *h_scalco, h_dt, *h_samples;
  int *h_ntraces_by_cdp_id, *ctr, *ntraces_by_cdp_id;
  real *gx, *gy, *sx, *sy, *scalco, *h0, *m0x, *m0y, *str, *stk;
  std::vector<real*> hs(ng), m2s(ng), ms(ng), cdpsmpls(ng), nums(ng), stts(ng);
  std::vector<int> size(ng);
  real4 *par;

  // Linearize gather data in order to improove data coalescence in GPU
  gather.linearize(h_ntraces_by_cdp_id, h_samples, h_dt, h_gx, h_gy, h_sx, h_sy, h_scalco, nc);
  const int ttraces = gather.ttraces(); // Total traces -> Total amount of traces read
  const int ncdps = gather().size();    // Number of cdps -> Total number of cdps read
  const int ns = gather.ns();           // Number of samples
  const int ntrs = gather.ntrs();       // Max number of traces by cdp
  const real inc_a = (a1-a0) * (1.0 / (real)na);
  const real inc_b = (b1-b0) * (1.0 / (real)nb);
  const real inc_c = (c1-c0) * (1.0 / (real)nc);
  const int npar = na * nb * nc;
  const int max_gather = gather.max_gather();
  int number_of_semblances = 0;

  int *h_ctr;
  real* h_str, *h_stk, *h_m0x, *h_m0y;

  real dt = h_dt / 1000000.0f;
  real idt = 1.0f / dt;
  int tau = (int)( itau * idt) > 0 ? (int)( itau * idt)  : 0;
  int w = (2 * tau) + 1;

  LOG(DEBUG, "Starting CUDA devices");

  cuda runtime;
  std::vector<cudaStream_t> streams(ng);

  // Start streams. 0 .. ng
  for(auto& stream: streams)
    cudaSafe(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Alloc memory
  cudaSafe(cudaMalloc((void**)&gx, sizeof(real)*ttraces));
  cudaSafe(cudaMalloc((void**)&gy, sizeof(real)*ttraces));
  cudaSafe(cudaMalloc((void**)&sx, sizeof(real)*ttraces));
  cudaSafe(cudaMalloc((void**)&sy, sizeof(real)*ttraces));
  cudaSafe(cudaMalloc((void**)&scalco, sizeof(real)*ttraces));

  cudaSafe(cudaMalloc((void** )&ntraces_by_cdp_id, sizeof(int)*ncdps));

  cudaSafe(cudaMemcpy(gx    , h_gx    , sizeof(real)*ttraces, cudaMemcpyHostToDevice));
  cudaSafe(cudaMemcpy(gy    , h_gy    , sizeof(real)*ttraces, cudaMemcpyHostToDevice));
  cudaSafe(cudaMemcpy(sx    , h_sx    , sizeof(real)*ttraces, cudaMemcpyHostToDevice));
  cudaSafe(cudaMemcpy(sy    , h_sy    , sizeof(real)*ttraces, cudaMemcpyHostToDevice));
  cudaSafe(cudaMemcpy(scalco, h_scalco, sizeof(real)*ttraces, cudaMemcpyHostToDevice));

  cudaSafe(cudaMemcpy(ntraces_by_cdp_id, h_ntraces_by_cdp_id, sizeof(int)*ncdps, cudaMemcpyHostToDevice));

  cudaSafe(cudaMalloc((void** ) &par, sizeof(real4)*npar));
  cudaSafe(cudaMalloc((void** ) &h0,  sizeof(real)*ttraces));
  cudaSafe(cudaMalloc((void** ) &m0x, sizeof(real)*ttraces));
  cudaSafe(cudaMalloc((void** ) &m0y, sizeof(real)*ttraces));
  cudaSafe(cudaMalloc((void** ) &ctr, sizeof(int )*ncdps*ns));
  cudaSafe(cudaMalloc((void** ) &str, sizeof(real)*ncdps*ns));
  cudaSafe(cudaMalloc((void** ) &stk, sizeof(real)*ncdps*ns));
  for(int i=0; i < ng; i++) {
    cudaSafe(cudaMalloc((void** ) &hs      [i], sizeof(real)*ntrs*max_gather));
    cudaSafe(cudaMalloc((void** ) &m2s     [i], sizeof(real)*ntrs*max_gather));
    cudaSafe(cudaMalloc((void** ) &ms      [i], sizeof(real)*ntrs*max_gather));
    cudaSafe(cudaMalloc((void** ) &nums    [i], sizeof(real)*ns*npar));
    cudaSafe(cudaMalloc((void** ) &stts    [i], sizeof(real)*ns*npar));
    cudaSafe(cudaMalloc((void** ) &cdpsmpls[i], sizeof(real)*ntrs*max_gather*ns));
  }

  cudaSafe(cudaMallocHost((void**)&h_ctr, sizeof(int )*ncdps*ns));
  cudaSafe(cudaMallocHost((void**)&h_str, sizeof(real)*ncdps*ns));
  cudaSafe(cudaMallocHost((void**)&h_stk, sizeof(real)*ncdps*ns));

  cudaSafe(cudaMallocHost((void**)&h_ctr, sizeof(int)*ncdps*ns));
  cudaSafe(cudaMallocHost((void**)&h_str, sizeof(real)*ncdps*ns));
  cudaSafe(cudaMallocHost((void**)&h_stk, sizeof(real)*ncdps*ns));
  cudaSafe(cudaMallocHost((void**)&h_m0x, sizeof(real)*ttraces));
  cudaSafe(cudaMallocHost((void**)&h_m0y, sizeof(real)*ttraces));

  std::vector<std::pair<cudaEvent_t, cudaEvent_t>> events(ncdps*3);
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
  init_par<<<npar, 1, 0, streams[0]>>>(par, a0, b0, c0, inc_a, inc_b, inc_c, na, nb, nc);

  // Evaluate halfoffset points in x anb y coordinates
  init_mid<<<ttraces, 1, 0, streams[0]>>>(scalco, gx, gy, sx, sy, m0x, m0y, h0);

  cudaStreamSynchronize(streams[0]);

  cudaSafe(cudaMemcpyAsync(h_m0x, m0x, sizeof(real)*ttraces, cudaMemcpyDeviceToHost, streams[0]));
  cudaSafe(cudaMemcpyAsync(h_m0y, m0y, sizeof(real)*ttraces, cudaMemcpyDeviceToHost, streams[0]));

  cudaSafe(cudaDeviceSynchronize());

  // Compute max semblances and get max C for each CDP
  for(int cdp_id=0; cdp_id < ncdps; cdp_id++) {
    int id = (cdp_id)%ng;

    real m0x_cdp_id = (cdp_id > 0) ? h_m0x[h_ntraces_by_cdp_id[cdp_id-1]] : 0;
    real m0y_cdp_id = (cdp_id > 0) ? h_m0y[h_ntraces_by_cdp_id[cdp_id-1]] : 0;
    int cdp0 = gather.cdps_by_cdp_id()[cdp_id].front();
    int cdpf = gather.cdps_by_cdp_id()[cdp_id].back();
    int t_id0 = cdp0 > 0 ? h_ntraces_by_cdp_id[cdp0-1] : 0;
    int t_idf = h_ntraces_by_cdp_id[cdpf];
    int ntraces = t_idf - t_id0;

    cudaSafe(cudaMemcpyAsync(cdpsmpls[id], h_samples+t_id0*ns, ntraces*ns*sizeof(real), cudaMemcpyHostToDevice, streams[id]));

    cudaEventRecord(events[3*cdp_id].first, streams[id]);
    compute_points_for_gather<<<cdpf-cdp0+1, 1, 0, streams[id]>>>(m0x, m0y, h0, hs[id], m2s[id], ms[id], ntraces_by_cdp_id, m0x_cdp_id, m0y_cdp_id, cdp0, cdpf);
    cudaEventRecord(events[3*cdp_id].second, streams[id]);

    // Compute semblances for each c for each sample
    cudaEventRecord(events[3*cdp_id+1].first, streams[id]);
    compute_semblances<<<(ns*npar+NTHREADS-1)/NTHREADS, NTHREADS, 0, streams[id]>>>(hs[id], m2s[id], ms[id], par, cdpsmpls[id], nums[id], stts[id], ntraces, idt, dt, tau, w, npar, ns);
    cudaEventRecord(events[3*cdp_id+1].second, streams[id]);

    // Get max C for max semblance for each sample on this cdp
    cudaEventRecord(events[3*cdp_id+2].first, streams[id]);
    redux_semblances<<<(ns+NTHREADS-1)/NTHREADS, NTHREADS, 0, streams[id]>>>(nums[id], stts[id], ctr, str, stk, npar, cdp_id, ns);
    cudaEventRecord(events[3*cdp_id+2].second, streams[id]);

    number_of_semblances += ntraces;

    LOG(DEBUG, "CUDA Progress: " + std::to_string(cdp_id) + "/" + std::to_string(ncdps));
  }
  // Wait for all commands to finish
  // DO NOT REMOVE THIS, OTHERWISE THE METRICS ON THE NEXT LINES WILL HAVE BAD VALUE
  cudaSafe(cudaDeviceSynchronize());

  end = std::chrono::high_resolution_clock::now();

  for(auto& e: events) {
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, e.first, e.second);

    kernel_total_exec_time += milliseconds/1e3;
  }

  // Copy results back to host
  cudaSafe(cudaMemcpy(h_ctr, ctr, sizeof(int ) * ncdps * ns, cudaMemcpyDeviceToHost));
  cudaSafe(cudaMemcpy(h_str, str, sizeof(real) * ncdps * ns, cudaMemcpyDeviceToHost));
  cudaSafe(cudaMemcpy(h_stk, stk, sizeof(real) * ncdps * ns, cudaMemcpyDeviceToHost));

  // Logs stats (exec time and semblance-traces per second)
  double total_exec_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - beg).count();
  double stps = (number_of_semblances / 1e9 ) * (ns * npar / total_exec_time);
  double kernel_stps = (number_of_semblances / 1e9 ) * (ns * npar / kernel_total_exec_time);
  std::string stats = "Total Execution Time: " + std::to_string(total_exec_time);
  stats += ": Giga-Semblances-Trace/s: " + std::to_string(stps);
  stats += ": Kernel Execution Time: " + std::to_string(kernel_total_exec_time);
  stats += ": Kernel Giga-Semblances-Trace/s: " + std::to_string(kernel_stps);
  LOG(INFO, stats);

  // Delinearizes data anb save it into a *.su file
  for(int i=0; i < ncdps; i++) {
    su_trace atr_t = gather[i].traces()[0];
    su_trace btr_t = gather[i].traces()[0];
    su_trace ctr_t = gather[i].traces()[0];
    su_trace str_t = gather[i].traces()[0];
    su_trace stk_t = gather[i].traces()[0];

    atr_t.offset() = 0;
    atr_t.sx() = atr_t.gx() = (gather[i].traces()[0].sx() + gather[i].traces()[0].gx()) >> 1;
    atr_t.sy() = atr_t.gy() = (gather[i].traces()[0].sy() + gather[i].traces()[0].gy()) >> 1;
    btr_t.offset() = 0;
    btr_t.sx() = btr_t.gx() = (gather[i].traces()[0].sx() + gather[i].traces()[0].gx()) >> 1;
    btr_t.sy() = btr_t.gy() = (gather[i].traces()[0].sy() + gather[i].traces()[0].gy()) >> 1;
    ctr_t.offset() = 0;
    ctr_t.sx() = ctr_t.gx() = (gather[i].traces()[0].sx() + gather[i].traces()[0].gx()) >> 1;
    ctr_t.sy() = ctr_t.gy() = (gather[i].traces()[0].sy() + gather[i].traces()[0].gy()) >> 1;

    for(int k=0; k < ns; k++) atr_t.data()[k] = (a0 + inc_a * (h_ctr[i*ns+k]/(nc*nb)));
    for(int k=0; k < ns; k++) btr_t.data()[k] = (b0 + inc_b * ((h_ctr[i*ns+k]/nc)%nb));
    for(int k=0; k < ns; k++) ctr_t.data()[k] = (c0 + inc_c * (h_ctr[i*ns+k]%nc)) / FACTOR;
    str_t.data().assign(h_str + i*ns, h_str + (i+1)*ns);
    stk_t.data().assign(h_stk + i*ns, h_stk + (i+1)*ns);

    atr_t.fputtr(a_out);
    btr_t.fputtr(b_out);
    ctr_t.fputtr(c_out);
    str_t.fputtr(s_out);
    stk_t.fputtr(stack);
  }

  cudaSafe(cudaFree(gx     ));
  cudaSafe(cudaFree(gy     ));
  cudaSafe(cudaFree(sx     ));
  cudaSafe(cudaFree(sy     ));
  cudaSafe(cudaFree(scalco ));
  cudaSafe(cudaFree(h0     ));
  cudaSafe(cudaFree(m0x    ));
  cudaSafe(cudaFree(m0y    ));
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
  delete [] h_ntraces_by_cdp_id ;

  return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
