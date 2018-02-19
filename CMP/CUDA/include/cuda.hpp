////////////////////////////////////////////////////////////////////////////////
/**
 * @file cuda.hpp
 * @date 2017-03-31
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

#ifndef CUDA_HPP
#define CUDA_HPP

////////////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <builtin_types.h>

////////////////////////////////////////////////////////////////////////////////
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define cudaSafe(err)  __cudaSafe (err, __FILE__, __LINE__)

void __cudaSafe(cudaError_t err, const char *file, const int line);

////////////////////////////////////////////////////////////////////////////////

class cuda {
  private:
    int _cuDevice;

  public:
    cuda(size_t dev_id=0);

};

////////////////////////////////////////////////////////////////////////////////

#endif /*! CUDA_HPP */

////////////////////////////////////////////////////////////////////////////////
