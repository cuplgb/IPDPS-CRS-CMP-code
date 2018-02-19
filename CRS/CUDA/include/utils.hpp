////////////////////////////////////////////////////////////////////////////////
/**
 * @file utils.hpp
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

#ifndef UTILS_HPP
#define UTILS_HPP

////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////

//#define DOUBLE

////////////////////////////////////////////////////////////////////////////////

#ifdef DOUBLE
#define real double
#else
#define real float
#endif

////////////////////////////////////////////////////////////////////////////////

typedef struct real4_t
{
  real a,b,c,d;
} real4;

////////////////////////////////////////////////////////////////////////////////

#ifndef NDEBUG
  #define cl_safe(exp) _cl_safe(exp, __FILE__, __LINE__)
#else
  #define cl_safe(exp)
#endif

////////////////////////////////////////////////////////////////////////////////

#endif /*! UTILS_HPP */

////////////////////////////////////////////////////////////////////////////////
