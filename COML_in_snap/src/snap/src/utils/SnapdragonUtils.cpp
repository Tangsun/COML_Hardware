/****************************************************************************
 *   Copyright (c) 2017 Brett T. Lopez. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name snap nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/
#include "SnapdragonUtils.hpp"

// Quaterion product
void  qProd(Quaternion &c, Quaternion a, Quaternion b){
  c.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
  c.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
  c.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
  c.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
}

// Conjugate quaternion product
void  qConjProd(Quaternion &c, Quaternion a, Quaternion b){
  a.x = -a.x;
  a.y = -a.y;
  a.z = -a.z;

  c.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
  c.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
  c.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
  c.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
}

// Vector to quaternion
void vec2quat(Quaternion &q , Vector v){
	q.w = 0;
	q.x = v.x;
	q.y = v.y;
	q.z = v.z;
}

// Saturate value
float saturate(float val, float min, float max){
	if (val>max){val=max;}
	if (val<min){val=min;}
	return val;
}