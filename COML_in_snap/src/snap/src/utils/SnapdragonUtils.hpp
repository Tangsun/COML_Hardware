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
#ifndef STRUCT_
#define STRUCT_
#include "structs.h"
#endif

/**
 * Quaternion product
 * c = a x b
 * @param c Final quaternion
 * @param a First quaternion
 * @param b Second quaternion
 **/
void  qProd(Quaternion &c, Quaternion a, Quaternion b);

/**
 * Conjugate quaternion product
 * c = a* x b
 * @param c Final quaternion
 * @param a First quaternion - conjugate created internally
 * @param b Second quaternion
 **/
void  qConjProd(Quaternion &c, Quaternion a, Quaternion b);

/**
 * Vector to quaternion conversion
 * q = [0 v]
 * @param q Final quaternion
 * @param v Vector
 **/
void vec2quat(Quaternion &q , Vector v);

/**
 * Saturates the input value
 * @param val Input value
 * @param min Minimum value
 * @param max Maximum value
 * @return Saturated value 
 **/
float saturate(float val, float min, float max);
