//********************************************************
//
// Jamie Shotton
// Machine Intelligence Laboratory
// Department of Engineering
// University of Cambridge, UK
// Copyright (c) 2006
// All Rights Reserved.
//
//  Permission is hereby granted, free of charge, to use and distribute
//  this software and its documentation without restriction, including
//  without limitation the rights to use, copy, modify, merge, publish,
//  distribute, sublicense, and/or sell copies of this work, and to
//  permit persons to whom this work is furnished to do so, subject to
//  the following conditions:
//   1. The code must retain the above copyright notice, this list of
//      conditions and the following disclaimer.
//   2. Any modifications must be clearly marked as such.
//   3. Original authors' names are not deleted.
//   4. The authors' names are not used to endorse or promote products
//      derived from this software without specific prior written
//      permission.
//
//  THE UNIVERSITY OF CAMBRIDGE AND THE CONTRIBUTORS TO THIS WORK
//  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
//  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
//  SHALL THE UNIVERSITY OF CAMBRIDGE NOR THE CONTRIBUTORS BE LIABLE
//  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
//  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
//  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
//  THIS SOFTWARE.
//
//********************************************************
//                      Author :  Jamie Shotton
//                      Date   :  May 2006
//  This work pertains to the research described in the ECCV 2006 paper
//  TextonBoost: Joint Appearance, Shape and Contex Modeling
//  for Multi-Class Object Recognition and Segmentation
//********************************************************



using System;
using System.Collections.Generic;
using System.Text;

namespace Misc
{
    public class MyRandom : Random
    {
        public MyRandom() : base() { }

        public MyRandom(int Seed) : base(Seed) { }

        public double NextNormal() // N(0,1)
        {
            // Box Muller transform
            double x1 = NextDouble();
            double x2 = NextDouble();
            double z1 = Math.Sqrt(-2 * Math.Log(x1)) * Math.Cos(2 * Math.PI * x2);
            //double z2 = Math.Sqrt(-2*Math.Log(x1)) * Math.Sin(2*Math.PI*x2);
            return z1;
        }

        public double NextNormal(double mean, double variance) // N(mean,variance)
        {
            double normal = NextNormal();

            return normal * Math.Sqrt(variance) + mean;
        }

        public int RndRange(int from, int to)
        {
            return Next(from, to);
        }

        public float RndRange(float from, float to)
        {
            if (from == to)
                return from;
            else
                return (float) (NextDouble() * (to - from) + from);
        }
    }
}
