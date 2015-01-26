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
    public static class MyMath
    {
        public static readonly float PIf = (float) Math.PI;

        public static int Mod(int x, int b)
        {
            if (b <= 0)
                throw new ArgumentException("Mod base must be positive");
            while (x < 0)
                x += b;
            while (x >= b)
                x -= b;
            return x;
        }

        public static float Mod(float x, float b)
        {
            if (b <= 0)
                throw new ArgumentException("Mod base must be positive");
            while (x < 0)
                x += b;
            while (x >= b)
                x -= b;
            return x;
        }

        public static double Mod(double x, double b)
        {
            if (b <= 0)
                throw new ArgumentException("Mod base must be positive");
            while (x < 0)
                x += b;
            while (x >= b)
                x -= b;
            return x;
        }

        public static double Gaussian(double x, double mean, double variance)
        {
            return 1.0 / Math.Sqrt(2.0 * Math.PI * variance) * Math.Exp(-(x - mean) * (x - mean) / (2.0 * variance));
        }
    }

    public class LEQComparer<T> : IComparer<T> where T : IComparable
    {
        public int Compare(T x, T y)
        {
            if (x.CompareTo(y) <= 0) return -1;
            else return 1;
        }
    }

    public class LEQPairComparerDI : IComparer<KeyValuePair<double, int>>
    {
        public int Compare(KeyValuePair<double, int> x, KeyValuePair<double, int> y)
        {
            if (x.Key <= y.Key) return -1;
            else return 1;
        }
    }


    public class LEQPairComparer<T1,T2> : IComparer<KeyValuePair<T1,T2>> where T1 : IComparable
    {
        public int Compare(KeyValuePair<T1, T2> x, KeyValuePair<T1, T2> y)
        {
            if (x.Key.CompareTo(y.Key) <= 0) return -1;
            else return 1;
        }
    }
}
