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
    public class Range<T> : IEnumerable<T>
    {
        IEnumerable<T> enumerable;
        int count = -1;

        public int Count
        {
            get
            {
                if (count == -1)
                {
                    count = 0;
                    foreach (T t in enumerable)
                        count++;
                }
                return count;
            }
        }

        public static Range<int> UniformInteger(int from, int to) { return UniformInteger(from, to, 1); }
        public static Range<int> UniformInteger(int from, int to, int step)
        {
            Range<int> range = new Range<int>();
            range.enumerable = Private_UniformInteger(from, to, step);
            return range;
        }

        public static Range<int> ConcatenatedInteger(Range<int> part1, Range<int> part2)
        {
            Range<int> range = new Range<int>();
            range.enumerable = Private_ConcatenatedInteger(part1.enumerable, part2.enumerable);
            return range;
        }

        public static Range<float> UniformFloat(float from, float to, float step)
        {
            Range<float> range = new Range<float>();
            range.enumerable = Private_UniformFloat(from, to, step);
            return range;
        }

        #region Static IEnumerable<> creators

        private static IEnumerable<int> Private_UniformInteger(int from, int to, int step)
        {
            for(int i=from; i<to; i+= step)
                yield return i;
        }

        private static IEnumerable<int> Private_ConcatenatedInteger(IEnumerable<int> part1, IEnumerable<int> part2)
        {
            foreach (int i in part1)
                yield return i;
            foreach (int i in part2)
                yield return i;
        }

        private static IEnumerable<float> Private_UniformFloat(float from, float to, float step)
        {
            for (float f = from; f < to; f += step)
                yield return f;
        }

        #endregion

        #region IEnumerable and IEnumerable<T> Members

        public IEnumerator<T> GetEnumerator()
        {
            return enumerable.GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return enumerable.GetEnumerator();
        }

        #endregion
    }
}