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
using System.Drawing;
using System.Text;

using Image;

namespace ImageProcessing
{
    public static class Transformations
    {
        public static Image<byte> ResizeImage(Image<byte> input, double scaleFactor)
        {
            if (input.Bands != 1 && input.Bands != 3)
                throw new ArgumentException("Resize only implemented for images with 1 or 3 bands");

            Bitmap bInput = ImageIO.ConvertToBitmap<byte>(input);

            int newWidth = (int)Math.Round(input.Width * scaleFactor);
            int newHeight = (int)Math.Round(input.Height * scaleFactor);

            Bitmap bOutput = new Bitmap(bInput, new Size(newWidth, newHeight));

            return ImageIO.ConvertFromBitmap<byte>(bOutput);
        }

        public static Image<int> ResizeImage(Image<int> input, double scaleFactor)
        {
            if (input.Bands != 1)
                throw new ArgumentException("Resize only implemented for images with 1 band");

            int newWidth = (int)Math.Round(input.Width * scaleFactor);
            int newHeight = (int)Math.Round(input.Height * scaleFactor);

            Image<int> output = new Image<int>(newHeight, newWidth);

            foreach (Point p in output.Bounds)
            {
                int y = Math.Min(input.Height - 1, (int)Math.Round(p.Y / scaleFactor));
                int x = Math.Min(input.Width - 1, (int)Math.Round(p.X / scaleFactor));
                output[p] = input[y, x];
            }

            return output;
        }
    }
}
