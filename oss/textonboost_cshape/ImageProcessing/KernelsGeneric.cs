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
using System.Text.RegularExpressions;

using Misc;
using Image;

namespace ImageProcessing
{
    public partial class Kernel1D<T> where T : struct
    {
        private enum ConvolveDirection { X, Y };

        public void ConvolveY(Image<T> imgOut, Image<T> imgIn, ExtendMode extend)
        {
            Convolve(ConvolveDirection.Y, imgOut, imgIn, extend);
        }

        public void ConvolveX(Image<T> imgOut, Image<T> imgIn, ExtendMode extend)
        {
            Convolve(ConvolveDirection.X, imgOut, imgIn, extend);
        }
        
        private void Convolve(ConvolveDirection direction, Image<T> imgOut, Image<T> imgIn, ExtendMode extend)
        {
            // Check dimensions are equal
            if (!imgOut.DimensionsEqual(imgIn))
                throw new ArgumentException("Convolution routines require input and output images to have same dimensions");

            // Check not overwriting data
            if (imgOut == imgIn)
                throw new ArgumentException("Convolve overwrites data and requires two separate images");

            string type = imgOut.GenericType.ToString();

            // Retrieve a convolution object
            string hashString = "Convolve." + direction + "." + extend + "." + this.centre + "." + this.size +  ".Image<" + type + ">";
            DynamicCode dCode = DynamicCode.Hash(hashString);
            if (dCode == null)
            {
                string code = "";
                switch (extend)
                {
                    case ExtendMode.Black:
                        code = @"
                            fixed(T1* dataOutBase = dataOut)
                            fixed(T2* dataInBase = dataIn)
                            fixed(T3* kernelBase = kernel)
                            {
                                T1* dataOutPtr = dataOutBase + startOut;
                                T2* dataInPtr = dataInBase + startIn;
                                
                                for(int y=0; y<height; y++, dataOutPtr+=incYOut, dataInPtr+=incYIn)
                                    for(int x=0; x<width; x++, dataOutPtr+=incXOut, dataInPtr+=incXIn)
                                        for(int b=0; b<bands; b++, dataOutPtr++, dataInPtr++)
                                        {
                                            T2 val = 0;
                                            int convOffStart = -KERNELCENTRE; if (convOffStart+CONVDIM < 0) convOffStart = -CONVDIM;
                                            int convOffEnd = -KERNELCENTRE + KERNELSIZE - 1; if (convOffEnd+CONVDIM >= CONVEXT) convOffEnd = CONVEXT - 1 - CONVDIM;

                                            T2* dataInPtrConv = dataInPtr + convOffStart * incConvIn;
                                            T3* kernelPtrConv = kernelBase + (convOffStart + KERNELCENTRE);

                                            for(int convOff = convOffStart; convOff<=convOffEnd; convOff++, dataInPtrConv += incConvIn, kernelPtrConv ++)
                                                val += (*dataInPtrConv) * (*kernelPtrConv);

                                            (*dataOutPtr) = (T1) val;
                                        }
                            }
                            ";
                        break;
                    case ExtendMode.Extend:
                        code = @"
                            fixed(T1* dataOutBase = dataOut)
                            fixed(T2* dataInBase = dataIn)
                            fixed(T3* kernelBase = kernel)
                            {
                                T1* dataOutPtr = dataOutBase + startOut;
                                T2* dataInPtr = dataInBase + startIn;
                                
                                for(int y=0; y<height; y++, dataOutPtr+=incYOut, dataInPtr+=incYIn)
                                    for(int x=0; x<width; x++, dataOutPtr+=incXOut, dataInPtr+=incXIn)
                                        for(int b=0; b<bands; b++, dataOutPtr++, dataInPtr++)
                                        {
                                            T2 val = 0;

                                            int convOffStartTrue = -KERNELCENTRE, convOffEndTrue = -KERNELCENTRE + KERNELSIZE - 1;
                                            int convOffStart = convOffStartTrue; if (convOffStart+CONVDIM < 0) convOffStart = -CONVDIM;
                                            int convOffEnd = convOffEndTrue; if (convOffEnd+CONVDIM >= CONVEXT) convOffEnd = CONVEXT - 1 - CONVDIM;

                                            T2* dataInPtrConv = dataInPtr + convOffStart * incConvIn;
                                            T3* kernelPtrConv = kernelBase;

                                            for(int convOff = convOffStartTrue; convOff<convOffStart; convOff++, kernelPtrConv ++)
                                                val += (*dataInPtrConv) * (*kernelPtrConv);
                                            for(int convOff = convOffStart; convOff<=convOffEnd; convOff++, dataInPtrConv += incConvIn, kernelPtrConv ++)
                                                val += (*dataInPtrConv) * (*kernelPtrConv);
                                            dataInPtrConv -= incConvIn;
                                            for(int convOff = convOffEnd+1; convOff<=convOffEndTrue; convOff++, kernelPtrConv ++)
                                                val += (*dataInPtrConv) * (*kernelPtrConv);

                                            (*dataOutPtr) = (T1) val;
                                        }
                            }
                            ";
                        break;
                }

                code = Regex.Replace(code, "CONVDIM", direction == ConvolveDirection.Y ? "y" : "x");
                code = Regex.Replace(code, "CONVEXT", direction == ConvolveDirection.Y ? "height" : "width");
                code = Regex.Replace(code, "KERNELCENTRE", "(" + this.centre + ")");
                code = Regex.Replace(code, "KERNELSIZE", "(" + this.size + ")");

                dCode = new DynamicCode(new string[] { type, type, type }, @"T1[,,] dataOut, T2[,,] dataIn, T3[] kernel, int height, int width, int bands, int startOut, int incYOut, int incXOut, int startIn, int incYIn, int incXIn, int incConvIn", code);

                DynamicCode.AddHash(hashString, dCode);
            }

            // Call the convolution object
            int startOut = 0, incYOut = 0, incXOut = 0;
            if (imgOut is VirtualImage<T>)
            {
                VirtualImage<T> imgOutVI = imgOut as VirtualImage<T>;
                startOut = ((imgOutVI.yOff * imgOutVI.original.Width) + imgOutVI.xOff) * imgOutVI.original.Bands + imgOutVI.bOff;
                incYOut = (imgOutVI.original.Width - imgOutVI.Width) * imgOutVI.original.Bands;
                incXOut = imgOutVI.original.Bands - imgOutVI.Bands;
            }

            int startIn = 0, incYIn = 0, incXIn = 0;
            int incConvIn = direction == ConvolveDirection.Y ? (imgIn.Width * imgIn.Bands) : imgIn.Bands;
            if (imgIn is VirtualImage<T>)
            {
                VirtualImage<T> imgInVI = imgIn as VirtualImage<T>;
                startIn = ((imgInVI.yOff * imgInVI.original.Width) + imgInVI.xOff) * imgInVI.original.Bands + imgInVI.bOff;
                incYIn = (imgInVI.original.Width - imgInVI.Width) * imgInVI.original.Bands;
                incXIn = imgInVI.original.Bands - imgInVI.Bands;
                incConvIn = direction == ConvolveDirection.Y ? (imgInVI.original.Width * imgInVI.original.Bands) : imgInVI.original.Bands;
            }

            dCode.Invoke(null, imgOut.Data, imgIn.Data, this.data, imgOut.Height, imgOut.Width, imgOut.Bands, startOut, incYOut, incXOut, startIn, incYIn, incXIn, incConvIn);
        }
    }
}
