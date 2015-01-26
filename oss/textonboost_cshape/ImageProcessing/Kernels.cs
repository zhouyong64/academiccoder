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

using Misc;
using Image;

namespace ImageProcessing
{
    public enum ExtendMode { Black, Extend }

    public partial class Kernel1D<T> where T : struct
    {
        public readonly int centre, size;

        public T[] data;

        public Kernel1D(int size, int centre)
        {
            this.size = size;
            this.centre = centre;
            this.data = new T[size];
        }

        #region Convolution Routines

        public Image<T> ConvolveY(Image<T> imgIn, ExtendMode extend)
        {
            Image<T> imgOut = new Image<T>(imgIn.Size, imgIn.Bands);
            ConvolveY(imgOut, imgIn, extend);
            return imgOut;
        }

        public Image<T> ConvolveX(Image<T> imgIn, ExtendMode extend)
        {
            Image<T> imgOut = new Image<T>(imgIn.Size, imgIn.Bands);
            ConvolveX(imgOut, imgIn, extend);
            return imgOut;
        }

        #endregion

        #region Static kernel creation routines

        public static Kernel1D<T> CreateGaussian(double sigma)
        {
            string genericType = typeof(T).ToString();
            string hashString = "CreateGaussianKernel1D." + genericType;
            DynamicCode dCode = DynamicCode.Hash(hashString);
            if (dCode == null)
            {
                string code = @"
                    int halfSize = (int) Math.Ceiling(3.0*sigma);
                    Kernel1D<T1> kernel = new Kernel1D<T1>(2*halfSize + 1, halfSize);

                    double s2 = sigma * sigma;
                    double f = 1.0 / Math.Sqrt(2.0 * Math.PI * s2);
                    double w2 = 1.0 / (2.0 * s2);

                    for(int i = 0; i < kernel.size; i++)
                    {
                        int p = i - kernel.centre;
                        kernel.data[i] = (T1) (f * Math.Exp(-(p * p) * w2));
                    }
                    RET = kernel;
                ";
                dCode = new DynamicCode(new string[] { genericType }, "double sigma", code);
                DynamicCode.AddHash(hashString, dCode);
            }

            return (Kernel1D<T>) dCode.Invoke(null, sigma);
        }

        public static Kernel1D<T> CreateGaussianDerivative(double sigma)
        {
            string genericType = typeof(T).ToString();
            string hashString = "CreateGaussianDerivativeKernel1D." + genericType;
            DynamicCode dCode = DynamicCode.Hash(hashString);
            if (dCode == null)
            {
                string code = @"
                    int halfSize = (int) Math.Ceiling(3.0*sigma);
                    Kernel1D<T1> kernel = new Kernel1D<T1>(2*halfSize + 1, halfSize);

                    double s2 = sigma * sigma;
                    double f = 1.0 / Math.Sqrt(2.0 * Math.PI * s2);
                    double w = 1.0 / s2;
                    double w2 = 1.0 / (2.0 * s2);

                    for(int i = 0; i < kernel.size; i++)
                    {
                        int p = i - kernel.centre;
                        kernel.data[i] = (T1) (-p * w * f * Math.Exp(-(p * p) * w2));
                    }
                    RET = kernel;
                ";
                dCode = new DynamicCode(new string[] { genericType }, "double sigma", code);
                DynamicCode.AddHash(hashString, dCode);
            }

            return (Kernel1D<T>) dCode.Invoke(null, sigma);
        }

        #endregion

        public static Kernel1D<T> CreateGaussian2ndDerivative(double sigma)
        {
            string genericType = typeof(T).ToString();
            string hashString = "CreateGaussian2ndDerivativeKernel1D." + genericType;
            DynamicCode dCode = DynamicCode.Hash(hashString);
            if (dCode == null)
            {
                string code = @"
                    int halfSize = (int) Math.Ceiling(3.0*sigma);
                    Kernel1D<T1> kernel = new Kernel1D<T1>(2*halfSize + 1, halfSize);

                    double s2 = sigma * sigma;
                    double f = 1.0 / Math.Sqrt(2.0 * Math.PI * s2);
                    double w = 1.0 / s2;
                    double w2 = 1.0 / (2.0 * s2);

                    for(int i = 0; i < kernel.size; i++)
                    {
                        int p = i - kernel.centre;
                        kernel.data[i] = (T1) ((p * p * w * w - w) * f * Math.Exp(-(p * p) * w2));
                    }
                    RET = kernel;

                ";
                dCode = new DynamicCode(new string[] { genericType }, "double sigma", code);
                DynamicCode.AddHash(hashString, dCode);
            }

            return (Kernel1D<T>)dCode.Invoke(null, sigma);
        }
    }

    public abstract class Kernel2D<T> where T : struct
    {
        public Image<T> Convolve(Image<T> imgIn, ExtendMode extend)
        {
            Image<T> imgOut = new Image<T>(imgIn.Size, imgIn.Bands);
            Convolve(imgOut, imgIn, extend);
            return imgOut;
        }

        public abstract void Convolve(Image<T> imgOut, Image<T> imgIn, ExtendMode extend);

        #region Static kernel creation routines

        public static Kernel2D<T> CreateGaussian(double sigma)
        {
            SeparableKernel2D<T> kernel = new SeparableKernel2D<T>();
            
            kernel.kernelX = Kernel1D<T>.CreateGaussian(sigma);
            kernel.kernelY = kernel.kernelX;

            return kernel;
        }

        public static Kernel2D<T> CreateGaussianDerivativeX(double sigma)
        {
            return CreateGaussianDerivativeX(sigma, sigma);
        }

        public static Kernel2D<T> CreateGaussianDerivativeX(double sigmaX, double sigmaY)
        {
            SeparableKernel2D<T> kernel = new SeparableKernel2D<T>();

            kernel.kernelX = Kernel1D<T>.CreateGaussianDerivative(sigmaX);
            kernel.kernelY = Kernel1D<T>.CreateGaussian(sigmaY);

            return kernel;
        }

        public static Kernel2D<T> CreateGaussianDerivativeY(double sigma)
        {
            return CreateGaussianDerivativeY(sigma, sigma);
        }

        public static Kernel2D<T> CreateGaussianDerivativeY(double sigmaX, double sigmaY)
        {
            SeparableKernel2D<T> kernel = (SeparableKernel2D<T>) Kernel2D<T>.CreateGaussianDerivativeX(sigmaY, sigmaX);

            // Swap x and y
            Kernel1D<T> temp = kernel.kernelX;
            kernel.kernelX = kernel.kernelY;
            kernel.kernelY = temp;

            return kernel;
        }

        public static Kernel2D<T> CreateLaplacian(double sigma)
        {
            LoGKernel2D<T> kernel = new LoGKernel2D<T>();

            kernel.kernel2ndDerivative = Kernel1D<T>.CreateGaussian2ndDerivative(sigma);
            kernel.kernelGaussian = Kernel1D<T>.CreateGaussian(sigma);

            return kernel;
        }


        // TODO Add more 2d filters

        #endregion
    }

    public class SeparableKernel2D<T> : Kernel2D<T> where T : struct
    {
        public Kernel1D<T> kernelY, kernelX;

        public override void Convolve(Image<T> imgOut, Image<T> imgIn, ExtendMode extend)
        {
            if (!imgOut.DimensionsEqual(imgIn))
                throw new ArgumentException("Convolution routines require input and output images to have same dimensions");

            Image<T> imgTemp = new Image<T>(imgIn.Size, imgIn.Bands);

            kernelY.ConvolveY(imgTemp, imgIn, extend);
            kernelX.ConvolveX(imgOut, imgTemp, extend);            
        }
    }

    public class NonSeparableKernel2D<T> : Kernel2D<T> where T : struct
    {
        public override void Convolve(Image<T> imgOut, Image<T> imgIn, ExtendMode extend)
        {
            if (!imgOut.DimensionsEqual(imgIn))
                throw new ArgumentException("Convolution routines require input and output images to have same dimensions");

            // NB use virtualimage compatible code

            // TODO implement non separable filters
            throw new Exception("The method or operation is not implemented.");
        }
    }

    public class LoGKernel2D<T> : Kernel2D<T> where T : struct
    {
        public Kernel1D<T> kernel2ndDerivative, kernelGaussian;

        public override void Convolve(Image<T> imgOut, Image<T> imgIn, ExtendMode extend)
        {
            if (!imgOut.DimensionsEqual(imgIn))
                throw new ArgumentException("Convolution routines require input and output images to have same dimensions");

            Image<T> imgTemp = new Image<T>(imgIn.Size, imgIn.Bands);
            Image<T> imgTemp2 = new Image<T>(imgIn.Size, imgIn.Bands);

            // Convolve in one direction
            kernel2ndDerivative.ConvolveY(imgTemp, imgIn, extend);
            kernelGaussian.ConvolveX(imgOut, imgTemp, extend);

            // Convolve in other direction
            kernel2ndDerivative.ConvolveX(imgTemp, imgIn, extend);
            kernelGaussian.ConvolveY(imgTemp2, imgTemp, extend);

            // Sum results
            Image<T>.BinaryOperation("V1 = (T1) (V1+V2);", "", "", imgOut, imgTemp2, null);
        }
    }
}
