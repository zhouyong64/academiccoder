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
    public static class ColourConversion
    {
        #region Colour Conversion

        public static void MonochromeToRGB<T>(Image<T> imgOut, Image<T> imgIn) where T : struct
        {
            if (imgOut.Height != imgIn.Height || imgOut.Width != imgIn.Width || imgIn.Bands != 1 || imgOut.Bands != 3)
                throw new ArgumentException("ColourConversion.MonochromeToRGB takes images of equal width and height, the input having 1 band, the output 3");

            for (int band = 0; band < 3; band++)
                new VirtualImage<T>(imgOut, band).OverwriteImage<T>(imgIn);
        }

        public static void RGBToMonochrome<T>(Image<T> imgOut, Image<T> imgIn) where T : struct
        {
            if (imgOut.Height != imgIn.Height || imgOut.Width != imgIn.Width || imgIn.Bands != 3 || imgOut.Bands != 1)
                throw new ArgumentException("ColourConversion.RGBToMonochrome takes images of equal width and height, the input having 3 bands, the output only 1");

            string convert = @"
                A1[0] = (T1) (0.299 * (double) A2[0] + 0.587 * (double) A2[1] + 0.114 * (double) A2[2]);
    
//                double R = ((double) A2[0]) / 255.0;
//                double G = ((double) A2[1]) / 255.0;
//                double B = ((double) A2[2]) / 255.0;
//
//                // Convert to XYZ
//	            double Y =  0.212671 * R + 0.715160 * G + 0.072169 * B;
//
//                // Convert to Lab
//	            double yr = Y/Yn;
//	
//                double L;
//            	if (yr>thresh)
//                    L = 116.0 * Math.Pow(yr, r13) - 16.0;
//	            else
//                    L = 903.3 * yr;
//
//                // Store result
//                A1[0] = (T1) L;
            ";

            string initialisation = @"
//                const double Yn = 1.000;
//                const double thresh=0.008856, r13=1.0/3.0, r16116=16.0/116.0;
            ";

            Image<T>.BinaryOperationPerPixel<T, T>(convert, initialisation, "", imgOut, imgIn, null);

        }

        public static void RGBToLab<T>(Image<T> imgOut, Image<T> imgIn) where T : struct
        {
            if (imgOut.Height != imgIn.Height || imgOut.Width != imgIn.Width || imgIn.Bands != 3 || imgOut.Bands != 3)
                throw new ArgumentException("ColourConversion.RGBToLab takes images of equal width and height, the input and output both having 3 bands");

            string convert = @"
                double R = ((double) A2[0]) / 255.0;
                double G = ((double) A2[1]) / 255.0;
                double B = ((double) A2[2]) / 255.0;

                // Convert to XYZ
	            double X =  0.412453 * R + 0.357580 * G + 0.180423 * B;
	            double Y =  0.212671 * R + 0.715160 * G + 0.072169 * B;
	            double Z =  0.019334 * R + 0.119193 * G + 0.950227 * B;

                // Convert to Lab
	            double xr = X/Xn, yr = Y/Yn, zr = Z/Zn;
	
                double L, a, b;
            	if (yr>thresh)
                    L = 116.0 * Math.Pow(yr, r13) - 16.0;
	            else
                    L = 903.3 * yr;

	            double fxr, fyr, fzr;

                if (xr>thresh)
                    fxr = Math.Pow(xr, r13);
	            else
                    fxr = 7.787*xr + r16116;

	            if (yr>thresh)
                    fyr = Math.Pow(yr, r13);
	            else
                    fyr = 7.787*yr + r16116;

                if (zr>thresh)
                    fzr = Math.Pow(zr, r13);
	            else
                    fzr = 7.787*zr + r16116;

	            a = 500.0 * (fxr - fyr);
	            b = 200.0 * (fyr - fzr);

                // Store result
                A1[0] = (T1) L;
                A1[1] = (T1) a;
                A1[2] = (T1) b;
            ";

            string initialisation = @"
                const double Xn = 0.950456, Yn = 1.000, Zn = 1.088854;
                const double thresh=0.008856, r13=1.0/3.0, r16116=16.0/116.0;
            ";

            Image<T>.BinaryOperationPerPixel<T, T>(convert, initialisation, "", imgOut, imgIn, null);
        }

        public static void LabToRGB<T>(Image<T> imgOut, Image<T> imgIn) where T : struct
        {
            if (imgOut.Height != imgIn.Height || imgOut.Width != imgIn.Width || imgIn.Bands != 3 || imgOut.Bands != 3)
                throw new ArgumentException("ColourConversion.LabToRGB takes images of equal width and height, the input and output both having 3 bands");

            string convert = @"
                double L = (double) A2[0];
                double a = (double) A2[1];
                double b = (double) A2[2];

                // Convert Lab to XYZ
                double X, Y, Z;
                double P = (L+16.0)/116.0;

	            if (L>7.9996)
                    Y = Yn * P * P * P;
	            else
                    Y = Yn * L / 903.3;

                double yr = Y/Yn, fy;
	            if (yr>thresh)
                    fy = Math.Pow(yr, r13);
	            else
                    fy = 7.787 * yr + r16116;

	            double fx = a / 500.0 + fy, fz = fy - b / 200.0;

	            if (fx>thresh2)
                    X = Xn*fx*fx*fx;
	            else
                    X = Xn/7.787 * ( fx - r16116 );

	            if (fz>thresh2)
                    Z = Zn*fz*fz*fz;
	            else
                    Z = Zn / 7.787 * ( fz - r16116 );

                // Convert to RGB
	            double R =   3.240479 * X - 1.537150 * Y - 0.498535 * Z;
	            double G = - 0.969256 * X + 1.875992 * Y + 0.041556 * Z;
	            double B =   0.055648 * X - 0.204043 * Y + 1.057311 * Z;

                // Store result
                A1[0] = (T1) Math.Max(0.0, Math.Min(255.0, 255.0*R));;
                A1[1] = (T1) Math.Max(0.0, Math.Min(255.0, 255.0*G));;
                A1[2] = (T1) Math.Max(0.0, Math.Min(255.0, 255.0*B));;
            ";

            string initialisation = @"
	            const double Xn = 0.950456, Yn = 1.000, Zn = 1.088854;
                const double thresh=0.008856, thresh2=0.2069, r13=1.0/3.0, r16116=16.0/116.0;
            ";

            Image<T>.BinaryOperationPerPixel<T, T>(convert, initialisation, "", imgOut, imgIn, null);
        }

        #endregion

        #region Convenience wrappers

        public static Image<T> MonochromeToRGB<T>(Image<T> imgIn) where T : struct
        {
            Image<T> imgOut = new Image<T>(imgIn.Size, 3);
            MonochromeToRGB(imgOut, imgIn);
            return imgOut;
        }

        public static Image<T> RGBToMonochrome<T>(Image<T> imgIn) where T : struct
        {
            Image<T> imgOut = new Image<T>(imgIn.Size, 1);
            RGBToMonochrome(imgOut, imgIn);
            return imgOut;
        }

        public static Image<T> RGBToLab<T>(Image<T> imgIn) where T : struct
        {
            Image<T> imgOut = new Image<T>(imgIn.Size, 3);
            RGBToLab(imgOut, imgIn);
            return imgOut;
        }

        public static Image<T> LabToRGB<T>(Image<T> imgIn) where T : struct
        {
            Image<T> imgOut = new Image<T>(imgIn.Size, 3);
            LabToRGB(imgOut, imgIn);
            return imgOut;
        }

        #endregion
    }
}
