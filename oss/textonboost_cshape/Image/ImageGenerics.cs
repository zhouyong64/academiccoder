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
using System.Drawing;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

using Misc;

namespace Image
{
    public partial class Image<T> where T : struct
    {
        public Type GenericType
        {
            get { return typeof(T); }
        }

        public Bitmap ConvertToBitmap(IOutputMapping outputMapping)
        {
            OutputMapping<T> outputMappingTyped = (OutputMapping<T>) outputMapping;
            return ImageIO.ConvertToBitmap<T>(this, outputMappingTyped);
        }

        public static object UnaryOperation<T1>(string operation, string initialisation, string finalisation, Image<T1> img1, object inpValue)
            where T1 : struct
        {
            return UnaryOperation<T1, int>(operation, initialisation, finalisation, img1, 0, inpValue);
        }

        protected internal static object UnaryOperationPerPixel<T1>(string operation, string initialisation, string finalisation, Image<T1> img1, object inpValue)
            where T1 : struct
        {
            return UnaryOperationPerPixel<T1, int>(operation, initialisation, finalisation, img1, 0, inpValue);
        }

        protected internal static object UnaryOperationPerPixel<T1, T2>(string operation, string initialisation, string finalisation, Image<T1> img1, T2 val2, object inpValue)
            where T1 : struct
            where T2 : struct
        {
            string type1 = img1.GenericType.ToString();
            string type2 = typeof(T2).ToString();

            return VirtualImage<T>.UnaryOperationPerPixelVirtual<T1, T2>(operation, initialisation, finalisation, img1, type1, val2, type2, inpValue);
        }

        public static object UnaryOperation<T1, T2>(string operation, string initialisation, string finalisation, Image<T1> img1, T2 val2, object inpValue)
            where T1 : struct
            where T2 : struct
        {
            string type1 = img1.GenericType.ToString();
            string type2 = typeof(T2).ToString();

            if (img1 is VirtualImage<T1>)
                return VirtualImage<T>.UnaryOperationVirtual<T1, T2>(operation, initialisation, finalisation, img1, type1, val2, type2, inpValue);

            // Retrieve a copy object
            string hashString = "ImageOperation." + operation + "." + initialisation + "." + finalisation + "." + ".Image<" + type1 + ">." + type2;
            DynamicCode dCode = DynamicCode.Hash(hashString);
            if (dCode == null)
            {
                operation = Regex.Replace(operation, "V1", "(*data1Ptr)");
                operation = Regex.Replace(operation, "V2", "(data2)");
                string code = @"
                "+initialisation+@";
                fixed(T1* data1Base = data1)
                {
                    T1* data1Ptr = data1Base;
                    for(int i=0;i<count;i++, data1Ptr++)
                    {
                        "+operation+@";
                    }
                }
                "+finalisation+@";
                ";
                dCode = new DynamicCode(new string[] { type1, type2 }, @"int count, T1[,,] data1, T2 data2", code);

                DynamicCode.AddHash(hashString, dCode);
            }

            // Call the dynamic operation
            int count = img1.Height * img1.Width * img1.Bands;
            return dCode.Invoke(inpValue, count, img1.data, val2);
        }

        public static object BinaryOperation<T1, T2>(string operation, string initialisation, string finalisation, Image<T1> img1, Image<T2> img2, object inpValue)
            where T1 : struct
            where T2 : struct
        {
            string type1 = img1.GenericType.ToString();
            string type2 = img2.GenericType.ToString();

            if (img1 is VirtualImage<T1> || img2 is VirtualImage<T2>)
                return VirtualImage<T>.BinaryOperationVirtual<T1, T2>(operation, initialisation, finalisation, img1, type1, img2, type2, inpValue);

            // Check dimensions are equal
            if (!img1.DimensionsEqual(img2))
                throw new ArgumentException("two images must have same dimensions for operation");

            // Retrieve a copy object
            string hashString = "ImageOperation." + operation + "." + initialisation + "." + finalisation + "." + ".Image<" + type1 + ">.Image<" + type2 + ">";
            DynamicCode dCode = DynamicCode.Hash(hashString);
            if (dCode == null)
            {
                operation = Regex.Replace(operation, "V1", "(*data1Ptr)");
                operation = Regex.Replace(operation, "V2", "(*data2Ptr)");
                string code = @"
                    " + initialisation + @";
                    fixed(T1* data1Base = data1)
                    fixed(T2* data2Base = data2)
                    {
                        T1* data1Ptr = data1Base;
                        T2* data2Ptr = data2Base;
                        for(int i=0;i<count;i++, data1Ptr++, data2Ptr++)
                        {
                            " +operation+@";
                        }
                    }
                    " + finalisation + @";
                    ";
                dCode = new DynamicCode(new string[] { type1, type2 }, @"int count, T1[,,] data1, T2[,,] data2", code);

                DynamicCode.AddHash(hashString, dCode);
            }

            // Call the dynamic operation
            int count = img1.Height * img1.Width * img1.Bands;
            return dCode.Invoke(inpValue, count, img1.data, img2.data);
        }

        protected internal static object BinaryOperationPerPixel<T1, T2>(string operation, string initialisation, string finalisation, Image<T1> img1, Image<T2> img2, object inpValue)
            where T1 : struct
            where T2 : struct
        {
            string type1 = img1.GenericType.ToString();
            string type2 = img2.GenericType.ToString();

            return VirtualImage<T>.BinaryOperationPerPixelVirtual<T1, T2>(operation, initialisation, finalisation, img1, type1, img2, type2, inpValue);
        }
    }

    public partial class VirtualImage<T>
    {
        protected internal static object UnaryOperationVirtual<T1, T2>(string operation, string initialisation, string finalisation, Image<T1> img1, string type1, T2 val2, string type2, object inpValue)
            where T1 : struct
            where T2 : struct
        {
            // Retrieve a copy object
            string hashString = "ImageOperationVirtual." + operation + "." + initialisation + "." + finalisation + "." + ".Image<" + type1 + ">." + type2;
            DynamicCode dCode = DynamicCode.Hash(hashString);
            if (dCode == null)
            {
                operation = Regex.Replace(operation, "V1", "(*data1Ptr)");
                operation = Regex.Replace(operation, "V2", "(val2)");
                string code = @"
                    "+initialisation+@";
                    fixed(T1* data1Base = data1)
                    {
                        T1* data1Ptr = data1Base + start1;
                        
                        for(int y=0;y<height;y++, data1Ptr+=incY1)
                            for(int x=0;x<width;x++, data1Ptr+=incX1)
                                for(int b=0;b<bands;b++, data1Ptr++)
                                {
                                    " + operation + @";
                                }
                    }
                    "+finalisation+@";
                    ";
                dCode = new DynamicCode(new string[] { type1, type2 }, @"T1[,,] data1, T2 val2, int height, int width, int bands, int start1, int incY1, int incX1", code);

                DynamicCode.AddHash(hashString, dCode);
            }

            // Call the copy object
            int start1 = 0, incY1 = 0, incX1 = 0;
            if (img1 is VirtualImage<T1>)
            {
                VirtualImage<T1> img1VI = img1 as VirtualImage<T1>;
                start1 = ((img1VI.yOff * img1VI.original.Width) + img1VI.xOff) * img1VI.original.Bands + img1VI.bOff;
                incY1 = (img1VI.original.Width - img1VI.Width) * img1VI.original.Bands;
                incX1 = img1VI.original.Bands - img1VI.Bands;
            }

            return dCode.Invoke(inpValue, img1.Data, val2, img1.Height, img1.Width, img1.Bands, start1, incY1, incX1);
        }

        protected internal static object BinaryOperationVirtual<T1, T2>(string operation, string initialisation, string finalisation, Image<T1> img1, string type1, Image<T2> img2, string type2, object inpValue)
            where T1 : struct
            where T2 : struct
        {
            // Check dimensions are equal
            if (!img1.DimensionsEqual(img2))
                throw new ArgumentException("to and from images must have same dimensions");

            // Retrieve a copy object
            string hashString = "ImageOperationVirtual." + operation + "." + initialisation + "." + finalisation + "." + ".Image<" + type1 + ">.Image<" + type2 + ">";
            DynamicCode dCode = DynamicCode.Hash(hashString);
            if (dCode == null)
            {
                operation = Regex.Replace(operation, "V1", "(*data1Ptr)");
                operation = Regex.Replace(operation, "V2", "(*data2Ptr)");
                string code = @"
                    "+initialisation+@";
                    fixed(T1* data1Base = data1)
                    fixed(T2* data2Base = data2)
                    {
                        T1* data1Ptr = data1Base + start1;
                        T2* data2Ptr = data2Base + start2;
                        
                        for(int y=0;y<height;y++, data1Ptr+=incY1, data2Ptr+=incY2)
                            for(int x=0;x<width;x++, data1Ptr+=incX1, data2Ptr+=incX2)
                                for(int b=0;b<bands;b++, data1Ptr++, data2Ptr++)
                                {
                                    "+operation+@";
                                }
                    }
                    "+finalisation+@";
                    ";
                dCode = new DynamicCode(new string[] { type1, type2 }, @"T1[,,] data1, T2[,,] data2, int height, int width, int bands, int start1, int incY1, int incX1, int start2, int incY2, int incX2", code);

                DynamicCode.AddHash(hashString, dCode);
            }

            // Call the copy object
            int start1 = 0, incY1 = 0, incX1 = 0;
            if (img1 is VirtualImage<T1>)
            {
                VirtualImage<T1> img1VI = img1 as VirtualImage<T1>;
                start1 = ((img1VI.yOff * img1VI.original.Width) + img1VI.xOff) * img1VI.original.Bands + img1VI.bOff;
                incY1 = (img1VI.original.Width - img1VI.Width)*img1VI.original.Bands;
                incX1 = img1VI.original.Bands - img1VI.Bands;
            }

            int start2 = 0, incY2 = 0, incX2 = 0;
            if (img2 is VirtualImage<T2>)
            {
                VirtualImage<T2> img2VI = img2 as VirtualImage<T2>;
                start2 = ((img2VI.yOff * img2VI.original.Width) + img2VI.xOff) * img2VI.original.Bands + img2VI.bOff;
                incY2 = (img2VI.original.Width - img2VI.Width)*img2VI.original.Bands;
                incX2 = img2VI.original.Bands - img2VI.Bands;
            }

            return dCode.Invoke(inpValue, img1.Data, img2.Data, img1.Height, img1.Width, img1.Bands, start1, incY1, incX1, start2, incY2, incX2);
        }

        protected internal static object BinaryOperationPerPixelVirtual<T1, T2>(string operation, string initialisation, string finalisation, Image<T1> img1, string type1, Image<T2> img2, string type2, object inpValue)
            where T1 : struct
            where T2 : struct
        {
            // Check dimensions are equal
            if (img1.Height != img2.Height || img1.Width != img2.Width)
                throw new ArgumentException("two images must have same width and height for per pixel operation");

            // Retrieve a copy object
            string hashString = "ImageOperationPerPixelVirtual." + operation + "." + initialisation + "." + finalisation + "." + ".Image<" + type1 + ">.Image<" + type2 + ">";
            DynamicCode dCode = DynamicCode.Hash(hashString);
            if (dCode == null)
            {
                operation = Regex.Replace(operation, "A1", "data1Ptr");
                operation = Regex.Replace(operation, "A2", "data2Ptr");
                operation = Regex.Replace(operation, "V1", "(*data1Ptr)");
                operation = Regex.Replace(operation, "V2", "(*data2Ptr)");
                string code = @"
                    "+initialisation+@";
                    fixed(T1* data1Base = data1)
                    fixed(T2* data2Base = data2)
                    {
                        T1* data1Ptr = data1Base + start1;
                        T2* data2Ptr = data2Base + start2;
                        
                        for(int y=0;y<height;y++, data1Ptr+=incY1, data2Ptr+=incY2)
                            for(int x=0;x<width;x++, data1Ptr+=incX1, data2Ptr+=incX2)
                            {
                                "+operation+@";
                            }
                    }
                    "+finalisation+@";
                    ";
                dCode = new DynamicCode(new string[] { type1, type2 }, @"T1[,,] data1, T2[,,] data2, int height, int width, int start1, int incY1, int incX1, int start2, int incY2, int incX2", code);

                DynamicCode.AddHash(hashString, dCode);
            }

            // Call the copy object
            int start1 = 0, incY1 = 0, incX1 = img1.Bands;
            if (img1 is VirtualImage<T1>)
            {
                VirtualImage<T1> img1VI = img1 as VirtualImage<T1>;
                start1 = ((img1VI.yOff * img1VI.original.Width) + img1VI.xOff) * img1VI.original.Bands + img1VI.bOff;
                incY1 = (img1VI.original.Width - img1VI.Width)*img1VI.original.Bands;
                incX1 = img1VI.original.Bands;
            }

            int start2 = 0, incY2 = 0, incX2 = img2.Bands;
            if (img2 is VirtualImage<T2>)
            {
                VirtualImage<T2> img2VI = img2 as VirtualImage<T2>;
                start2 = ((img2VI.yOff * img2VI.original.Width) + img2VI.xOff) * img2VI.original.Bands + img2VI.bOff;
                incY2 = (img2VI.original.Width - img2VI.Width)*img2VI.original.Bands;
                incX2 = img2VI.original.Bands;
            }

            return dCode.Invoke(inpValue, img1.Data, img2.Data, img1.Height, img1.Width, start1, incY1, incX1, start2, incY2, incX2);
        }

        protected internal static object UnaryOperationPerPixelVirtual<T1, T2>(string operation, string initialisation, string finalisation, Image<T1> img1, string type1, T2 val2, string type2, object inpValue)
            where T1 : struct
            where T2 : struct
        {
            // Retrieve a copy object
            string hashString = "ImageOperationPerPixelVirtual." + operation + "." + initialisation + "." + finalisation + "." + ".Image<" + type1 + ">." + type2;
            DynamicCode dCode = DynamicCode.Hash(hashString);
            if (dCode == null)
            {
                operation = Regex.Replace(operation, "A1", "data1Ptr");
                operation = Regex.Replace(operation, "V1", "(*data1Ptr)");
                operation = Regex.Replace(operation, "V2", "(val2)");
                string code = @"
                    " + initialisation + @";
                    fixed(T1* data1Base = data1)
                    {
                        T1* data1Ptr = data1Base + start1;
                        
                        for(int y=0;y<height;y++, data1Ptr+=incY1)
                            for(int x=0;x<width;x++, data1Ptr+=incX1)
                            {
                                " + operation + @";
                            }
                    }
                    " + finalisation + @";
                    ";
                dCode = new DynamicCode(new string[] { type1, type2 }, @"T1[,,] data1, T2 val2, int height, int width, int start1, int incY1, int incX1", code);

                DynamicCode.AddHash(hashString, dCode);
            }

            // Call the copy object
            int start1 = 0, incY1 = 0, incX1 = img1.Bands;
            if (img1 is VirtualImage<T1>)
            {
                VirtualImage<T1> img1VI = img1 as VirtualImage<T1>;
                start1 = ((img1VI.yOff * img1VI.original.Width) + img1VI.xOff) * img1VI.original.Bands + img1VI.bOff;
                incY1 = (img1VI.original.Width - img1VI.Width) * img1VI.original.Bands;
                incX1 = img1VI.original.Bands;
            }

            return dCode.Invoke(inpValue, img1.Data, val2, img1.Height, img1.Width, start1, incY1, incX1);
        }
    }
}