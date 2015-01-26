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
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Runtime.Serialization;
using System.Text;

using Misc;


namespace Image
{
    public sealed class ImageIO
    {
        public static Bitmap ConvertToBitmap<T1>(Image<T1> image) where T1 : struct
        {
            return ConvertToBitmap<T1>(image, new StandardOutputMapping<T1>());
        }

        public static Bitmap ConvertToBitmap<T1>(Image<T1> image, OutputMapping<T1> outputMapping) where T1 : struct
        {
            // Perform output mapping
            Image<byte> byteImage = new Image<byte>(outputMapping.Height(image.Height), outputMapping.Width(image.Width), outputMapping.Bands(image.Bands));
            outputMapping.Map(image, byteImage);

            // Choose appropriate PixelFormat
            PixelFormat format;
            if (byteImage.Bands == 1)
                format = PixelFormat.Format8bppIndexed;
            else if (byteImage.Bands == 3)
                format = PixelFormat.Format32bppRgb;
            else
                throw new Exception("Cannot convert images with Bands!=1 || Bands!=3 to Bitmaps");

            // Create the bitmap
            Bitmap bitmap = new Bitmap(byteImage.Width, byteImage.Height, format);

            // Add a palette for monochrome images
            if (format == PixelFormat.Format8bppIndexed)
            {
                ColorPalette p = bitmap.Palette;
                for (int i = 0; i < 255; i++)
                    p.Entries[i] = Color.FromArgb(i, i, i);
                bitmap.Palette = p;
            }   

            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.WriteOnly, format);

            switch (bitmap.PixelFormat)
            {
                case PixelFormat.Format8bppIndexed:
                    CopyTo8bppBitmapData(byteImage, bitmapData);
                    break;

                case PixelFormat.Format32bppRgb:
                    CopyTo32bppBitmapData(byteImage, bitmapData);
                    break;
            }

            bitmap.UnlockBits(bitmapData);

            return bitmap;
        }

        public static void SaveImage<T1>(string filename, Image<T1> image) where T1 : struct
        {
            if (Path.GetExtension(filename) == ".data")
                SaveImageSerialized<T1>(filename, image);
            else
                SaveImage<T1>(filename, image, new StandardOutputMapping<T1>());
        }

        public static void SaveImage<T1>(string filename, Image<T1> image, OutputMapping<T1> outputMapping) where T1 : struct
        {
            // Create bitmap object
            Bitmap bitmap = ConvertToBitmap(image, outputMapping);

            // Write bitmap to file
            SaveBitmap(filename, bitmap);

            bitmap.Dispose();
        }

        public static Image<T1> LoadImage<T1>(string filename) where T1 : struct
        {
            if (Path.GetExtension(filename) == ".data")
                return LoadImageSerialized<T1>(filename);
            else
                return LoadImage<T1>(filename, new StandardInputMapping<T1>());
        }

        public static Image<T1> LoadImage<T1>(string filename, InputMapping<T1> inputMapping) where T1 : struct
        {
            // Load bitmap from file
            Bitmap bitmap = LoadBitmap(filename);

            Image<T1> image = ConvertFromBitmap<T1>(bitmap, inputMapping);

            bitmap.Dispose();

            return image;
        }

        public static Image<T1> ConvertFromBitmap<T1>(Bitmap bitmap) where T1 : struct
        {
            return ConvertFromBitmap<T1>(bitmap, new StandardInputMapping<T1>());
        }

        public static Image<T1> ConvertFromBitmap<T1>(Bitmap bitmap, InputMapping<T1> inputMapping) where T1 : struct
        {
            // Get dimensions
            int bands = bitmap.PixelFormat == PixelFormat.Format1bppIndexed || bitmap.PixelFormat == PixelFormat.Format8bppIndexed ? 1 : 3;

            // Create a temporary byte image to store incoming data
            Image<byte> byteImage = new Image<byte>(bitmap.Height, bitmap.Width, bands);

            // Treat differently according to format
            PixelFormat format = bands == 1 ? bitmap.PixelFormat : PixelFormat.Format32bppRgb;
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, format);

            switch (format)
            {
                case PixelFormat.Format1bppIndexed:
                    CopyFrom1bppBitmapData(byteImage, bitmapData);
                    break;

                case PixelFormat.Format8bppIndexed:
                    CopyFrom8bppBitmapData(byteImage, bitmapData);
                    break;

                case PixelFormat.Format32bppRgb:
                    CopyFrom32bppBitmapData(byteImage, bitmapData);
                    break;
            }

            bitmap.UnlockBits(bitmapData);

            // Create image object and map data across
            Image<T1> image = new Image<T1>(inputMapping.Height(byteImage), inputMapping.Width(byteImage), inputMapping.Bands(byteImage));
            inputMapping.Map(byteImage, image);

            return image;
        }

        private static unsafe void CopyFrom1bppBitmapData(Image<byte> image, BitmapData bitmapData)
        {
            byte* dataPtr = (byte*) bitmapData.Scan0.ToPointer();

            for (int y = 0; y < image.Height; y++)
            {
                int bit = 7;
                int offset = 0;
                byte grey = 0;
                for (int x = 0; x < image.Width; x++)
                {
                    if (bit == 7)
                        grey = dataPtr[offset];

                    byte greyVal = (byte) (((grey >> bit) & 1) == 1 ? 255 : 0);
                    image[y, x] = greyVal;

                    bit--;
                    if (bit == -1)
                    {
                        bit = 7;
                        offset++;
                    }
                }
                dataPtr += bitmapData.Stride;
            }                
        }

        private static unsafe void CopyFrom8bppBitmapData(Image<byte> image, BitmapData bitmapData)
        {
            byte* dataPtr = (byte*) bitmapData.Scan0.ToPointer();
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    image[y, x] = dataPtr[x];
                }
                dataPtr += bitmapData.Stride;
            }
        }

        private static unsafe void CopyFrom32bppBitmapData(Image<byte> image, BitmapData bitmapData)
        {
            int* dataPtr = (int*) bitmapData.Scan0.ToPointer();
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    int colour = dataPtr[x];

                    image[y, x, 0] = (byte) ((colour >> 16) & 0xff);   // r
                    image[y, x, 1] = (byte) ((colour >> 8) & 0xff);     // g
                    image[y, x, 2] = (byte) (colour & 0xff);                // b
                }
                dataPtr += bitmapData.Stride / 4;
            }
        }

        private static unsafe void CopyTo8bppBitmapData(Image<byte> image, BitmapData bitmapData)
        {
            byte* dataPtr = (byte*) bitmapData.Scan0.ToPointer();
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    dataPtr[x] = image[y, x];
                }
                dataPtr += bitmapData.Stride;
            }
        }

        private static unsafe void CopyTo32bppBitmapData(Image<byte> image, BitmapData bitmapData)
        {
            int* dataPtr = (int*) bitmapData.Scan0.ToPointer();
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    int r = image[y, x, 0];
                    int g = image[y, x, 1];
                    int b = image[y, x, 2];
                    dataPtr[x] = (r << 16) + (g << 8) + b;
                }
                dataPtr += bitmapData.Stride / 4;
            }
        }

        internal static Bitmap LoadBitmap(string filename)
        {
            return (Bitmap) Bitmap.FromFile(filename);
            //return new Bitmap(filename);
        }

        internal static void SaveBitmap(string filename, Bitmap bitmap)
        {
            ImageFormat format;
            if (Path.GetExtension(filename).ToLower() == ".bmp")
                format = ImageFormat.Bmp;
            else
                format = ImageFormat.Png;
            bitmap.Save(filename,format);
        }

        private static Image<T1> LoadImageSerialized<T1>(string filename) where T1 : struct
        {
            using (Stream inStream = new FileStream(filename, FileMode.Open))
            {
                BinaryFormatter fmt = new BinaryFormatter();

                object obj = fmt.Deserialize(inStream);
                inStream.Close();

                return (Image<T1>) obj;
            }
        }

        private static void SaveImageSerialized<T1>(string filename, Image<T1> image) where T1 : struct
        {
            using (FileStream outStream = new FileStream(filename, FileMode.Create))
            {
                BinaryFormatter fmt = new BinaryFormatter(null, new StreamingContext(StreamingContextStates.File));
                fmt.Serialize(outStream, image);
                outStream.Close();
            }
        }
    }
}
