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
using System.IO;
using System.Runtime.Serialization;
using System.Text;

using Misc;

namespace Image
{
    public interface IImage : ISerializable
    {
        int Width { get; }
        int Height { get; }
        Size Size { get; }
        string ToString(Point p);
        string ToString(int y, int x);
        Bitmap ConvertToBitmap(IOutputMapping outputMapping);
        void SaveData(string filename);
        Type GenericType { get; }
    }

    [Serializable]
    public partial class Image<T> : IImage where T : struct
    {
        #region Private data

        // Stores the data
        protected T[, ,] data;

        protected virtual internal T[, ,] Data { get { return data; } }

        // Dimensions
        private int height, width, bands;

        #endregion

        #region IO
        public virtual void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            //info.AddValue("ZippedData", ZipUtil.ZipObject(data)); // About 2 times slower
            info.AddValue("Data", data);
            info.AddValue("Height", height);
            info.AddValue("Width", width);
            info.AddValue("Bands", bands);
        }

        public Image(SerializationInfo info, StreamingContext context)
        {
            //data = (T[, ,])ZipUtil.UnZipObject((byte[])info.GetValue("ZippedData", typeof(byte[])));  // About 2 times slower
            data = (T[, ,])info.GetValue("Data", typeof(T[, ,]));
            height = info.GetInt32("Height");
            width = info.GetInt32("Width");
            bands = info.GetInt32("Bands");
        }

        public void SaveData(string filename)
        {
            StreamWriter s = new StreamWriter(filename);

            s.WriteLine("{0}\t{1}\t{2}", this.Height, this.Width, this.Bands);

            for (int y = 0; y < Height; y++)
                for (int x = 0; x < Width; x++)
                    for (int b = 0; b < Bands; b++)
                        s.Write("{0}\t", this[y, x, b].ToString());

            s.Close();
        }
        #endregion

        #region Constructors

        private void InitialiseData()
        {
            data = new T[height, width, bands];
        }

        // Standard constructors
        public Image(Size size) : this(size, 1) { }
        public Image(Size size, int bands) : this(size.Height, size.Width, bands) { }
        public Image(int height, int width) : this(height, width, 1) { }
        public Image(int height, int width, int bands) : this(height, width, bands, true) { }
        protected Image(int height, int width, int bands, bool allocateMemory)
        {
            if (height < 0 || width < 0 || bands < 0)
                throw new ArgumentException("Image requries positive extent");

            this.height = height;
            this.width = width;
            this.bands = bands;

            if (allocateMemory)
                InitialiseData();
        }

        public Image<T> CopyImage()
        {
            return CopyImage(this);
        }

        // Copy constructor
        public static Image<T> CopyImage<T2>(Image<T2> from) where T2 : struct
        {
            // Create return object
            Image<T> to = new Image<T>(from.Height, from.Width, from.Bands);

            to.OverwriteImage(from);

            return to;
        }

        // Overwrite whole image with another whole image
        public void OverwriteImage<T2>(Image<T2> from) where T2 : struct
        {
            BinaryOperation("V1 = (T1) V2;", "", "", this, from, null);
        }

        #endregion

        #region Manipulation

        public bool DimensionsEqual<T2>(Image<T2> img2) where T2 : struct
        {
            return Height == img2.Height && Width == img2.Width && Bands == img2.Bands;
        }

        public void Fill(T val)
        {
            UnaryOperation("V1 = V2;", "", "", this, val, null);
        }

        public T GetSum()
        {
            return (T) UnaryOperation("sum += V1;", "T1 sum = 0;", "RET = sum", this, null);
        }

        public T[] GetMinMaxValues()
        {
            return (T[]) UnaryOperation("if (maxVal==null || V1>maxVal) maxVal = V1; if (minVal==null || V1<minVal) minVal = V1", "T1? maxVal = null, minVal = null;", "RET = new T1[] { (T1) minVal, (T1) maxVal };", this, null);
        }

        #endregion

        #region Accessors

        unsafe public T[,,] DataPointer
        {
            get { return data; }
        }

        public virtual T this[Point p]
        {
            get { return this[p.Y, p.X]; }
            set { this[p.Y, p.X] = value; }
        }

        public virtual T this[Point p, int band]
        {
            get { return this[p.Y, p.X, band]; }
            set { this[p.Y, p.X, band] = value; }
        }

        public virtual T this[int y, int x]
        {
            get
            {
                if (bands != 1) throw new Exception("Must specify which band to access.");
                return data[y, x, 0];
            }
            set
            {
                if (bands != 1) throw new Exception("Must specify which band to access.");
                data[y, x, 0] = value;
            }
        }

        public virtual T this[int y, int x, int band]
        {
            get { return data[y, x, band]; }
            set { data[y, x, band] = value; }
        }

        public string ToString(Point p)
        {
            return ToString(p.Y, p.X);
        }

        public string ToString(int y, int x)
        {
            string ans = "[";
            for (int b = 0; b < Bands - 1; b++)
                ans += this[y, x, b] + ", ";
            ans += this[y, x, Bands - 1] + "]";
            return ans;
        }

        public int Height
        {
            get { return height; }
        }

        public int Width
        {
            get { return width; }
        }

        public int Bands
        {
            get { return bands; }
        }

        public Size Size
        {
            get { return new Size(width, height); }
        }

        public Bounds Bounds
        {
            get { return new Bounds(Width, Height); }
        }

        #endregion
    }

    [Serializable]
    public partial class VirtualImage<T> : Image<T> where T : struct
    {
        #region Private Data

        internal Image<T> original;
        internal T[, ,] originalData;

        // Offsets relative to original image
        internal int yOff, height;
        internal int xOff, width;
        internal int bOff, bands;

        protected override internal T[, ,] Data { get { return originalData; } }

        #endregion

        #region IO
        public override void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            base.GetObjectData(info, context);

            info.AddValue("Original", original);
            info.AddValue("yOff", yOff);
            info.AddValue("height", height);
            info.AddValue("xOff", xOff);
            info.AddValue("width", width);
            info.AddValue("bOff", bOff);
            info.AddValue("bands", bands);
        }

        public VirtualImage(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
            Image<T> original = (Image<T>) info.GetValue("Original", typeof(Image<T>));
            int yOff = info.GetInt32("yOff");
            int height = info.GetInt32("height");
            int xOff = info.GetInt32("xOff");
            int width = info.GetInt32("width");
            int bOff = info.GetInt32("bOff");
            int bands = info.GetInt32("bands");

            Initialise(original, yOff, xOff, bOff, height, width, bands);
        }

        #endregion

        #region Constructors

        public VirtualImage(Image<T> original) : this(original, 0, original.Bands) { }
        public VirtualImage(Image<T> original, int band) : this(original, band, 1) {}
        public VirtualImage(Image<T> original, int band, int bands) : this(original, 0, 0, band, original.Height, original.Width, bands) { }
        public VirtualImage(Image<T> original, Bounds bounds) : this(original, 0, original.Bands) { }
        public VirtualImage(Image<T> original, Bounds bounds, int band) : this(original, bounds, band, 1) {}
        public VirtualImage(Image<T> original, Bounds bounds, int band, int bands) : this(original, bounds.MinY, bounds.MinX, band, bounds.Height, bounds.Width, bands) {}
        public VirtualImage(Image<T> original, int y, int x, int height, int width) : this(original, y, x, 0, height, width, original.Bands) { }
        public VirtualImage(Image<T> original, int y, int x, int band, int height, int width) : this(original, y, x, band, height, width, 1) { }
        public VirtualImage(Image<T> original, int y, int x, int band, int height, int width, int bands) : base(height, width, bands, false)
        {
            if (height<=0 || width<=0 || bands<=0)
                throw new ArgumentException("VirtualImage requires positive extent");
            if (y<0 || (y+height)>original.Height || x<0 || (x+width)>original.Width || band<0 || (band+bands)>original.Bands)
                throw new ArgumentException("VirtualImage bounds must lie entirely within original image bounds");

            Initialise(original, y, x, band, height, width, bands);
        }

        private void Initialise(Image<T> original, int y, int x, int band, int height, int width, int bands)
        {
            this.original = original;
            this.originalData = original.Data;

            this.yOff = y; this.height = height;
            this.xOff = x; this.width = width;
            this.bOff = band; this.bands = bands;
        }

        #endregion

        #region Accessors

        public override T this[int y, int x]
        {
            get
            {
                if (Bands != 1) throw new Exception("Must specify which band to access.");
                return this[y,x,0];
            }
            set
            {
                if (Bands != 1) throw new Exception("Must specify which band to access.");
                this[y,x,0] = value;
            }
        }

        public override T this[int y, int x, int band]
        {
            get
            {
                if (y < 0 || y >= Height || x < 0 || x >= Width || band < 0 || band >= Bands)
                    throw new ArgumentException("Attempt to index data outside VirtualImage bounds");

                return originalData[y + yOff, x + xOff, band + bOff];
            }
            set
            {
                if (y < 0 || y >= Height || x < 0 || x >= Width || band < 0 || band >= Bands)
                    throw new ArgumentException("Attempt to index data outside VirtualImage bounds");

                originalData[y + yOff, x + xOff, band + bOff] = value;
            }
        }


        #endregion
    }
}
