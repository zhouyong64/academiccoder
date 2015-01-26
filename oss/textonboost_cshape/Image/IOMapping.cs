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

using Misc;

namespace Image
{
    public delegate void IOMappingChangedDelegate(object sender);

    public interface IOMapping
    {
        event IOMappingChangedDelegate MappingChanged;

        Type GenericType { get; }

        Size Size(IImage image);
    }

    public interface IInputMapping : IOMapping
    {
    }

    public interface IOutputMapping : IOMapping
    {
    }

    // Class that maps from a byte image (directly derived from a Bitmap object) to an image of type T
    [Serializable]
    public abstract class InputMapping<T> : IInputMapping where T : struct
    {
        public Type GenericType
        {
            get { return typeof(T); }
        }

        #region Dimensions Accessors

        public Size Size(IImage image)
        {
            Image<byte> imageGeneric = (Image<byte>) image;
            return new Size(Width(imageGeneric), Height(imageGeneric));
        }

        public virtual int Height(Image<byte> input)
        {
            return input.Height;
        }

        public virtual int Width(Image<byte> input)
        {
            return input.Width;
        }

        public virtual int Bands(Image<byte> input)
        {
            if (input.Bands != 1 && input.Bands != 3)
                throw new ArgumentException("InputMapping requires either 1 or 3 colour bands");

            return input.Bands;
        }

        protected virtual bool CheckCompatible(Image<byte> input, Image<T> output)
        {
            return output.Height == Height(input) && output.Width == Width(input) && output.Bands == Bands(input);
        }

        #endregion

        // Perform the input mapping
        public abstract void Map(Image<byte> input, Image<T> output);

        #region Event Handling

        public event IOMappingChangedDelegate MappingChanged;

        protected void FireChangedEvent()
        {
            if (MappingChanged != null)
                MappingChanged(this);
        }

        #endregion
    }

    // Class that maps from a Image<T> to an Image<byte> (for outputting to a bitmap object)
    [Serializable]
    public abstract class OutputMapping<T> : IOutputMapping where T : struct
    {
        public Type GenericType
        {
            get { return typeof(T); }
        }

        #region Dimensions Accessors

        public Size Size(IImage image)
        {
            Image<T> imageGeneric = (Image<T>) image;
            return new Size(Width(imageGeneric.Width), Height(imageGeneric.Height));
        }

        public virtual int Height(int inputHeight)
        {
            return inputHeight;
        }

        public virtual int Width(int inputWidth)
        {
            return inputWidth;
        }

        public virtual int Bands(int inputBands)
        {
            if (inputBands != 1 && inputBands != 3)
                throw new ArgumentException("OutputMapping requires either 1 or 3 colour bands");

            return inputBands;
        }

        protected virtual bool CheckCompatible(Image<T> input, Image<byte> output)
        {
            return output.Height == Height(input.Height) && output.Width == Width(input.Width) && output.Bands == Bands(input.Bands);
        }

        #endregion

        // Perform the output mapping
        public abstract void Map(Image<T> input, Image<byte> output);

        #region Event Handling

        public event IOMappingChangedDelegate MappingChanged;

        protected void FireChangedEvent()
        {
            if (MappingChanged != null)
                MappingChanged(this);
        }

        #endregion
    }

    [Serializable]
    public class StandardOutputMapping<T> : OutputMapping<T> where T : struct
    {
        public override void  Map(Image<T> input, Image<byte> output)
        {
            if (!CheckCompatible(input, output))
 	            throw new ArgumentException("Input and output do not have compatible dimensions");

            // Perform the mapping
            Image<byte>.BinaryOperation("V1 = (T1) V2", "", "", output, input, null);
        }
    }

    [Serializable]
    public class StandardInputMapping<T> : InputMapping<T> where T : struct
    {
        public override void Map(Image<byte> input, Image<T> output)
        {
            if (!CheckCompatible(input, output))
                throw new ArgumentException("Input and output do not have compatible dimensions");

            // Perform the mapping
            Image<byte>.BinaryOperation("V1 = (T1) V2", "", "", output, input, null);
        }
    }

    [Serializable]
    public class IdOutputMapping<T> : OutputMapping<T> where T : struct
    {
        public static Color MapToColor(int id)
        {
            byte valR = 0, valG = 0, valB = 0;
            for (int j = 0; id > 0; j++, id >>= 3)
            {
                valR |= (byte)(((id >> 0) & 1) << (7 - j));
                valG |= (byte)(((id >> 1) & 1) << (7 - j));
                valB |= (byte)(((id >> 2) & 1) << (7 - j));
            }

            return Color.FromArgb(valR, valG, valB);
        }

        public override int Bands(int inputBands)
        {
            if (inputBands != 1)
                throw new ArgumentException("IdOutputMapping requires 1 colour band");

            return 3; // Maps 1 band to 3 bands
        }

        public override void Map(Image<T> input, Image<byte> output)
        {
            if (!CheckCompatible(input, output))
                throw new ArgumentException("Input and output do not have compatible dimensions");

            const string code =
                @"
                int id = (int) V2;
                byte valR = 0, valG = 0, valB = 0;
                for(int j=0; id>0; j++, id >>= 3)
                {
                    valR |= (byte) (((id >> 0) & 1) << (7 - j));
                    valG |= (byte) (((id >> 1) & 1) << (7 - j));
                    valB |= (byte) (((id >> 2) & 1) << (7 - j));
                }
                A1[0] = (T1) valR;
                A1[1] = (T1) valG;
                A1[2] = (T1) valB;
                ";

            Image<byte>.BinaryOperationPerPixel(code, "", "", output, input, null);
        }
    }

    [Serializable]
    public class IdInputMapping<T> : InputMapping<T> where T : struct
    {
        public override int Bands(Image<byte> input)
        {
            if (input.Bands != 3)
                throw new ArgumentException("IdInputMapping requires 3 colour bands");

            return 1; // Maps 3 bands to 1 band
        }

        public override void Map(Image<byte> input, Image<T> output)
        {
            if (!CheckCompatible(input, output))
                throw new ArgumentException("Input and output do not have compatible dimensions");

            const string code =
                @"
                int id = 0;
                byte valR= (byte) A2[0];
                byte valG = (byte) A2[1];
                byte valB = (byte) A2[2];
                
                for (int j = 0; j < 8; j++)
                    id = (id << 3) | (((valR >> j) & 1) << 0) | (((valG >> j) & 1) << 1) | (((valB >> j) & 1) << 2);
                V1 = (T1) id;
                ";

            Image<byte>.BinaryOperationPerPixel(code, "", "", output, input, null);
        }
    }

    [Serializable]
    public class NormalisingOutputMapping<T> : OutputMapping<T> where T : struct
    {
        bool initialised = false;
        double minValue;
        double maxValue;

        public NormalisingOutputMapping()
        {
        }

        public NormalisingOutputMapping(Image<T> image)
        {
            IncludeImage(image);
        }

        public void IncludeImage(Image<T> image)
        {
            const string code = @"
                double val = (double) V1;
                if (maxVal==null || val>maxVal)
                    maxVal = val;
                if (minVal==null || val<minVal)
                    minVal = val;
            ";
            double[] minMax = (double[]) Image<T>.UnaryOperation(code, "double? maxVal = null, minVal = null;", "RET = new double[] { (double) minVal, (double) maxVal };", image, null);

            if (!initialised)
            {
                minValue = minMax[0];
                maxValue = minMax[1];
                initialised = true;
            }
            else
            {
                minValue = Math.Min(minValue, minMax[0]);
                maxValue = Math.Max(maxValue, minMax[1]);
            }

            FireChangedEvent();
        }

        public override void Map(Image<T> input, Image<byte> output)
        {
            if (!CheckCompatible(input, output))
                throw new ArgumentException("Input and output do not have compatible dimensions");

            const string code = @"
                double val = (double) V2;
                if (maxValue == minValue)
                    V1 = (T1) 255.0;
                else
                    V1 = (T1) Math.Min(255.0, 256.0 * (val - minValue) / (maxValue - minValue));
            ";
            Image<byte>.BinaryOperation(code, "double[] minMax = (double[]) INP; double minValue = minMax[0], maxValue = minMax[1];", "", output, input, new double[] { minValue, maxValue });
        }
    }

    [Serializable]
    public class NegativeOutputMapping : OutputMapping<byte>
    {
        public NegativeOutputMapping()
        {
        }

        public override void Map(Image<byte> input, Image<byte> output)
        {
            if (!CheckCompatible(input, output))
                throw new ArgumentException("Input and output do not have compatible dimensions");

            const string code = @"
                double val = (double) V2;
                V1 = (T1) (255 - val);
            ";
            Image<byte>.BinaryOperation(code, "", "", output, input, null);
        }
    }

    [Serializable]
    public class ZoomOutputMapping<T> : OutputMapping<T> where T : struct
    {
        private double scaleFactor;
        public ZoomOutputMapping(double scaleFactor)
        {
            this.scaleFactor = scaleFactor;
        }

        public override int Height(int inputHeight)
        {
            return (int) (inputHeight * scaleFactor);
        }

        public override int Width(int inputWidth)
        {
            return (int) (inputWidth * scaleFactor);
        }

        public override void Map(Image<T> input, Image<byte> output)
        {
            if (!CheckCompatible(input, output))
                throw new ArgumentException("Input and output do not have compatible dimensions");

            string type = input.GenericType.ToString();
            string hashString = "ZoomOutputMapping." + type;

            DynamicCode dCode = DynamicCode.Hash(hashString);
            if (dCode == null)
            {
                string code = @"
                    foreach (Point p in output.Bounds)
                        for (int b = 0; b < output.Bands; b++)
                            output[p, b] = input[(int)(p.Y / scaleFactor), (int)(p.X / scaleFactor), b];
                ";
                dCode = new DynamicCode(new string[] { type }, @"Image<T1> input, Image<byte> output, double scaleFactor", code);

                DynamicCode.AddHash(hashString, dCode);
            }

            // Call the dynamic operation
            dCode.Invoke(null, input, output, scaleFactor);
        }
    }


    [Serializable]
    public class ChainOutputMapping<T> : OutputMapping<T> where T : struct
    {
        private OutputMapping<T> first;
        private OutputMapping<byte> second;

        public ChainOutputMapping(OutputMapping<T> first, OutputMapping<byte> second)
        {
            this.first = first;
            this.second = second;
        }

        public override int Height(int inputHeight)
        {
            return second.Height(first.Height(inputHeight));
        }

        public override int Width(int inputWidth)
        {
            return second.Width(first.Width(inputWidth));
        }

        public override void Map(Image<T> input, Image<byte> output)
        {
            if (!CheckCompatible(input, output))
                throw new ArgumentException("Input and output do not have compatible dimensions");

            Image<byte> temp = new Image<byte>(first.Height(input.Height), first.Width(input.Width), first.Bands(input.Bands));
            first.Map(input, temp);
            second.Map(temp, output);
        }
    }

}
