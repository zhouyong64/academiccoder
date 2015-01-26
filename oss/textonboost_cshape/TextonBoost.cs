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
using System.Drawing;

using Algorithms;
using Image;
using Misc;

namespace TextonBoost
{
    public class TextonBoost : JointBoost.IBoost
    {
        [Serializable]
        public struct ShapeFilter
        {
            public JointBoost.WeakLearner wl; // The weak learner
            public Rectangle r; // The rectangle
            public int t; // The texton index
            public ClassList classList; // The class list - needed to map full class numbers to compact ones

            public override string ToString()
            {
                string result = "{\n";

                string kStr = "{ ";
                if (wl.k == null) kStr += "}";
                else
                    for (int i = 0; i < wl.k.Length; i++)
                        kStr += wl.k[i] + (i < wl.k.Length - 1 ? ", " : " }");

                string cString = "{ ";
                for (int i = 0; i < classList.NumCompact; i++)
                    if (wl.n.Contains(i))
                        cString += classList.GetName(classList.ToFull(i)) + " ";
                cString += "}";
                result += "wl = {error=" + wl.error + ", time=" + wl.timeTaken + ", a=" + wl.a + ", b=" + wl.b + ", k[]=" + kStr + ", theta=" + wl.theta + ", classes=" + cString + " }\n";
                result += "r = " + r + "\n";
                result += "t = " + t + "\n";

                return result + "}\n";
            }
        }

        [Serializable]
        public class ClassList
        {
            private string[] classNames;
            private int[] toCompact;
            private int[] toFull;

            public ClassList(string[] classNames)
            {
                this.classNames = classNames;

                // Generate toCompact
                toCompact = new int[classNames.Length];
                int cCompact = 0;
                for (int i = 0; i < classNames.Length; i++)
                    toCompact[i] = (classNames[i].StartsWith("-")) ? -1 : (cCompact++);

                // Generate toFull
                toFull = new int[cCompact];
                cCompact = 0;
                for (int i = 0; i < classNames.Length; i++)
                    if (toCompact[i] == cCompact)
                        toFull[cCompact++] = i;
            }

            public string GetName(int cFull)
            {
                if (!IsValid(cFull))
                    throw new ArgumentException("Class " + classNames[cFull] + " is not in use!");
                return classNames[cFull];
            }

            public int NumFull
            {
                get { return toCompact.Length; }
            }

            public int NumCompact
            {
                get { return toFull.Length; }
            }

            public bool IsValid(int cFull)
            {
                return toCompact[cFull] != -1;
            }

            public int ToCompact(int cFull)
            {
                return toCompact[cFull];
            }

            public int ToFull(int cCompact)
            {
                return toFull[cCompact];
            }
        }

        // Threshold parameters
        private int numThetas;
        private int thetaStart, thetaInc;

        // Shape parameters
        private int minShapeSize, maxShapeSize, numShapes;

        // Misc parameters
        private int subSample;
        private double acceptFraction;

        // Fields
        int numExamples;
        private int numTextons;
        private BoolArray[] allowExamples;
        private Rectangle[] shapes;
        private Image<int>[][] integralImages;
        private int[][] targets;
        private Size[] subSampledSizes;
        private Point[][] validPoints;

        private ClassList classList;
        private List<ImageFilenamePair> trainingDataFilenames;

        public void DoWork()
        {
            // Get list of training data
            trainingDataFilenames = (List<ImageFilenamePair>) MainClass.Load("Filenames.Training");

            // Get number of textons
            List<KMeans.Cluster> clusters = (List<KMeans.Cluster>) MainClass.Load("Clusters");
            numTextons = clusters.Count;

            // Get class list
            classList = new ClassList(MainClass.Classes);

            // Other parameters
            int numRounds = MainClass.NumRoundsBoosting;
            acceptFraction = MainClass.RandomizationFactor;
            subSample = MainClass.BoostingSubSample;

            numThetas = MainClass.NumberOfThresholds;
            thetaStart = MainClass.ThresholdStart;
            thetaInc = MainClass.ThresholdIncrement;

            numShapes = MainClass.NumberOfRectangles;
            minShapeSize = MainClass.MinimumRectangleSize;
            maxShapeSize = MainClass.MaximumRectangleSize;

            // Create the booster object
            JointBoost booster = new JointBoost(this);

            Console.WriteLine("Boosting from " + NumImages + " images...");

            // Iterate number of rounds
            for (int round = 0; round < numRounds; round++)
            {
                Console.WriteLine("  Boosting round " + round);

                ShapeFilter sf = new ShapeFilter();
                sf.wl = booster.PerformRound();
                sf.classList = classList; // NB stored sharing set uses *compact* class list
                int d = sf.wl.d;
                sf.t = d / numShapes;
                sf.r = SuperSample(shapes[d % numShapes], subSample);

                Console.WriteLine("    ShapeFilter = " + sf.ToString());

                MainClass.Save("ShapeFilter." + round, sf);
            }

        }

        unsafe internal static Image<int>[] CalculateIntegralImages(Image<int> textonMap, int numTextons, int subSample, Size subSampledSize)
        {
            Image<int>[] integralImages = new Image<int>[numTextons];

            for (int textonNum = 0; textonNum < numTextons; textonNum++)
            {
                integralImages[textonNum] = new Image<int>(subSampledSize);

                int ssW = subSampledSize.Width;
                int ssH = subSampledSize.Height;
                int W = textonMap.Width;
                int H = textonMap.Height;

                // Use pointer arithmetic for speed
                fixed (int* tFixed = textonMap.DataPointer, iFixed = integralImages[textonNum].DataPointer)
                {
                    int* tPtr = tFixed, iPtr = iFixed;

                    // First, down-sample
                    for (int y = 0; y < H; y++)
                        for (int x = 0; x < W; x++, tPtr++)
                            if (*tPtr == textonNum)
                                iPtr[(y / subSample) * ssW + (x / subSample)]++;

                    // Second, accumulate
                    for (int y = 0; y < ssH; y++)
                        for (int x = 0; x < ssW; x++, iPtr++)
                            (*iPtr) += ((x > 0) ? *(iPtr - 1) : 0)
                                + ((y > 0) ? *(iPtr - ssW) : 0)
                                - ((x > 0 && y > 0) ? *(iPtr - ssW - 1) : 0);
                }
            }

            return integralImages;
        }

        private void InitialiseShapes()
        {
            shapes = new Rectangle[numShapes];
            for (int s = 0; s < shapes.Length; s++)
            {
                shapes[s].Width = MainClass.Rnd.Next(minShapeSize, maxShapeSize + 1);
                shapes[s].Height = MainClass.Rnd.Next(minShapeSize, maxShapeSize + 1);
                shapes[s].X = MainClass.Rnd.Next(maxShapeSize + 1 - shapes[s].Width) - maxShapeSize / 2;
                shapes[s].Y = MainClass.Rnd.Next(maxShapeSize + 1 - shapes[s].Height) - maxShapeSize / 2;

                // Sub-sample shape
                shapes[s] = SubSample(shapes[s], subSample);
            }
        }

        public static Rectangle SubSample(Rectangle input, int subSample)
        {
            Rectangle result = new Rectangle();

            result.Width = input.Width / subSample;
            result.Height = input.Height / subSample;

            result.X = ((input.X + Math.Sign(input.X) * subSample / 2) / subSample); // NB Care with sign
            result.Y = ((input.Y + Math.Sign(input.Y) * subSample / 2) / subSample); // NB Care with sign

            return result;
        }

        public static Rectangle SuperSample(Rectangle input, int subSample)
        {
            Rectangle result = new Rectangle();

            result.Width = input.Width * subSample;
            result.Height = input.Height * subSample;

            result.X = input.X * subSample;
            result.Y = input.Y * subSample;

            return result;
        }

        #region IBoost Members

        public int N
        {
            get { return numExamples; }
        }

        public int D
        {
            get { return numTextons * numShapes; }
        }

        public int C
        {
            get { return classList.NumCompact; }
        }

        public int ThetaStart
        {
            get { return thetaStart; }
        }

        public int ThetaInc
        {
            get { return thetaInc; }
        }

        public int NumThetas
        {
            get { return numThetas; }
        }

        public double AcceptFraction
        {
            get { return acceptFraction; }
        }

        public int NumImages
        {
            get
            {
                return integralImages.Length;
            }
        }

        public void Initialise()
        {
            // Initialise the shapes
            InitialiseShapes();

            // Which training images to boost off?
            List<int> imageList = new List<int>();
            for (int i = 0; i < trainingDataFilenames.Count; i++)
                imageList.Add(i);

            // Allocate memory
            integralImages = new Image<int>[imageList.Count][];
            targets = new int[imageList.Count][];
            validPoints = new Point[imageList.Count][];
            subSampledSizes = new Size[imageList.Count];

            numExamples = 0;

            for (int i = 0; i < imageList.Count; i++)
            {
                Console.WriteLine("Loading training image " + imageList[i]);

                // Load images
                Image<int> textonMap = (Image<int>) MainClass.Load("TextonMaps." + trainingDataFilenames[imageList[i]].im);
                Image<int> groundTruth = ImageIO.LoadImage<int>(trainingDataFilenames[imageList[i]].gt, new IdInputMapping<int>());

                // Calculate sub sampled image sizes
                subSampledSizes[i].Width = (textonMap.Width + subSample - 1) / subSample; // Care with off by one errors
                subSampledSizes[i].Height = (textonMap.Height + subSample - 1) / subSample; // Care with off by one errors

                // Calculate integral images
                integralImages[i] = CalculateIntegralImages(textonMap, numTextons, subSample, subSampledSizes[i]);

                // Ignore unused class labels
                List<Point> tempList = new List<Point>();
                foreach (Point p in new Bounds(subSampledSizes[i]))
                {
                    int gtLabel = groundTruth[p.Y * subSample, p.X * subSample];
                    if (gtLabel > 0 && classList.IsValid(gtLabel))
                        tempList.Add(p);
                }
                validPoints[i] = tempList.ToArray();
                numExamples += validPoints[i].Length;

                // Calculate target values
                int n = 0;
                targets[i] = new int[validPoints[i].Length];
                foreach (Point p in validPoints[i])
                {
                    int classFull = groundTruth[p.Y * subSample, p.X * subSample];
                    targets[i][n++] = classList.ToCompact(classFull);
                }
            }

            // Precompute allow examples
            Console.WriteLine("Calculating allow examples...");
            Rectangle maxRect = SubSample(new Rectangle(-(maxShapeSize + 1) / 2, -(maxShapeSize + 1) / 2, maxShapeSize + 1, maxShapeSize + 1), subSample);
            allowExamples = new BoolArray[numTextons];
            for (int t = 0; t < numTextons; t++)
            {
                allowExamples[t] = new BoolArray(N);

                int n = 0;
                for (int i = 0; i < NumImages; i++)
                {
                    Image<int> integralImage = integralImages[i][t];

                    foreach (Point p in validPoints[i])
                        allowExamples[t][n++] = CalculateShapeFilterResponse(integralImage, maxRect, p) > 0;
                }
            }
        }

        public void GetTargets(int[] targets)
        {
            int n = 0;

            for (int i = 0; i < NumImages; i++)
            {
                for (int j = 0; j < this.targets[i].Length; j++)
                    targets[n++] = this.targets[i][j];
            }
        }

        public void CalculateFeatureValues(int[] values, int d)
        {
            int t = d / numShapes;
            Rectangle shape = shapes[d % numShapes];

            int n = 0;

            for (int i = 0; i < NumImages; i++)
            {
                Image<int> integralImage = integralImages[i][t];
                foreach (Point p in validPoints[i])
                {
                    if (!allowExamples[t][n])
                        values[n] = 0;
                    else
                        values[n] = CalculateShapeFilterResponse(integralImage, shape, p);

                    n++;
                }
            }
        }

        // Calculate summed response from integral image
        internal static int CalculateShapeFilterResponse(Image<int> integralImage, Rectangle shape, Point p)
        {
            Size size = integralImage.Size;

            int x1 = p.X + shape.X;
            int y1 = p.Y + shape.Y;
            int x2 = x1 + shape.Width;
            int y2 = y1 + shape.Height;

            // Check if rectangle is at least partially inside image bounds
            if (x1 >= size.Width || y1 >= size.Height || x2 <= 0 || y2 <= 0)
                return 0;

            int maxX = Math.Min(size.Width - 1, x2 - 1);
            int maxY = Math.Min(size.Height - 1, y2 - 1);

            int m = integralImage[maxY, maxX];
            int mx = x1 > 0 ? integralImage[maxY, x1 - 1] : 0;
            int my = y1 > 0 ? integralImage[y1 - 1, maxX] : 0;
            int mxy = x1 > 0 && y1 > 0 ? integralImage[y1 - 1, x1 - 1] : 0;

            return m - mx - my + mxy;
        }

        #endregion
    }
}
