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

namespace Misc
{
    [Serializable]
    public struct BoundsF
    {
        private bool isAssigned;
        private float minX, minY, maxX, maxY;

        public PointF GetCentre()
        {
            return new PointF((maxX + minX) / 2f, (maxY + minY) / 2f);
        }

        public float MinX
        {
            get { return minX; }
        }

        public float MinY
        {
            get { return minY; }
        }

        public float MaxX
        {
            get { return maxX; }
        }

        public float MaxY
        {
            get { return maxY; }
        }

        public float Width
        {
            get { return maxX - minX + 1; }
        }

        public float Height
        {
            get { return maxY - minY + 1; }
        }

        // INCLUSIVE COORDS
        public BoundsF(float minX, float minY, float maxX, float maxY)
        {
            this.minX = minX;
            this.minY = minY;
            this.maxX = maxX;
            this.maxY = maxY;
            this.isAssigned = true;
        }

        public bool Contains(PointF p)
        {
            return minX <= p.X && p.X <= maxX && minY <= p.Y && p.Y <= maxY;
        }

        public BoundsF EnlargeInclude(PointF p)
        {
            if (!isAssigned)
                return new BoundsF(p.X, p.Y, p.X, p.Y);
            else
                return new BoundsF(Math.Min(minX, p.X), Math.Min(minY, p.Y), Math.Max(maxX, p.X), Math.Max(maxY, p.Y));
        }

        public override string ToString()
        {
            return "[ (" + minX + ", " + minY + "), (" + maxX + ", " + maxY + ")]";
        }
    }

    [Serializable]
    public struct Bounds : IEnumerable<Point>
    {
        private bool nonEmpty;
        private int minX, minY, maxX, maxY;

        public PointF GetCentre()
        {
            return new PointF((maxX + minX) / 2f, (maxY + minY) / 2f);
        }

        public Point[] GetCorners()
        {
            return new Point[] { new Point(minX, minY), new Point(minX, maxY), new Point(maxX, maxY), new Point(maxX, minY) };
        }

        public Bounds IntersectWith(Bounds with)
        {
            Bounds ret = new Bounds();
            if (!this.nonEmpty || !with.nonEmpty)
                ret.nonEmpty = false;
            else
            {
                ret.minX = Math.Max(this.MinX, with.MinX);
                ret.minY = Math.Max(this.MinY, with.MinY);
                ret.maxX = Math.Min(this.MaxX, with.MaxX);
                ret.maxY = Math.Min(this.MaxY, with.MaxY);
                if (ret.MaxX < ret.MinX || ret.MaxY < ret.MinY)
                {
                    ret.nonEmpty = false;
                }
                else
                    ret.nonEmpty = true;
            }
            return ret;
        }

        public Point MinPoint
        {
            get { return new Point(MinX, MinY); }
        }

        public Point MaxPoint
        {
            get { return new Point(MaxX, MaxY); }
        }

        public int MinX
        {
            get { return minX; }
        }

        public int MinY
        {
            get { return minY; }
        }

        public int MaxX
        {
            get { return maxX; }
        }

        public int MaxY
        {
            get { return maxY; }
        }

        public int Width
        {
            get { return Math.Max(0, maxX - minX + 1); }
        }

        public int Height
        {
            get { return Math.Max(0, maxY - minY + 1); }
        }

        public Size Size
        {
            get { return new Size(Width, Height); }
        }

        private bool CheckNonEmpty()
        {
            return Width > 0 && Height > 0;
        }

        public Bounds ShrinkBy(int size)
        {
            return new Bounds(minX + size, minY + size, maxX - size, maxY - size);
        }

        public Bounds OffsetBy(Point p)
        {
            return new Bounds(minX + p.X, minY + p.Y, maxX + p.X, maxY + p.Y);
        }

        public Bounds OffsetBy(PointF p)
        {
            return new Bounds(minX + (int) p.X, minY + (int) p.Y, maxX + (int) Math.Ceiling(p.X), maxY + (int) Math.Ceiling(p.Y));
        }

        // INCLUSIVE COORDS
        public Bounds(int minX, int minY, int maxX, int maxY)
        {
            this.minX = minX;
            this.minY = minY;
            this.maxX = maxX;
            this.maxY = maxY;
            this.nonEmpty = true;
            this.nonEmpty = CheckNonEmpty();
        }

        public Bounds(Size s)
        {
            this.minX = 0;
            this.minY = 0;
            this.maxX = s.Width - 1;
            this.maxY = s.Height - 1;
            this.nonEmpty = true;
            this.nonEmpty = CheckNonEmpty();
        }

        public Bounds(int width, int height)
        {
            this.minX = 0;
            this.minY = 0;
            this.maxX = width - 1;
            this.maxY = height - 1;
            this.nonEmpty = true;
            this.nonEmpty = CheckNonEmpty();
        }

        public Bounds(Rectangle r)
        {
            this.minX = r.X;
            this.minY = r.Y;
            this.maxX = r.X + r.Width - 1;
            this.maxY = r.Y + r.Height - 1;
            this.nonEmpty = true;
            this.nonEmpty = CheckNonEmpty();
        }

        public bool Contains(Point p)
        {
            if (nonEmpty)
                return minX <= p.X && p.X <= maxX && minY <= p.Y && p.Y <= maxY;
            else
                return false;
        }

        public bool Contains(PointF pF)
        {
            if (nonEmpty)
                return minX <= pF.X && pF.X <= maxX && minY <= pF.Y && pF.Y <= maxY;
            else
                return false;
        }

        public Bounds EnlargeInclude(Point p)
        {
            return EnlargeInclude(p.X, p.Y, p.X, p.Y);
        }

        public Bounds EnlargeInclude(PointF p)
        {
            return EnlargeInclude((int) Math.Floor(p.X), (int) Math.Floor(p.Y), (int) Math.Ceiling(p.X), (int) Math.Ceiling(p.Y));
        }

        private Bounds EnlargeInclude(int lX, int lY, int hX, int hY)
        {
            if (!nonEmpty)
                return new Bounds(lX, lY, hX, hY);
            else
                return new Bounds(Math.Min(minX, lX), Math.Min(minY, lY), Math.Max(maxX, hX), Math.Max(maxY, hY));
        }

        public override string ToString()
        {
            if (nonEmpty)
                return "[ (" + minX + ", " + minY + "), (" + maxX + ", " + maxY + ")]";
            else
                return "[ Empty Bounds ]";
        }

        public System.Collections.IEnumerator GetEnumerator()
        {
            return new BoundsEnumerator(this);
        }

        IEnumerator<Point> IEnumerable<Point>.GetEnumerator()
        {
            return new BoundsEnumerator(this);
        }

        private class BoundsEnumerator : IEnumerator<Point>
        {
            public void Dispose() { }

            private Bounds bounds;
            private bool reset = true;
            private Point p = new Point();

            public BoundsEnumerator(Bounds b)
            {
                this.bounds = b;
                Reset();
            }

            #region IEnumerator Members

            public void Reset()
            {
                reset = true;

            }

            public object Current
            {
                get { return p; }
            }

            Point IEnumerator<Point>.Current
            {
                get { return p; }
            }

            public bool MoveNext()
            {
                if (!bounds.nonEmpty)
                    return false;
                else if (reset)
                {
                    reset = false;
                    p.X = bounds.MinX;
                    p.Y = bounds.MinY;
                    return true;
                }
                else
                {

                    p.X++;
                    if (p.X > bounds.MaxX)
                    {
                        p.X = bounds.MinX;
                        p.Y++;
                    }
                    return p.Y <= bounds.MaxY;
                }
            }

            #endregion

        }
    }

    [Serializable]
    public struct CircleBounds : IEnumerable<Point>
    {
        private bool nonEmpty;
        private float cX, cY, radius;

        public PointF GetCentre()
        {
            return new PointF(cX, cY);
        }

        public float CentreX
        {
            get { return cX; }
        }

        public float CentreY
        {
            get { return cY; }
        }

        public float Radius
        {
            get { return radius; }
        }


        private bool CheckNonEmpty()
        {
            return radius >= 0;
        }

        public CircleBounds(float cX, float cY, float radius)
        {
            this.cX = cX;
            this.cY = cY;
            this.radius = radius;

            this.nonEmpty = true;
            this.nonEmpty = CheckNonEmpty();
        }

        public bool Contains(Point p)
        {
            if (nonEmpty)
                return (p.Y - cY) * (p.Y - cY) + (p.X - cX) * (p.X - cX) <= radius * radius;
            else
                return false;
        }

        public bool Contains(PointF pF)
        {
            if (nonEmpty)
                return (pF.Y - cY) * (pF.Y - cY) + (pF.X - cX) * (pF.X - cX) <= radius * radius;
            else
                return false;
        }

        public override string ToString()
        {
            if (nonEmpty)
                return "[ (" + cX + ", " + cY + "), radius " + radius + "]";
            else
                return "[ Empty CircleBounds (negative radius) ]";
        }

        public System.Collections.IEnumerator GetEnumerator()
        {
            return new CircleBoundsEnumerator(this);
        }

        IEnumerator<Point> IEnumerable<Point>.GetEnumerator()
        {
            return new CircleBoundsEnumerator(this);
        }

        private class CircleBoundsEnumerator : IEnumerator<Point>
        {
            public void Dispose() { }
            private CircleBounds bounds;
            private bool reset = true;
            private Point p = new Point();

            public CircleBoundsEnumerator(CircleBounds b)
            {
                this.bounds = b;
                Reset();
            }

            #region IEnumerator Members

            public void Reset()
            {
                reset = true;
            }

            public object Current
            {
                get { return p; }
            }

            Point IEnumerator<Point>.Current
            {
                get { return p; }
            }

            private bool FindInternalPoint()
            {
                while (true)
                {
                    if (bounds.Contains(p))
                        return true;
                    else
                        p.X++;
                    if (p.X > (int) Math.Ceiling(bounds.cX + bounds.radius))
                    {
                        p.Y++;
                        p.X = (int) (bounds.cX - bounds.radius);
                    }
                    if (p.Y > (int) Math.Ceiling(bounds.cY + bounds.radius))
                        return false;
                }
            }

            public bool MoveNext()
            {
                if (!bounds.nonEmpty)
                    return false;
                else if (reset)
                {
                    reset = false;
                    p.X = (int) (bounds.cX - bounds.radius);
                    p.Y = (int) (bounds.cY - bounds.radius);
                    return FindInternalPoint();
                }
                else
                {
                    p.X++;
                    return FindInternalPoint();
                }
            }

            #endregion

        }
    }
}
