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
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Reflection;

using System.CodeDom.Compiler;
using Microsoft.CSharp;

namespace Misc
{
    public class DynamicCode
    {
        #region Code Hash Table
        private static readonly object lockObject = new object();

        private static readonly Dictionary<string, DynamicCode> dynamicCodeHash = new Dictionary<string, DynamicCode>();

        public static DynamicCode Hash(string hashString)
        {
            lock (lockObject)
            {
                if (!dynamicCodeHash.ContainsKey(hashString))
                    return null;
                return dynamicCodeHash[hashString];
            }
        }

        public static void AddHash(string hashString, DynamicCode code)
        {
            lock (lockObject)
            {
                if (!dynamicCodeHash.ContainsKey(hashString)) // To be thread safe - doesn't matter too much if two identical dynamic code objects are created
                    dynamicCodeHash.Add(hashString, code);
            }
        }
        #endregion

        private static CodeDomProvider compiler = null;
        private static CompilerParameters compilerParameters;

        private object invokerObject;

        public DynamicCode(string[] genericTypeNames, string arguments, string codeBody)
        {
            lock (lockObject)
            {
                // Create a compiler
                if (compiler == null)
                    compiler = CodeDomProvider.CreateProvider("CSharp");

                // Initialise compiler parameters
                compilerParameters = new CompilerParameters();
                compilerParameters.GenerateInMemory = true;
                compilerParameters.CompilerOptions += " /unsafe";
                compilerParameters.ReferencedAssemblies.Add("System.dll");
                compilerParameters.ReferencedAssemblies.Add("System.Drawing.dll");
                compilerParameters.ReferencedAssemblies.Add(Assembly.GetExecutingAssembly().Location);  // Add in MyLib2 dll (with full path)

                // Find and replace typeNames for T1, T2, etc in arguments and codeBody
                for (int i = 0; i < genericTypeNames.Length; i++)
                {
                    arguments = Regex.Replace(arguments, "T" + (i + 1), genericTypeNames[i]);
                    codeBody = Regex.Replace(codeBody, "T" + (i + 1), genericTypeNames[i]);
                }
                codeBody = Regex.Replace(codeBody, "RET", "retValue");
                codeBody = Regex.Replace(codeBody, "INP", "inpValue");

                // Generate a complete source code string
                string code = @"
using System;
using System.Collections.Generic;
using System.Drawing;

using Misc;
using Image;
using ImageProcessing;

namespace DynamicCodeNamespace
{
    public class DynamicCode
    {
        public unsafe object DynamicCodeFunction(" + arguments + @", object inpValue)
        {
            object retValue = null;
            " + codeBody + @"
            return retValue;
        }
    }
}
";

                CompilerResults compiled = compiler.CompileAssemblyFromSource(compilerParameters, code);

                if (compiled.Errors.HasErrors)
                {
                    string errString = "";
                    errString += compiled.Errors.Count + " Errors:";
                    for (int i = 0; i < compiled.Errors.Count; i++)
                        errString += "\r\nLine: " + compiled.Errors[i].Line + " - " + compiled.Errors[i].ErrorText;
                    throw new Exception(errString);
                }

                Assembly assembly = compiled.CompiledAssembly;

                invokerObject = assembly.CreateInstance("DynamicCodeNamespace.DynamicCode");
                if (invokerObject == null)
                    throw new Exception("Could not create copy object!");
            }
        }

        public object Invoke(object inpValue, params object[] arguments)
        {
            try
            {
                object[] args = new object[arguments.Length + 1];
                for (int i = 0; i < arguments.Length; i++)
                    args[i] = arguments[i];
                args[args.Length - 1] = inpValue;
                return invokerObject.GetType().InvokeMember("DynamicCodeFunction", BindingFlags.InvokeMethod, null, invokerObject, args);
            }
            catch (TargetInvocationException tie)
            {
                throw tie.InnerException;
            }
        }
    }

}
