// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html




using System;




// namespace



// Activation functions.


public static class Activation
{

internal static float sigmoid( float z )
{
// z is called the logit.  LOH-jit
// The logit is the inverse of the standard
// Logistic Function.  The Logistic Function
// is for things like population growth.
// Like with a limited food supply.

// 1.0 / ( 1.0 + exp( -z ))
// Derivative:
// (1.0 + exp( -z ))^-2  *  (e^-z)

float a = (float)(1.0 /
               ( 1.0 + MathF.exp( -z )));

return a;
}




internal static float derivSigmoid( float z )
{
float sig = sigmoid( z );
float result = sig * (1.0F - sig);

return result;
}




internal static float reLU( float z )
{
// ReLU f(x) = max( 0, x )

if( z > 0 )
  return z;

return 0;
}



internal static float derivReLU( float z )
{
if( z > 0 )
  return 1;

return 0;
}


} // Class
