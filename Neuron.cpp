// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html




#include "Neuron.h"
#include "../Math/MathC.h"
#include "../CppBase/StIO.h"
#include "../CryptoBase/Randomish.h"



Float32 Neuron::sigmoid( Float64 z )
{
// z is called the Weighted Input.
// Also called the logit.  LOH-jit
// The logit is the inverse of the standard
// Logistic Function.  The Logistic Function
// is for things like population growth.
// Like with a limited food supply.


// Sigmoid Neuron
// Sigmoid function

activation = static_cast<Float32>(
          1.0 / ( 1.0 + MathC::exp( -z )));

return activation;
}



void Neuron::test( void )
{
Float32 x = -100.0;

// The y output goes from zero to one.

for( Int32 count = 0; count < 200; count++ )
  {
  StIO::printF( "x: " );
  StIO::printFlt64( static_cast<Float64>( x ));
  StIO::putLF();
  x += 1;
  Float32 y = sigmoid(
                 static_cast<Float64>( x ));
  StIO::printF( "y: " );
  StIO::printFlt64( static_cast<Float64>( y ));
  StIO::putLF();
  }
}




void Neuron::setRandomWeights( void )
{
const Int32 max = weightArSize;
for( Int32 count = 0; count < max; count++ )
  {
  Float32 setTo = Randomish::
                      makeRandomFloat32();
  for( Int32 countBits = 0; countBits < 32;
                                  countBits++ )
    {
    // The output from the sigmoid function
    // would be 0 to 1.

    if( setTo <= 1.0f )
      break;

    // This is a crude way of making
    // sort-of-random numbers.
    setTo = setTo / 10.0f;
    }

  StIO::printF( "Weight: " );
  StIO::printFlt64( static_cast<Float64>(
                                       setTo ));
  StIO::putLF();

  setWeight( count, setTo );
  }
}
