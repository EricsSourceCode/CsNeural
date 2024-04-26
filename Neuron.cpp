// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html




#include "Neuron.h"
#include "../Math/MathC.h"
#include "../CppBase/StIO.h"



Float32 Neuron::sigmoid( Float32 z )
{
// z is called the Weighted Input.

// Sigmoid Neuron
// Sigmoid function

activation = Casting::float64To32(
          1.0 / ( 1.0 + MathC::exp(
              Casting::float32To64( -z ))));

return activation;
}



void Neuron::test( void )
{
Float32 x = -100.0;

// The y output goes from zero to one.

for( Int32 count = 0; count < 200; count++ )
  {
  StIO::printF( "x: " );
  StIO::printFlt64( Casting::float32To64( x ));
  StIO::putLF();
  x += 1;
  Float32 y = sigmoid( x );
  StIO::printF( "y: " );
  StIO::printFlt64( Casting::float32To64( y ));
  StIO::putLF();
  }
}
