// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html




#include "Neuron.h"
#include "../Math/MathC.h"
#include "../CppBase/StIO.h"



Float64 Neuron::sigmoid( Float64 sum )
{
// sum is the sum of all the values and
// weights minus the bias.
// WjXj - b

// Sigmoid Neuron
// Sigmoid function

return 1.0 / ( 1.0 + MathC::exp( -sum ));
}



void Neuron::test( void )
{
Float64 x = -100.0;

// The y output goes from zero to one.

for( Int32 count = 0; count < 200; count++ )
  {
  StIO::printF( "x: " );
  StIO::printFlt64( x );
  StIO::putLF();
  x += 1;
  Float64 y = sigmoid( x );
  StIO::printF( "y: " );
  StIO::printFlt64( y );
  StIO::putLF();
  }
}
