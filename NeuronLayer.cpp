// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html




#include "NeuronLayer.h"
#include "../CppBase/StIO.h"
#include "../CryptoBase/Randomish.h"



#include "../CppMem/MemoryWarnTop.h"



void NeuronLayer::setSize( const Int32 howBig )
{
if( howBig == arraySize )
  return;

arraySize = howBig;
delete[] neuronAr;
neuronAr = new Neuron[
                 Casting::i32ToU64( arraySize )];
}



void NeuronLayer::setRandomOutput( void )
{
const Int32 max = arraySize;
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

  StIO::printF( "Activity: " );
  StIO::printFlt64( static_cast<Float64>(
                                       setTo ));
  StIO::putLF();

  neuronAr[count].setActivation( setTo );
  }
}



#include "../CppMem/MemoryWarnBottom.h"
