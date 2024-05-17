// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html




#include "NeuralNet.h"
#include "../CppBase/StIO.h"
// #include "../CryptoBase/Randomish.h"


void NeuralNet::test( void )
{
StIO::putS( "\n\nThis is the Neural Net test." );


// Set the input layer neurons (activation value)
// to random values
// between 0 and 1.  Because that is what they
// would be from the sigmoid function.

inputLayer.setSize( 50 );
inputLayer.setRandomOutput();

// hiddenLayer;
// outputLayer;

StIO::putS( "Neural Net test finished.\n\n" );
}
