// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



using System;



// namespace



public class NeuronLayer
{
private MainData mData;
private Neuron[] neuronAr;



private NeuronLayer()
{
}




internal NeuronLayer( MainData useMainData )
{
mData = useMainData;

try
{
neuronAr = new Neuron[2];
for( int count = 0; count < 2; count++ )
  neuronAr[count] = new Neuron( mData );

}
catch( Exception ) // Except )
  {
  // freeAll();
  throw new Exception(
          "Not enough memory for NeuronLayer." );
  }
}



internal void setSize( int howBig )
{
int arraySize = neuronAr.Length;
if( howBig == arraySize )
  return;

try
{
neuronAr = new Neuron[howBig];
for( int count = 0; count < howBig; count++ )
  neuronAr[count] = new Neuron( mData );

}
catch( Exception ) // Except )
  {
  throw new Exception(
          "Not enough memory for NeuronLayer." );
  }
}




/*
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
*/


} // Class
