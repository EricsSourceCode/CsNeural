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



internal int getSize()
{
return neuronAr.Length;
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




internal void setWeightArSize( int setTo )
{
// All of the neurons in this layer have
// the same number of inputs from the
// previous layer.

int max = neuronAr.Length;
for( int count = 0; count < max; count++ )
  neuronAr[count].setWeightVecSize( setTo );

}



internal float getActivationAt( int where )
{
// By convention, index zero is for the
// bias, so this should always return 1 for
// the Activation at index 0.

// Is that true of every layer?

if( where == 0 )
  return 1;

int max = neuronAr.Length;
RangeT.test( where, 0, max - 1,
       "NeuronLayer.getActivationAt() range." );

return neuronAr[where].getActivation();
}



internal void setActivationAt( int where,
                               float setTo )
{
// By convention, it is the bias.
if( where == 0 )
  {
  neuronAr[where].setActivation( 1.0F );
  return;
  }

int max = neuronAr.Length;
RangeT.test( where, 0, max - 1,
       "NeuronLayer.setActivationAt() range." );

neuronAr[where].setActivation( setTo );
}



internal void setRandomWeights( float maxWeight )
{
int max = neuronAr.Length;
for( int count = 0; count < max; count++ )
  {
  neuronAr[count].setRandomWeights(
                    maxWeight, count );
  }
}



internal void calcZ( NeuronLayer prevLayer )
{
int max = neuronAr.Length;
for( int count = 0; count < max; count++ )
  {
  neuronAr[count].calcZ( prevLayer );
  }
}



internal void calcActivation()
{
int max = neuronAr.Length;
for( int count = 0; count < max; count++ )
  {
  neuronAr[count].calcActivation();
  }
}





} // Class
