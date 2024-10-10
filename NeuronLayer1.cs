// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



using System;



// namespace



public class NeuronLayer1
{
private MainData mData;
private Neuron1[] neuronAr;



private NeuronLayer1()
{
}




internal NeuronLayer1( MainData useMainData )
{
mData = useMainData;

try
{
neuronAr = new Neuron1[2];
for( int count = 0; count < 2; count++ )
  neuronAr[count] = new Neuron1( mData );

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
neuronAr = new Neuron1[howBig];
for( int count = 0; count < howBig; count++ )
  neuronAr[count] = new Neuron1( mData );

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



internal float getZSumAt( int where )
{
int max = neuronAr.Length;
RangeT.test( where, 0, max - 1,
          "NeuronLayer.getZSumAt() range." );

return neuronAr[where].getZSum();
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



internal void calcZ( NeuronLayer1 prevLayer )
{
int max = neuronAr.Length;
for( int count = 0; count < max; count++ )
  {
  neuronAr[count].calcZ( prevLayer );
  }
}



internal void calcActSigmoid()
{
int max = neuronAr.Length;
for( int count = 0; count < max; count++ )
  {
  neuronAr[count].calcActSigmoid();
  }
}



internal void calcActReLU()
{
int max = neuronAr.Length;
for( int count = 0; count < max; count++ )
  {
  neuronAr[count].calcActReLU();
  }
}



internal float getDeltaAt( int where )
{
// By convention, it is the bias
// if where == 0.  So check the range
// starting at 1.

int max = neuronAr.Length;
RangeT.test( where, 1, max - 1,
       "NeuronLayer.setDeltaAt() range." );

return neuronAr[where].getDelta();
}



internal void setDeltaAt( int where,
                          float setTo )
{
int max = neuronAr.Length;
RangeT.test( where, 1, max - 1,
       "NeuronLayer.setDeltaAt() range." );

neuronAr[where].setDelta( setTo );
}


internal float getWeight( int neuron, int where )
{
int max = neuronAr.Length;
RangeT.test( neuron, 0, max - 1,
    "NeuronLayer.getWeight() neuron range." );

int weightSize = neuronAr[neuron].
                        getWeightVecSize();
RangeT.test( where, 0, weightSize - 1,
     "NeuronLayer.getWeight() where range." );

return neuronAr[neuron].getWeight( where );
}





} // Class
