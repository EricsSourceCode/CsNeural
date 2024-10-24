// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



using System;



// namespace



// By convention, the weight at index
// zero is the bias.  And so neurons
// start at index 1 in order to match
// up with those weights.  And so later
// math optimizations (like GPU math)
// matches up indexes.
// The neuron at index 0 is not used.



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

// There is no neuron at zero.
// Starting at index 1.
for( int count = 1; count < max; count++ )
  neuronAr[count].setWeightVecSize( setTo );

}


internal void getActivationVec(
                           VectorFlt vecToGet )
{
int max = neuronAr.Length;

// If the size needs to be set.
vecToGet.setSize( max );

for( int count = 0; count < max; count++ )
  {
  float act = neuronAr[count].getActivation();
  vecToGet.setVal( count, act );
  }
}



internal float getActivationAt( int where )
{
int max = neuronAr.Length;
RangeT.test( where, 1, max - 1,
       "NeuronLayer.getActivationAt() range." );

return neuronAr[where].getActivation();
}



internal void setActivationAt( int where,
                               float setTo )
{
int max = neuronAr.Length;
RangeT.test( where, 1, max - 1,
       "NeuronLayer.setActivationAt() range." );

neuronAr[where].setActivation( setTo );
}



internal float getZSumAt( int where )
{
int max = neuronAr.Length;
RangeT.test( where, 1, max - 1,
          "NeuronLayer.getZSumAt() range." );

return neuronAr[where].getZSum();
}



internal void setRandomWeights( float maxWeight )
{
int max = neuronAr.Length;
for( int count = 1; count < max; count++ )
  {
  neuronAr[count].setRandomWeights(
                    maxWeight, count );
  }
}



internal void calcZ( NeuronLayer1 prevLayer )
{
int max = neuronAr.Length;
for( int count = 1; count < max; count++ )
  neuronAr[count].calcZ( prevLayer );

}



internal void calcActSigmoid()
{
int max = neuronAr.Length;
for( int count = 1; count < max; count++ )
  neuronAr[count].calcActSigmoid();

}



internal void calcActReLU()
{
int max = neuronAr.Length;
for( int count = 1; count < max; count++ )
  neuronAr[count].calcActReLU();

}


internal void clearAllDeltaAvg()
{
int max = neuronAr.Length;
for( int count = 1; count < max; count++ )
  neuronAr[count].clearDeltaAvg();

}



internal float getDeltaAt( int where )
{
// By convention, it is the bias
// if where == 0.  So check the range
// starting at 1.

int max = neuronAr.Length;
RangeT.test( where, 1, max - 1,
       "NeuronLayer.getDeltaAt() range." );

return neuronAr[where].getDelta();
}



internal float getDeltaAvgAt( int where )
{
int max = neuronAr.Length;
RangeT.test( where, 1, max - 1,
       "NeuronLayer.getDeltaAvgAt() range." );

return neuronAr[where].getDeltaAvg();
}




internal void addToDeltaAvgAt( int where,
                               float addTo )
{
int max = neuronAr.Length;
RangeT.test( where, 1, max - 1,
       "NeuronLayer.addToDeltaAvgAt() range." );

neuronAr[where].addToDeltaAvg( addTo );
}



internal void clearDeltaAvgAt( int where )
{
int max = neuronAr.Length;
RangeT.test( where, 1, max - 1,
       "NeuronLayer.clearDeltaAvgAt() range." );

neuronAr[where].clearDeltaAvg();
}




internal void setDeltaAt( int where,
                          float setTo )
{
int max = neuronAr.Length;
RangeT.test( where, 1, max - 1,
       "NeuronLayer.setDeltaAt() range." );

neuronAr[where].setDelta( setTo );
}



internal float getBias( int neuron )
{
int max = neuronAr.Length;
RangeT.test( neuron, 1, max - 1,
    "NeuronLayer.getBias() neuron range." );

return neuronAr[neuron].getBias();
}



internal void setBias( int neuron, float setTo )
{
int max = neuronAr.Length;
RangeT.test( neuron, 1, max - 1,
    "NeuronLayer.setBias() neuron range." );

neuronAr[neuron].setBias( setTo );
}



internal float getWeight( int neuron, int where )
{
int max = neuronAr.Length;
RangeT.test( neuron, 1, max - 1,
    "NeuronLayer.getWeight() neuron range." );

int weightSize = neuronAr[neuron].
                        getWeightVecSize();
RangeT.test( where, 1, weightSize - 1,
     "NeuronLayer.getWeight() where range." );

return neuronAr[neuron].getWeight( where );
}



internal void setWeight( int neuron, int where,
                                     float setTo )
{
int max = neuronAr.Length;
RangeT.test( neuron, 1, max - 1,
    "NeuronLayer.setWeight() neuron range." );

int weightSize = neuronAr[neuron].
                        getWeightVecSize();
RangeT.test( where, 1, weightSize - 1,
     "NeuronLayer.setWeight() where range." );

neuronAr[neuron].setWeight( where, setTo );
}




} // Class
