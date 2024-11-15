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
// up with those weights.  So later
// math optimizations (like GPU math)
// matches up indexes with this.



public class NeuronLayer1
{
private MainData mData;
private Neuron1[] neuronAr;
private NeuronLayer1 prevLayer;
private NeuronLayer1 nextLayer;



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



/*
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
*/


internal void setLayers(
                     NeuronLayer1 usePrevLayer,
                     NeuronLayer1 useNextLayer )
{
// A layer can be null if it's not there.

prevLayer = usePrevLayer;
nextLayer = useNextLayer;

int max = neuronAr.Length;

for( int count = 0; count < max; count++ )
  {
  neuronAr[count].setLayers( prevLayer,
                             nextLayer );
  }
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



internal void calcZ()
{
int max = neuronAr.Length;
for( int count = 1; count < max; count++ )
  neuronAr[count].calcZ();

}



internal void calcActSigmoid()
{
int max = neuronAr.Length;
for( int count = 1; count < max; count++ )
  neuronAr[count].calcActSigmoid();

}


/*
internal void calcActReLU()
{
int max = neuronAr.Length;
for( int count = 1; count < max; count++ )
  neuronAr[count].calcActReLU();

}
*/


// internal void clearAllDeltaAvg()
// {
// int max = neuronAr.Length;
// for( int count = 1; count < max; count++ )
  // neuronAr[count].clearDeltaAvg();

// }



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



// internal float getDeltaAvgAt( int where )
// {
// int max = neuronAr.Length;
// RangeT.test( where, 1, max - 1,
//      "NeuronLayer.getDeltaAvgAt() range." );

// return neuronAr[where].getDeltaAvg();
// }




// internal void addToDeltaAvgAt( int where,
//                               float addTo )
// {
// int max = neuronAr.Length;
// RangeT.test( where, 1, max - 1,
//     "NeuronLayer.addToDeltaAvgAt() range." );

// neuronAr[where].addToDeltaAvg( addTo );
// }



// internal void clearDeltaAvgAt( int where )
// {
// int max = neuronAr.Length;
// RangeT.test( where, 1, max - 1,
//      "NeuronLayer.clearDeltaAvgAt() range." );

// neuronAr[where].clearDeltaAvg();
// }




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



internal bool setDeltaForHidden()
{
int max = getSize();
int maxNext = nextLayer.getSize();

// Start counting at 1 because the bias
// is the only thing at zero.

// Big exponential loops.

// countT is the neuron in this layer.

for( int countT = 1; countT < max; countT++ )
  {
  if( (countT % 10) == 0 )
    {
    if( !mData.checkEvents())
      return false;
    }


  float sumToSet = 0;

  // The z for this layer.
  float z = getZSumAt( countT );

  // countNext is the neuron in the next layer.

  for( int countNext = 1;
           countNext < maxNext; countNext++ )
    {
    float deltaNext = nextLayer.getDeltaAt(
                                countNext );

    // Here is a Matrix.  Row and column.

    float weight = nextLayer.getWeight(
                          countNext, countT );

    // Using the z in this layer.
    float partSum = (weight * deltaNext) *
                        Activ.drvSigmoid( z );
                        // drvReLU( z )
    sumToSet += partSum;
    }

  setDeltaAt( countT, sumToSet );

  // toSetLayer.addToDeltaAvgAt( countT,
  //                            sumToSet );
  }

return true;
}



internal void adjustBias( float rate )
{
// dCost / dBias = delta

int max = getSize();

for( int count = 1; count < max; count++ )
  {
  // The delta for this neuron and the
  // bias for this neuron.

  float delta = getDeltaAt( count );
  // float delta = layer.getDeltaAvgAt( count );
  // delta = delta / batchSize;

  // ======
  // A product for these two vectors.

  float bias = getBias( count );
  float biasAdj = delta * rate;
  bias += biasAdj;
  setBias( count, bias );
  }
}




internal void adjustWeights( float rate )
{
// dError / dW = activation * delta

int maxPrev = prevLayer.getSize();
int max = getSize();

for( int countPrev = 1; countPrev < maxPrev;
                        countPrev++ )
  {
  if( (countPrev % 10) == 0 )
    {
    if( !mData.checkEvents())
      return;
    }

  float act = prevLayer.getActivationAt(
                                   countPrev );

  for( int count = 1; count < max; count++ )
    {
    float delta = getDeltaAt( count );
    // float delta = getDeltaAvgAt( count );
    // delta = delta / batchSize;

    float weight = getWeight( count,
                              countPrev );

    float wAdjust = rate * delta * act;
    weight += wAdjust;
    setWeight( count, countPrev, weight );
    }
  }
}



} // Class
