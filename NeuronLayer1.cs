// Copyright Eric Chauvin 2024 - 2025.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



using System;



// namespace



// This way of doing the bias is for a
// later version:
// By convention, the weight at index
// zero is the bias.



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
RangeT.test( where, 0, max - 1,
       "NeuronLayer.getActivationAt() range." );

return neuronAr[where].getActivation();
}



internal void setActivationAt( int where,
                               float setTo )
{
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



internal void calcZ()
{
int max = neuronAr.Length;
for( int count = 0; count < max; count++ )
  neuronAr[count].calcZ();

}



internal void calcActSigmoid()
{
int max = neuronAr.Length;
for( int count = 0; count < max; count++ )
  neuronAr[count].calcActSigmoid();

}




internal float getDeltaAt( int where )
{
int max = neuronAr.Length;
RangeT.test( where, 0, max - 1,
       "NeuronLayer.getDeltaAt() range." );

return neuronAr[where].getDelta();
}




internal void setDeltaAt( int where,
                          float setTo )
{
int max = neuronAr.Length;
RangeT.test( where, 0, max - 1,
       "NeuronLayer.setDeltaAt() range." );

neuronAr[where].setDelta( setTo );
}



internal float getBias( int neuron )
{
int max = neuronAr.Length;
RangeT.test( neuron, 0, max - 1,
    "NeuronLayer.getBias() neuron range." );

return neuronAr[neuron].getBias();
}



internal void setBias( int neuron, float setTo )
{
int max = neuronAr.Length;
RangeT.test( neuron, 0, max - 1,
    "NeuronLayer.setBias() neuron range." );

neuronAr[neuron].setBias( setTo );
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



internal void setWeight( int neuron, int where,
                                     float setTo )
{
int max = neuronAr.Length;
RangeT.test( neuron, 0, max - 1,
    "NeuronLayer.setWeight() neuron range." );

int weightSize = neuronAr[neuron].
                        getWeightVecSize();
RangeT.test( where, 0, weightSize - 1,
     "NeuronLayer.setWeight() where range." );

neuronAr[neuron].setWeight( where, setTo );
}



internal bool setDeltaForHidden()
{
int max = getSize();
int maxNext = nextLayer.getSize();

// Big exponential loops.
// A lot of computation is done in
// these loops.

// countT is the neuron in this layer.

for( int countT = 0; countT < max; countT++ )
  {
  if( (countT % 100) == 0 )
    {
    if( !mData.checkEvents())
      return false;
    }


  float sumToSet = 0;

  // The z for this layer.
  float z = getZSumAt( countT );

  // countNext is the neuron in the next layer.

  for( int countNext = 0;
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

    sumToSet += partSum;
    }

  setDeltaAt( countT, sumToSet );
  }

return true;
}



internal void adjustBias( float rate )
{
// dCost / dBias = delta

int max = getSize();

for( int count = 0; count < max; count++ )
  {
  // The delta for this neuron and the
  // bias for this neuron.

  float delta = getDeltaAt( count );
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

for( int countPrev = 0; countPrev < maxPrev;
                        countPrev++ )
  {
  if( (countPrev % 100) == 0 )
    {
    if( !mData.checkEvents())
      return;
    }

  float act = prevLayer.getActivationAt(
                                   countPrev );

  for( int count = 0; count < max; count++ )
    {
    float delta = getDeltaAt( count );

    float weight = getWeight( count,
                              countPrev );

    float wAdjust = rate * delta * act;
    weight += wAdjust;
    setWeight( count, countPrev, weight );
    }
  }
}



internal void copyWeightVecAt( int where,
                               VectorFlt toGet )
{
int max = neuronAr.Length;
RangeT.test( where, 0, max - 1,
      "NeuronLayer.copyWeightVecAt() range." );

neuronAr[where].copyWeightVec( toGet );
}




internal void copyBiasVec( VectorFlt toGet )
{
int last = getSize();
toGet.setSize( last );

for( int count = 0; count < last; count++ )
  {
  float setBias = neuronAr[count].getBias();
  toGet.setVal( count, setBias );
  }
}




internal void makeWeightVecArray(
                           VectorArray toGet )
{
toGet.clearLastAppend();

VectorFlt vec = new VectorFlt( mData );

int max = neuronAr.Length;
if( max < 1 )
  return;

copyWeightVecAt( 0, vec );

int columns = vec.getSize();

// Start with 100 rows.

toGet.setSize( 100, columns );

for( int count = 0; count < max; count++ )
  {
  copyWeightVecAt( count, vec );
  toGet.appendVecCopy( vec );
  }
}



} // Class
