// Copyright Eric Chauvin 2024 - 2025.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html




using System;



// This way of doing the bias is for a
// later version:
// The weight at index 0 is the standard way
// of doing the bias.  The activation is fixed
// at 1 so the weight times 1 makes the
// bias value.


// namespace




public class Neuron1
{
private MainData mData;
private float delta = 0;
private float zSum = 0;
private float activation = 0;
private NeuronLayer1 prevLayer;
private NeuronLayer1 nextLayer;
private VectorFlt weightVec;

// This bias is separate from the weights
// in this version.

private float bias = 0;



private Neuron1()
{
}


internal Neuron1( MainData useMainData )
{
mData = useMainData;
weightVec = new VectorFlt( mData );
}


internal void setLayers(
               NeuronLayer1 usePrevLayer,
               NeuronLayer1 useNextLayer )
{
// A layer can be null if it's not there.

prevLayer = usePrevLayer;
nextLayer = useNextLayer;

if( prevLayer == null )
  weightVec.setSize( 1 ); // Not used.
else
  weightVec.setSize( prevLayer.getSize());

}



internal int getWeightVecSize()
{
return weightVec.getSize();
}



internal float calcZ()
{
int max = weightVec.getSize();
if( max != prevLayer.getSize())
  {
  throw new Exception(
        "Neuron.calcZ() sizes not equal." );
  }

// Start with the bias value.
float z = bias;

for( int count = 0; count < max; count++ )
  {
  z += prevLayer.getActivationAt( count ) *
                   weightVec.getVal( count );
  }

zSum = z;
return zSum;
}



internal float calcActSigmoid()
{
activation = Activ.sigmoid( zSum );
return activation;
}




internal float getZSum()
{
return zSum;
}


internal float getDelta()
{
return delta;
}



internal void setDelta( float setTo )
{
delta = setTo;
}




internal float getActivation()
{
return activation;
}


internal void setActivation( float setTo )
{
activation = setTo;
}


internal float getBias()
{
return bias;
}


internal void setBias( float setTo )
{
bias = setTo;
}


internal void copyWeightVec( VectorFlt toGet )
{
toGet.copy( weightVec );
}


internal float getWeight( int where )
{
int max = weightVec.getSize();
RangeT.test( where, 0, max - 1,
       "Neuron.getWeight() range." );

return weightVec.getVal( where );
}



internal void setWeight( int where, float setTo )
{
int max = weightVec.getSize();
RangeT.test( where, 0, max - 1,
       "Neuron.setWeight() range." );

weightVec.setVal( where, setTo );
}



// Randomness breaks up symmetry.



internal void setRandomWeights( float maxWeight,
                                int randIndex )
{
TimeEC seedTime = new TimeEC();
seedTime.setToNow();
// int seed = (int)seedTime.getTicks();
int seed = (int)(seedTime.getIndex()
                 & 0x7FFFFFFFF );

seed += randIndex;

Random rand = new Random( seed );
int max = weightVec.getSize();

// Random value between 0 and maxWeight.

bias = (float)(rand.NextDouble() * maxWeight );

for( int count = 0; count < max; count++ )
  {
  float weight = (float)(rand.NextDouble() *
                                  maxWeight );

  weightVec.setVal( count, weight );
  }
}



} // Class
