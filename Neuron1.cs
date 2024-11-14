// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html




using System;




// The weight at index 0 is the standard way
// of doing the bias.  The activation is fixed
// at 1 so the weight times 1 makes the
// bias value.


// namespace




public class Neuron1
{
private MainData mData;
private float delta = 0;
// private float deltaAvg = 0;
private float zSum = 0;
private float activation = 0;
private NeuronLayer1 prevLayer;
private NeuronLayer1 nextLayer;
private VectorFlt weightVec;




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
  weightVec.setSize( 1 );
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

// Start with the bias value at zero.
float z = weightVec.getVal( 0 );

// Starting past the bias at 1.
for( int count = 1; count < max; count++ )
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



/*
internal float calcActReLU()
{
activation = Activ.reLU( zSum );
return activation;
}
*/



internal float getZSum()
{
return zSum;
}


internal float getDelta()
{
return delta;
}


// internal float getDeltaAvg()
// {
// return deltaAvg;
// }


internal void setDelta( float setTo )
{
delta = setTo;
}


// internal void addToDeltaAvg( float addTo )
// {
// deltaAvg += addTo;
// }


// internal void clearDeltaAvg()
// {
// deltaAvg = 0;
// }



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
return weightVec.getVal( 0 );
}


internal void setBias( float setTo )
{
weightVec.setVal( 0, setTo );
}


internal float getWeight( int where )
{
int max = weightVec.getSize();
RangeT.test( where, 1, max - 1,
       "Neuron.getWeight() range." );

return weightVec.getVal( where );
}



internal void setWeight( int where, float setTo )
{
int max = weightVec.getSize();
RangeT.test( where, 1, max - 1,
       "Neuron.setWeight() range." );

weightVec.setVal( where, setTo );
}



// Randomness breaks up symmetry so all of
// the neurons/weights are not the same.


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
// float halfWeight = maxWeight / 2;

int max = weightVec.getSize();

// Random value between 0 and maxWeight.

// This sets a random bias at index zero too.

for( int count = 0; count < max; count++ )
  {
  float weight = (float)(rand.NextDouble() *
                                  maxWeight );

  // weight -= halfWeight;

  weightVec.setVal( count, weight );
  }
}



} // Class
