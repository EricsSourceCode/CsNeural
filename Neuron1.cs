// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html




using System;




// The weight at index 0 is the standard way
// of doing the bias.  The input is fixed
// at 1 and the weight makes the bias value.


// namespace




public class Neuron1
{
private MainData mData;
private float delta = 0;
private float zSum = 0;
private float activation = 0;

// The weight from this neuron to each
// neuron in the L - 1 layer.

private VectorFlt weightVec;




private Neuron1()
{
}


internal Neuron1( MainData useMainData )
{
mData = useMainData;
weightVec = new VectorFlt( mData );
}


internal void setWeightVecSize( int setTo )
{
weightVec.setSize( setTo );
}



internal int getWeightVecSize()
{
return weightVec.getSize();
}



internal float calcZ( NeuronLayer1 prevLayer )
{
// By convention, the weight at index
// zero is the bias.  The input at index
// zero should always be 1.

int max = weightVec.getSize();
if( max != prevLayer.getSize())
  {
  throw new Exception(
        "Neuron.calcZ() sizes not equal." );
  }

// Starting at zero, so zSum includes the
// bias value.
if( prevLayer.getActivationAt( 0 ) < 0.999 )
  {
  throw new Exception(
                   "Bias activation not 1." );
  }


float z = 0;
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
activation = Activation.sigmoid( zSum );
return activation;
}



internal float calcActReLU()
{
// mData.showStatus( "zSum: " + zSum );

activation = Activation.reLU( zSum );

// mData.showStatus( "activation: " + activation );

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



internal float getWeight( int where )
{
return weightVec.getVal( where );
}



internal void setWeight( int where, float setTo )
{
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
float halfWeight = maxWeight / 2;

int max = weightVec.getSize();

// Random value between 0 and maxWeight.
// It is shifted down by half of max weight
// so that the range is -50 to 50 if the
// max weight is 100.

for( int count = 0; count < max; count++ )
  {
  float weight = (float)(rand.NextDouble() *
                                  maxWeight );

  // About half of the weights will be
  // negative.
  weight -= halfWeight;

  weightVec.setVal( count, weight );
  }
}



} // Class
