// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html




/*
ReLU f(x) = max( 0, x )
ReLU(x):
  if x>0:
    return x
  else:
    return 0

Derivivative of ReLU(x):
  if x>0:
    return 1
  else:
    return 0

*/



using System;



// My MathF class:
// MathF.log( double x )


// The weight at index 0 is the standard way
// of doing the bias.  The input is fixed
// at 1 and the weight makes the bias value.


// namespace




public class Neuron
{
private MainData mData;
private float delta = 0;
private float zSum = 0;
private float activation = 0;

// The weight from this neuron to each
// neuron in the L - 1 layer.

private Float32Array weightAr;




private Neuron()
{
}


internal Neuron( MainData useMainData )
{
mData = useMainData;
weightAr = new Float32Array();
}


internal void setWeightArSize( int setTo )
{
weightAr.setSize( setTo );
}



internal float calcZ( NeuronLayer prevLayer )
{
// By convention, the weight at index
// zero is the bias.  The input at index
// zero should always be 1.

int max = weightAr.getSize();
if( max != prevLayer.getSize())
  {
  throw new Exception(
        "Neuron.calcZ() sizes not equal." );
  }

float z = 0;
for( int count = 0; count < max; count++ )
  {
  mData.showStatus( "activation: " +
        prevLayer.getActivationAt( count ));
  mData.showStatus( "weight: " +
        weightAr.getVal( count ));

  z += prevLayer.getActivationAt( count ) *
                   weightAr.getVal( count );
  }

zSum = z;
mData.showStatus( "zSum: " + zSum );
mData.showStatus( " " );
mData.showStatus( " " );

return zSum;
}



internal float getZSum()
{
return zSum;
}


internal float getDelta()
{
return delta;
}



internal float getActivation()
{
return activation;
}


internal void setActivation( float setTo )
{
activation = setTo;
}



/*
  inline Float32 getWeight(
                   const Int32 where ) const
    {
    return weightAr.getVal( where );
    }

  inline void setWeight( const Int32 where,
                         const Float32 toSet )
    {
    weightAr.setVal( where, toSet );
    }



Float32 Neuron::sigmoid( Float64 z )
{
// z is called the Weighted Input.
// Also called the logit.  LOH-jit
// The logit is the inverse of the standard
// Logistic Function.  The Logistic Function
// is for things like population growth.
// Like with a limited food supply.

// 1.0 / ( 1.0 + exp( -z ))
// Derivative:
// (1.0 + exp( -z ))^-2  *  (e^-z)

// Sigmoid Neuron
// Sigmoid function

activation = static_cast<Float32>(
          1.0 / ( 1.0 + MathC::exp( -z )));

return activation;
}
*/



internal void setRandomWeights( float maxWeight,
                                int randIndex )
{
TimeEC seedTime = new TimeEC();
seedTime.setToNow();
// int timeSeed = (int)seedTime.getTicks();
int seed = (int)seedTime.getIndex();

seed += randIndex;

Random rand = new Random( seed );
float halfWeight = maxWeight / 2;

int max = weightAr.getSize();

// Random value between 0 and maxWeight.
// It is shifted down by half of max weight
// so that the range is -50 to 50 if the
// max weight is 100.

for( int count = 0; count < max; count++ )
  {
  // float weight = randIndex + count;
  float weight = (float)(rand.NextDouble() *
                                  maxWeight );

  // About half of the weights will be
  // negative.
  weight -= halfWeight;

  weightAr.setVal( count, weight );

  // mData.showStatus(
  //   "Weight: " + weightAr.getVal( count ));
  }
}



} // Class
