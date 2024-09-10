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

private FloatVec weightVec;




private Neuron()
{
}


internal Neuron( MainData useMainData )
{
mData = useMainData;
weightVec = new FloatVec( mData );
}


internal void setWeightVecSize( int setTo )
{
weightVec.setSize( setTo );
}



internal float calcZ( NeuronLayer prevLayer )
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
activation = sigmoid( zSum );
return activation;
}



internal float calcActReLU()
{
// mData.showStatus( "zSum: " + zSum );

activation = reLU( zSum );

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



internal static float sigmoid( float z )
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

float a = (float)(1.0 / ( 1.0 +
                           MathF.exp( -z )));

return a;
}



/*
internal static float derivSigmoid( float z )
{
// (1.0 + exp( -z ))^-2  *  (e^-z)

float a = (float)( (1.0 +

1.0 / ( 1.0 +
                           MathF.exp( -z )));

return a;
}
*/



internal static float reLU( float z )
{
// ReLU f(x) = max( 0, x )

if( z > 0 )
  return z;

return 0;

// Derivivative of ReLU(x):
//  if x>0:
//    return 1
//  else:
//    return 0
}



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
