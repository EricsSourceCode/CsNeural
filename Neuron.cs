// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html






// A good tutorial:
// http://neuralnetworksanddeeplearning.com/
//                                  chap1.html
//                                  chap2.html

// Sigmoid Neurons:
// http://neuralnetworksanddeeplearning.com/
//                   chap1.html#sigmoid_neurons


// GPT: Generative pre-trained transformer
// A type of large language model.


using System;


// Matrix notation:
// w for weight
// w<sup>L<sub>jk
// The weight for the connection
// from the kth neuron in the (L - 1)th
// layer to the jth neuron in the Lth
// layer.

// b for bias.
// b<sup>L<sub>j
// bias on the jth neuron in the Lth layer.

// a for activation
// a<sup>L<sub>j
// Activation for the jth neuron in
//            the Lth layer.
//            Meaning the output of that neuron.




// namespace




public class Neuron
{
private MainData mData;
// private float activation; // The output from
            // the Activation function.
            // Also called y.

// Each dendrite has a weight.
// The weight from this neuron to each
// neuron in the L - 1 layer.
private Float32Array weightAr;

// The weight at index 0 is the standard way
// of doing the bias.  The input is fixed
// at 1 and the weight makes the bias value.
// z = sigma( inputs * weights) + (bias * 1)


private Neuron()
{
}


internal Neuron( MainData useMainData )
{
mData = useMainData;
weightAr = new Float32Array();
// weightAr.setSize( to what? );
}



/*
internal float getActivation()
{
return activation;
}

  inline void setActivation(
                      const Float32 setTo )
    {
    activation = setTo;
    }

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

  void setRandomWeights( void );


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



void Neuron::test( void )
{
Float32 x = -100.0;

// The y output goes from zero to one.

for( Int32 count = 0; count < 200; count++ )
  {
  StIO::printF( "x: " );
  StIO::printFlt64( static_cast<Float64>( x ));
  StIO::putLF();
  x += 1;
  Float32 y = sigmoid(
                 static_cast<Float64>( x ));
  StIO::printF( "y: " );
  StIO::printFlt64( static_cast<Float64>( y ));
  StIO::putLF();
  }
}




void Neuron::setRandomWeights( void )
{
const Int32 max = weightArSize;
for( Int32 count = 0; count < max; count++ )
  {
  Float32 setTo = Randomish::
                      makeRandomFloat32();
  for( Int32 countBits = 0; countBits < 32;
                                  countBits++ )
    {
    // The output from the sigmoid function
    // would be 0 to 1.

    if( setTo <= 1.0f )
      break;

    // This is a crude way of making
    // sort-of-random numbers.
    setTo = setTo / 10.0f;
    }

  StIO::printF( "Weight: " );
  StIO::printFlt64( static_cast<Float64>(
                                       setTo ));
  StIO::putLF();

  setWeight( count, setTo );
  }
}
*/


} // Class
