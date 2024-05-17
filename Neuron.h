// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



#pragma once



// A good tutorial:
// http://neuralnetworksanddeeplearning.com/
//                                  chap1.html
//                                  chap2.html

// Sigmoid Neurons:
// http://neuralnetworksanddeeplearning.com/
//                   chap1.html#sigmoid_neurons



#include "../CppBase/BasicTypes.h"
#include "../CppMem/Float32Array.h"




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



class Neuron
  {
  private:
  bool testForCopy = false;
  Float32 activation; // The output from
            // the sigmoid Activation function.
            // Also called y.

  // Each dendrite has a weight.
  // The weight from this neuron to each
  // neuron in the L - 1 layer.
  Float32Array weightAr;
  Int32 weightArSize = 1;

  Float32 bias; // The bias is not something
                // that comes from another
                // neuron.

  public:
  Neuron( void )
    {
    }

  Neuron( const Neuron& in )
    {
    if( in.testForCopy )
      return;

    throw "Neuron copy constructor.";
    }

  ~Neuron( void )
    {
    }

  Float32 sigmoid( Float64 z );
  void test( void );

  inline Float32 getActivation( void ) const
    {
    return activation;
    }

  inline void setActivation(
                      const Float32 setTo )
    {
    activation = setTo;
    }

  inline Float32 getBias( void ) const
    {
    return bias;
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

  };
