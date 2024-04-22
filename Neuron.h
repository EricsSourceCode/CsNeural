// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



#pragma once



// A good tutorial:
// http://neuralnetworksanddeeplearning.com/
//                                  chap1.html

// Sigmoid Neurons:
// http://neuralnetworksanddeeplearning.com/
//               chap1.html#sigmoid_neurons



#include "../CppBase/BasicTypes.h"
// #include "../CppBase/CharBuf.h"



class Neuron
  {
  private:
  bool testForCopy = false;
  Float64 output;

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

  static Float64 sigmoid( Float64 sum );
  void test( void );

  inline Float64 getOutput( void )
    {
    return output;
    }

  };
