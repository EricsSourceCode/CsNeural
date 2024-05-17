// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



#pragma once



#include "../CppBase/BasicTypes.h"
// #include "../CppBase/CharBuf.h"
#include "Neuron.h"
#include "../CppBase/Casting.h"
#include "../CppBase/RangeC.h"



class NeuronLayer
  {
  private:
  bool testForCopy = false;
  Neuron* neuronAr;
  Int32 arraySize = 1;

  public:
  NeuronLayer( void )
    {
    arraySize = 1;
    neuronAr = new Neuron[
                  Casting::i32ToU64( arraySize )];

    }

  NeuronLayer( const NeuronLayer& in )
    {
    arraySize = 1;
    neuronAr = new Neuron[
                  Casting::i32ToU64( arraySize )];

    if( in.testForCopy )
      return;

    throw "NeuronLayer copy constructor.";
    }

  ~NeuronLayer( void )
    {
    delete[] neuronAr;
    }

  void setSize( const Int32 howBig );
  void setRandomOutput( void );

  };
