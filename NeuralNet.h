// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



#pragma once


#include "../CppBase/BasicTypes.h"
#include "NeuronLayer.h"



class NeuralNet
  {
  private:
  bool testForCopy = false;
  NeuronLayer inputLayer;
  NeuronLayer hiddenLayer;
  NeuronLayer outputLayer;

  public:
  NeuralNet( void )
    {
    }

  NeuralNet( const NeuralNet& in )
    {
    if( in.testForCopy )
      return;

    throw "NeuralNet copy constructor.";
    }

  ~NeuralNet( void )
    {
    }

  void test( void );


  };
