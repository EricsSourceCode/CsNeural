// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



#pragma once


#include "../CppBase/BasicTypes.h"
#include "../CppBase/CharBuf.h"

class Neuron
  {
  private:
  bool testForCopy = false;

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

  };
