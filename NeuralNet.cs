// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



using System;



// namespace



// This is not done with GPUs and matrices
// because it is explanatory.
// I don't have an NVidea processor on
// my laptop computer for CUDA, and those
// kinds of optimizations are for a much
// later version.



public class NeuralNet
{
private MainData mData;
NeuronLayer inputLayer;
//  NeuronLayer hiddenLayer;
NeuronLayer outputLayer;
Float32Array testLabelAr;



private NeuralNet()
{
}



internal NeuralNet( MainData useMainData )
{
mData = useMainData;
inputLayer = new NeuronLayer( mData );
//  hiddenLayer;
outputLayer = new NeuronLayer( mData );;
testLabelAr = new Float32Array();
}




internal void test()
{
/*
Train one neuron.
Inputs are just static values.
One output.
Use:
Quadratic Cost function
Sigmoid function.

*/

mData.showStatus(
          "This is the Neural Net test." );

inputLayer.setSize( 15 );
outputLayer.setInputSize( 15 );

// hiddenLayer.setSize( 10 );

// These two have to be the same size
// because of the output error function.
outputLayer.setSize( 1 );
testLabelAr.setSize( 1 );
testLabelAr.setVal( 0, 1.0F );

// Set the input layer neurons (activation value)
// to random values.
// They might be a number representing the
// index of a word in a dictionary, for
// example.

Random rand = new Random();

int max = inputLayer.getSize();

inputLayer.setActivationAt( 0, 1.0F );

// Random value between 0 and 100.
for( int count = 1; count < max; count++ )
  {
  inputLayer.setActivationAt( count,
           (float)(rand.NextDouble() * 100 ));
  }


mData.showStatus( "Neural Net test finished." );
}




} // Class
