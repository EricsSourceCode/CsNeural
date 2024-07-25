// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



using System;



// namespace



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
// Train one neuron.

mData.showStatus(
          "This is the Neural Net test." );

inputLayer.setSize( 15 );
// hiddenLayer.setSize( 10 );

// These two have to be the same size
// because of the output error function.
outputLayer.setSize( 1 );
testLabelAr.setSize( 1 );


/*
// Set the input layer neurons (activation value)
// to random values
// between 0 and 1.  Because that is what they
// would be from the sigmoid function.

Random rand = new Random();

// Random double between 0 and 100.
double nextRand = rand.NextDouble() * 100;

testLabelAr.setVal( 0, 0.0f );

*/

mData.showStatus( "Neural Net test finished." );
}




} // Class


