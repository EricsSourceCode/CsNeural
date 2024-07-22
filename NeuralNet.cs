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
//  NeuronLayer inputLayer;
//  NeuronLayer hiddenLayer;
//  NeuronLayer outputLayer;
//  Float32Array testLabelAr;



private NeuralNet()
{
}



internal NeuralNet( MainData useMainData )
{
mData = useMainData;
}


/*
void NeuralNet::test( void )
{
StIO::putS( "\n\nThis is the Neural Net test." );


// Set the input layer neurons (activation value)
// to random values
// between 0 and 1.  Because that is what they
// would be from the sigmoid function.

inputLayer.setSize( 15 );
hiddenLayer.setSize( 10 );

// These two have to be the same size
// because of the output error function.
outputLayer.setSize( 10 );
testLabelAr.setSize( 10 );

testLabelAr.setVal( 0, 0.0f );
testLabelAr.setVal( 1, 0.1f );
testLabelAr.setVal( 2, 0.2f );
testLabelAr.setVal( 3, 0.3f );
testLabelAr.setVal( 4, 0.4f );
testLabelAr.setVal( 5, 0.5f );
testLabelAr.setVal( 6, 0.6f );
testLabelAr.setVal( 7, 0.7f );
testLabelAr.setVal( 8, 0.8f );
testLabelAr.setVal( 9, 0.9f );


inputLayer.setRandomOutput();

// hiddenLayer;
// outputLayer;

StIO::putS( "Neural Net test finished.\n\n" );
}
*/



} // Class
