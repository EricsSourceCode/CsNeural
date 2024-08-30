// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



using System;



// namespace



// There is that old saying:
// Make it work first, then make it work fast.



public class NeuralNet
{
private MainData mData;
private NeuronLayer inputLayer;
private NeuronLayer hiddenLayer;
private NeuronLayer outputLayer;
private FloatVec errorOutAr;
private FloatMatrix inputMatrix;

// Each input vector is labeled to say what
// it is.  What the network is supposed to
// learn.
private FloatVec testLabelVec;



private NeuralNet()
{
}



internal NeuralNet( MainData useMainData,
                    FloatMatrix useMatrix,
                    FloatVec useLabelVec )
{
mData = useMainData;
inputMatrix = useMatrix;
testLabelVec = useLabelVec;

int last = inputMatrix.getLastAppend();
if( testLabelVec.getSize() <= last )
  {
  throw new Exception(
                "testLabelVec too small." );
  }

inputLayer = new NeuronLayer( mData );
hiddenLayer = new NeuronLayer( mData );
outputLayer = new NeuronLayer( mData );
errorOutAr = new FloatVec( mData );
}



internal void test()
{
mData.showStatus(
          "This is the Neural Net test." );

setupNetTopology();

// setRandomWeights();

/*

// The neuron at zero is always the bias.
// testLabelAr.setVal( 0, 0.0F );

// Set the result to 1 for true.
// testLabelAr.setVal( 1, 1.0F );

forwardPass();
*/

mData.showStatus( "Neural Net test finished." );
}



private void setupNetTopology()
{
int col = inputMatrix.getColumns();

// Plus 1 for the bias at zero.
int layerSize = col + 1;

inputLayer.setSize( layerSize );

// The weights aren't used here.
inputLayer.setWeightArSize( 1 );

hiddenLayer.setSize( layerSize );
hiddenLayer.setWeightArSize( layerSize );

// One for the bias at zero, and one for
// the yes or no answer makes two.
outputLayer.setSize( 2 );

outputLayer.setWeightArSize( layerSize );

errorOutAr.setSize( 2 );
}



private void setRandomWeights( float maxWeight )
{
// The input layer doesn't use weights.
// inputLayer.setRandomWeights()

hiddenLayer.setRandomWeights( maxWeight );
outputLayer.setRandomWeights( maxWeight );
}



====
private void setInputAt( int row )
{
// Set the input layer neurons (activation
// value) from data matrix.

int max = inputLayer.getSize();

// For the bias.
inputLayer.setActivationAt( 0, 1.0F );

for( int count = 1; count < max; count++ )
  {
  float val = inputMatrix.getVal( row, count );
  inputLayer.setActivationAt( count, val );
  }
}



private void forwardPass()
{
hiddenLayer.calcZ( inputLayer );
hiddenLayer.calcActivation();

outputLayer.calcZ( hiddenLayer );
outputLayer.calcActivation();

// The value at zero is the bias.
// So the one output is at index 1.
float aOut = outputLayer.getActivationAt( 1 );
/*
float testOut = testLabelAr.getVal( 1 );

float error = testOut - aOut;
errorOutAr.setVal( 1, error );

mData.showStatus( "aOut: " + aOut );
mData.showStatus( "testOut: " + testOut );
mData.showStatus( "error: " + error );
*/
}



} // Class
