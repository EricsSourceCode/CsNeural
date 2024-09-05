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
private FloatVec errorOutVec;
private FloatMatrix inputMatrix;
private FloatMatrix labelMatrix;




private NeuralNet()
{
}



internal NeuralNet( MainData useMainData,
                  FloatMatrix useMatrix,
                  FloatMatrix useLabelMatrix )
{
mData = useMainData;
inputMatrix = useMatrix;
labelMatrix = useLabelMatrix;

int last = inputMatrix.getLastAppend();
if( labelMatrix.getLastAppend() != last )
  {
  throw new Exception(
          "labelMatrix doesn't match last." );
  }

inputLayer = new NeuronLayer( mData );
hiddenLayer = new NeuronLayer( mData );
outputLayer = new NeuronLayer( mData );
errorOutVec = new FloatVec( mData );
}



internal void test()
{
mData.showStatus( "NeuralNet.test()." );

setupNetTopology();

setRandomWeights( 10.0F );

// Loop through these...
setInputRow( 0 );
// setLabelRow()  ??

forwardPass();


mData.showStatus( "NeuralNet.test() end." );
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

errorOutVec.setSize( 2 );
}




private void setRandomWeights( float maxWeight )
{
// The input layer doesn't use weights.
// inputLayer.setRandomWeights()

hiddenLayer.setRandomWeights( maxWeight );
outputLayer.setRandomWeights( maxWeight );
}




private void setInputRow( int row )
{
// Set the input layer neurons (activation
// value) from data matrix.

int col = inputMatrix.getColumns();

// Plus 1 for the bias at zero.
int layerSize = col + 1;
mData.showStatus( "InputLayer size: " + 
                   layerSize );

if( layerSize != inputLayer.getSize())
  {
  throw new Exception( 
             "layerSize != inputLayer size." );
  }

// The bias that isn't used in this row.
inputLayer.setActivationAt( 0, 1.0F );

for( int count = 0; count < col; count++ )
  {
  float val = inputMatrix.getVal( row, count );
  inputLayer.setActivationAt( count + 1, val );
  }
}



/*
To match up with the errorVec?
Or what?
======
private void setLabelRow( int row )
{
int col = labelMatrix.getColumns();

// Plus 1 for the bias at zero.
int layerSize = col + 1;
mData.showStatus( "LabelMatrix layerSize: " + 
                   layerSize );
=======
// The bias that isn't used in this row.
inputLayer.setActivationAt( 0, 1.0F );

for( int count = 0; count < col; count++ )
  {
  float val = inputMatrix.getVal( row, count );
  inputLayer.setActivationAt( count + 1, val );
  }
}
*/



private void forwardPass()
{
hiddenLayer.calcZ( inputLayer );
hiddenLayer.calcActReLU();

outputLayer.calcZ( hiddenLayer );
outputLayer.calcActSigmoid();


/*
// The value at zero is the bias.
// So the one output is at index 1.
float aOut = outputLayer.getActivationAt( 1 );

float testOut = testLabelAr.getVal( 1 );

float error = testOut - aOut;
errorOutAr.setVal( 1, error );

mData.showStatus( "aOut: " + aOut );
mData.showStatus( "testOut: " + testOut );
mData.showStatus( "error: " + error );
*/
}



} // Class
