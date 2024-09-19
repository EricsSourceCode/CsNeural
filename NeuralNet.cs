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
private VectorFlt errorOutVec;
private VectorArray inputMatrix;
private VectorArray labelMatrix;




private NeuralNet()
{
}



internal NeuralNet( MainData useMainData,
                  VectorArray useMatrix,
                  VectorArray useLabelMatrix )
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
errorOutVec = new VectorFlt( mData );
}



internal void test()
{
mData.showStatus( "NeuralNet.test()." );

setupNetTopology();

float randMax = 1.0F / inputLayer.getSize();
mData.showStatus( "randMax: " + randMax );

setRandomWeights( randMax );

for( int row = 0; row < 1; row++ )
  {
  setInputRow( row );
  forwardPass( row );
  backprop();
  }

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

// One for the bias at zero, and two more.
outputLayer.setSize( 3 );

outputLayer.setWeightArSize( layerSize );

errorOutVec.setSize( 3 );
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



private void forwardPass( int row )
{
mData.showStatus( " " );
mData.showStatus( "forwardPass(): " + row );

hiddenLayer.calcZ( inputLayer );
hiddenLayer.calcActReLU();

outputLayer.calcZ( hiddenLayer );
outputLayer.calcActSigmoid();

// The value at zero is the bias.
float aOut1 = outputLayer.getActivationAt( 1 );
float aOut2 = outputLayer.getActivationAt( 2 );

mData.showStatus( "aOut1: " + aOut1 );
mData.showStatus( "aOut2: " + aOut2 );

float label1 = labelMatrix.getVal( row, 1 );
float label2 = labelMatrix.getVal( row, 2 );

mData.showStatus( "label1: " + label1 );
mData.showStatus( "label2: " + label2 );

// If the output value is less than the label
// value then it is positive.  Meaning the
// weights have to be adjusted up.
// If the output value is more than the
// label value then it is negative, so
// the weights have to be adjusted down.
// (Adding a negative number.)

float error1 = label1 - aOut1;
float error2 = label2 - aOut2;

mData.showStatus( "error1: " + error1 );
mData.showStatus( "error2: " + error2 );

errorOutVec.setVal( 1, error1 );
errorOutVec.setVal( 2, error2 );
}



private void backprop()
{
mData.showStatus( " " );
mData.showStatus( "backprop(): " );

/*
// y - y(hat)
float error1 = errorOutVec.getVal( 1 );
*/

mData.showStatus( "End of backprop()." );
}



} // Class
