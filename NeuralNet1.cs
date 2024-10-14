// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



using System;



// namespace




public class NeuralNet1
{
private MainData mData;
private NeuronLayer1 inputLayer;
private NeuronLayer1 hiddenLayer;
private NeuronLayer1 outputLayer;
private VectorFlt errorOutVec;
private VectorArray inputMatrix;
private VectorArray labelMatrix;




private NeuralNet1()
{
}



internal NeuralNet1( MainData useMainData,
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

inputLayer = new NeuronLayer1( mData );
hiddenLayer = new NeuronLayer1( mData );
outputLayer = new NeuronLayer1( mData );
errorOutVec = new VectorFlt( mData );
}



internal void test()
{
mData.showStatus( "NeuralNet1.test()." );

setupNetTopology();

float randMax = 1.0F / inputLayer.getSize();
mData.showStatus( "randMax: " + randMax );

setRandomWeights( randMax );

for( int row = 0; row < 1; row++ )
  {
  setInputRow( row );
  forwardPass( row );
  backprop( row );

  // adjustBiasAt( NeuronLayer1 layer );

  // adjustWeightsAt( outputLayer );
  // adjustWeightsAt( hiddenLayer );
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
}



private void backprop( int row )
{
mData.showStatus( " " );
mData.showStatus( "Top of backprop()." );

// This is done after each forward pass
// using specific values of Activation
// ZSum, etc.

setDeltaAtOutput( row );
setDeltaAtHidden( outputLayer,
                  hiddenLayer, row );



mData.showStatus( "End of backprop()." );
}



private void setDeltaAtOutput( int row )
{
// The value at zero is the bias.
float aOut1 = outputLayer.getActivationAt( 1 );
float aOut2 = outputLayer.getActivationAt( 2 );

mData.showStatus( "aOut1: " + aOut1 );
mData.showStatus( "aOut2: " + aOut2 );

float label1 = labelMatrix.getVal( row, 1 );
float label2 = labelMatrix.getVal( row, 2 );

mData.showStatus( "label1: " + label1 );
mData.showStatus( "label2: " + label2 );

// delta is dError / dZ.
// Which is (dError / dA) * (dA / dZ)
// That is the Chain Rule for those
// derivatives.

// What are the different ways a Cost
// function could be done?

// The sign of the error makes it so that
// you change the weights in the right
// direction.  To make the errors smaller.
// The tutorial shows this as: aOut1 - label1
// The tutorial is showing the negative of
// the gradient vector.  To go downhill.

// But the book shows it like this:
// dError / dA = dErrorA1
float dErrorA1 = label1 - aOut1;
float dErrorA2 = label2 - aOut2;

float z1 = outputLayer.getZSumAt( 1 );
float z2 = outputLayer.getZSumAt( 2 );


/////////
// Test:
float dSigmoid = Activation.derivSigmoid( z1 );
// The value of deriveSigmoid() can be
// between 0 and 0.25.  A small number.
if( (dSigmoid < 0) || (dSigmoid > 0.25))
  throw new Exception( "dSigmoid range." );
///////////


float delta1 = dErrorA1 *
                 Activation.derivSigmoid( z1 );
float delta2 = dErrorA2 *
                 Activation.derivSigmoid( z2 );

mData.showStatus( "delta1: " + delta1 );
mData.showStatus( "delta2: " + delta2 );

outputLayer.setDeltaAt( 1, delta1 );
outputLayer.setDeltaAt( 2, delta2 );
}




private void setDeltaAtHidden(
                      NeuronLayer1 fromLayer,
                      NeuronLayer1 toSetLayer,
                      int row )
{
int maxToSet = toSetLayer.getSize();
int maxFrom = fromLayer.getSize();

// Start counting at 1 because the bias
// is the only thing at zero.
for( int weightAt = 1;
           weightAt < maxToSet; weightAt++ )
  {
  // weightAt is also the index of the neuron
  // in the layer that is about to be set.

  mData.showStatus( " " );
  mData.showStatus( "weightAt: " + weightAt );

  float sumToSet = 0;

  for( int fromNeuron = 1;
           fromNeuron < maxFrom; fromNeuron++ )
    {
    mData.showStatus( "  fromNeuron: " +
                                fromNeuron );

    float deltaFrom = fromLayer.getDeltaAt(
                                  fromNeuron );

    mData.showStatus( "  deltaFrom: " +
                                  deltaFrom );

    // What is the weight between these
    // two neurons?
    // Here is a Matrix.  Row and column.

    float weight = fromLayer.getWeight(
                       fromNeuron, weightAt );

    mData.showStatus( "  weight: " + weight );

    // If weight and deltaFrom were both
    // negative then sumToSet would have a
    // positive number added to it.  And
    // other +/- variations like that.

    // This is the z from the neuron I
    // am about to set.
    float z = toSetLayer.getZSumAt( weightAt );

    // If z <= 0 then the output from this
    // neuron activation would be zero.
    // So it would have nothing to do
    // with the output.
    // If z was <= 0 then derivReLu
    // would be zero.  So partSum would
    // be zero.

    // This would be a Hadamard Product for
    // the vectors:
    // result.hadamard( weight, deltaFrom ).
    // Then add the vector for ReLU( z ).

    float partSum = (weight * deltaFrom) *
                  Activation.derivReLU( z );

    sumToSet += partSum;
    }

  // Set the delta for the neuron in the
  // toSetLayer.
  toSetLayer.setDeltaAt( weightAt, sumToSet );
  mData.showStatus( "To set delta: " + sumToSet );
  }
}



private void adjustBias( NeuronLayer1 layer )
{
// dCost / dBias = delta

// This could be a parameter.  Different layers
// might have different step sizes.

/*
float stepSize = 0.1F;

int max = layer.getSize();

for( int count = 0; count < max; count++ )
  {
  // dError / dW = activation * delta
  float act = layer.getActivationAt( count );
  float delta = layer.getDeltaAt( count );

// float bias = layer.getWeight( int neuron, int where )

  }
*/
}




/*
private void adjustWeightsAt(
                    NeuronLayer1 fromLayer )
                    NeuronLayer1 toSetLayer )
{
// dError / dW = activation * delta

// This could be a parameter.  Different layers
// might have different step sizes.

float stepSize = 0.1;

int max = layer.getSize();

======= Starting at 0 =======

for( int count = 0; count < max; count++ )
  {
  // dError / dW = activation * delta
  float act = layer.getActivationAt( count );
  float delta = layer.getDeltaAt( count );
  float adjust = delta * act;
=====
Weight += stepSize * delta * activation.

  }
}
*/



} // Class
