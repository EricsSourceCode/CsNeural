// Copyright Eric Chauvin 2024 - 2025.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



using System;



// namespace




public class NeuralNet1
{
private MainData mData;

// Different layers might have different
// learning rates.

private float learnRate = 0.002F;

private Batch1 batch;
private NeuronLayer1 inputLayer;
private NeuronLayer1 hiddenLayer1;
private NeuronLayer1 outputLayer;
private VectorFlt errorOutVec;




private NeuralNet1()
{
}



internal NeuralNet1( MainData useMainData )
{
mData = useMainData;

inputLayer = new NeuronLayer1( mData );
hiddenLayer1 = new NeuronLayer1( mData );
outputLayer = new NeuronLayer1( mData );

errorOutVec = new VectorFlt( mData );
batch = new Batch1( mData );
}



internal void test( VectorArray demParagArray,
                    VectorArray repubParagArray )
{
mData.showStatus( "NeuralNet1.test()." );

TimeEC startTime = new TimeEC();
startTime.setToNow();

int columns = demParagArray.getColumns();
if( columns != repubParagArray.getColumns() )
  {
  throw new Exception(
           "dem and repub columns." );
  }

// One epoch is one complete pass through
// the entire training set.

int epoch = 200;

setupNetTopology( columns );

setRandomWeights();

double showMin = 0; // Show minutes.

for( int count = 0; count < epoch; count++ )
  {
  showMin = startTime.getMinutesToNow();
  mData.showStatus( "Minutes: " +
                    showMin.ToString( "N2" ) );

  if( !mData.checkEvents())
    {
    showMin = startTime.getMinutesToNow();
    mData.showStatus( "Minutes: " +
                     showMin.ToString( "N2" ) );
    return;
    }

  mData.showStatus( "Epoch: " + count );

  oneEpoch( demParagArray, repubParagArray );
  if( mData.getCancelled())
    {
    showMin = startTime.getMinutesToNow();
    mData.showStatus( "Minutes: " +
                    showMin.ToString( "N2" ) );
    return;
    }
  }

showMin = startTime.getMinutesToNow();
mData.showStatus( "Minutes: " +
                showMin.ToString( "N2" ));

mData.showStatus( "NeuralNet.test() end." );
}




private void oneEpoch(
                  VectorArray demParagArray,
                  VectorArray repubParagArray )
{
// Do all batches for one epoch.

// Start row for the entire set of data.

int startRow = 0;

int batchSize = batch.getSize();

// while( there is still data... )
for( int batchCount = 0; batchCount < 1000;
                         batchCount++ )
  {
  // The batch array is going to become a
  // Matrix in later versions.  And averaging
  // some values doesn't come in to this
  // until that batch is a Matrix.

  if( !batch.makeOneBatch( startRow,
                     demParagArray,
                     repubParagArray ))
    return; // No more batches.

  for( int rowCount = 0; rowCount < batchSize;
                                    rowCount++ )
    {
    // From the row in the batch array.
    setInputRow( rowCount );
    forwardPass();
    backprop( rowCount );

    if( mData.getCancelled())
      return;

    // Adjusting weights and biases comes
    // after the backProp() sets the
    // delta values.

    // Here the weights are being adjusted
    // after every batch row.

    outputLayer.adjustBias( learnRate );
    hiddenLayer1.adjustBias( learnRate );

    outputLayer.adjustWeights( learnRate );
    hiddenLayer1.adjustWeights( learnRate );

    show3D();

    if( mData.getCancelled())
      return;

    }

  startRow += batchSize;
  }
}



private void show3D()
{
VectorFlt weightVec0 = new VectorFlt( mData );
VectorFlt weightVec1 = new VectorFlt( mData );
VectorFlt biasVec1 = new VectorFlt( mData );

outputLayer.copyWeightVecAt( 0, weightVec0 );
outputLayer.copyWeightVecAt( 1, weightVec1 );

outputLayer.copyBiasVec( biasVec1 );
// hiddenLayer1.copyBiasVec( biasVec2 );

mData.setFromWeightVecs( weightVec0,
                         weightVec1 );

mData.setFromBiasVec1( biasVec1 );

}



private void setupNetTopology( int columns )
{
// Check WebPageDct.neuralSearch() for the
// minimum story text length.


int layerSize = columns;
int hiddenSize = layerSize * 2;

inputLayer.setSize( layerSize );
hiddenLayer1.setSize( hiddenSize );

outputLayer.setSize( 2 );

inputLayer.setLayers( null, hiddenLayer1 );
hiddenLayer1.setLayers( inputLayer,
                        outputLayer );

outputLayer.setLayers( hiddenLayer1, null );

errorOutVec.setSize( 2 );
}




private void setRandomWeights()
{
float randMax = 1.0F / inputLayer.getSize();

// A byte could have a value up to 255.
// But most letters are 97 'a' to 122 'z'.

randMax = randMax / 100.0F;

mData.showStatus( "randMax: " +
                    randMax.ToString( "N8" ));

// The input layer doesn't use weights.
// inputLayer.setRandomWeights()

hiddenLayer1.setRandomWeights( randMax );
outputLayer.setRandomWeights( randMax );
}




private void setInputRow( int row )
{
// Set the input layer neurons (activation
// value) from data matrix.

int col = batch.getColumns();

int layerSize = col;

if( layerSize != inputLayer.getSize())
  {
  throw new Exception(
             "layerSize != inputLayer size." );
  }

for( int count = 0; count < col; count++ )
  {
  float val = batch.getVal( row, count );
  inputLayer.setActivationAt( count, val );
  }
}




private void forwardPass()
{
hiddenLayer1.calcZ();
hiddenLayer1.calcActSigmoid();

outputLayer.calcZ();
outputLayer.calcActSigmoid();
}



private void backprop( int row )
{
// The row in the batch array.

setDeltaForOutput( row );
setDeltaForHidden();
}



private void setDeltaForOutput( int row )
{
VectorFlt actVec = new VectorFlt( mData );
VectorFlt labelVec = new VectorFlt( mData );
VectorFlt errorVec = new VectorFlt( mData );

outputLayer.getActivationVec( actVec );
batch.copyLabelVecAt( labelVec, row );

if( row < 2 )
  {
  float showAct0 = actVec.getVal( 0 );
  float showAct1 = actVec.getVal( 1 );
  float showLabel0 = labelVec.getVal( 0 );
  float showLabel1 = labelVec.getVal( 1 );

  mData.showStatus( " " );
  mData.showStatus( "Act0: " +
                 showAct0.ToString( "N4" ) );
  mData.showStatus( "Act1: " +
                 showAct1.ToString( "N4" ) );
  mData.showStatus( "Label0: " +
                 showLabel0.ToString( "N0" ) );
  mData.showStatus( "Label1: " +
                 showLabel1.ToString( "N0" ) );
  }

// label - act
errorVec.subtract( labelVec, actVec );

// For a Cost function:
// float errorNormSqr = errorVec.normSquared();

// delta is dError / dZ.
// Which is (dError / dA) * (dA / dZ)
// That is the Chain Rule for those
// derivatives.

float z0 = outputLayer.getZSumAt( 0 );
float z1 = outputLayer.getZSumAt( 1 );

// The value of derivSigmoid() can be
// between 0 and 0.25.

// dError / dA = dErrorA1
float dErrorA0 = errorVec.getVal( 0 );
float dErrorA1 = errorVec.getVal( 1 );

float delta0 = dErrorA0 *
                      Activ.drvSigmoid( z0 );
float delta1 = dErrorA1 *
                      Activ.drvSigmoid( z1 );

// Delta is on the z side of the activation
// function.  It is the z for this output
// layer.

outputLayer.setDeltaAt( 0, delta0 );
outputLayer.setDeltaAt( 1, delta1 );
}




private void setDeltaForHidden()
{
hiddenLayer1.setDeltaForHidden();
}




} // Class
