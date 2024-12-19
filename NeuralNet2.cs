/*

// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



using System;



// namespace




public class NeuralNet2
{
private MainData mData;

// Different layers might have different
// learning rates.

private float learnRate = 0.002F;

private Batch1 batch;
private NeuronLayer2 inputLayer;
private NeuronLayer2 hiddenLayer1;
// private NeuronLayer2 hiddenLayer2;
private NeuronLayer2 outputLayer;
private VectorFlt errorOutVec;




private NeuralNet2()
{
}



internal NeuralNet2( MainData useMainData )
{
mData = useMainData;

inputLayer = new NeuronLayer2( mData );
hiddenLayer1 = new NeuronLayer2( mData );
// hiddenLayer2 = new NeuronLayer2( mData );
outputLayer = new NeuronLayer2( mData );

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

int epoch = 2;

setupNetTopology( columns );

setRandomWeights();

double showMin = 0;

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

  // Also do oneEpochAvg().

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

// while( there is still data... )

int batchSize = batch.getSize();

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

  // Average values would be cleared to
  // zero here before the batch-Matrix
  // operations in what is now a loop.
  // All rows in the matrix would be getting
  // multiplied by the same weight-matrix
  // before that weight-matrix gets
  // changed.

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

    // } matrix loop end.

    // After a future batch-matrix is done
    // being multiplied, the weight and
    // biases would be adjusted after what
    // is now a loop for the batch rows.

    // Here the weights are being adjusted
    // after every batch row.

    outputLayer.adjustBias( learnRate );
    hiddenLayer1.adjustBias( learnRate );
    // hiddenLayer2.adjustBias( learnRate );

    outputLayer.adjustWeights( learnRate );
    hiddenLayer1.adjustWeights( learnRate );
    // hiddenLayer2.adjustWeights( learnRate );

    show3D();

    if( mData.getCancelled())
      return;

    }

  startRow += batchSize;
  }
}



private void show3D()
{
VectorFlt weightVec1 = new VectorFlt( mData );
VectorFlt weightVec2 = new VectorFlt( mData );
outputLayer.copyWeightVecAt( 1, weightVec1 );
outputLayer.copyWeightVecAt( 2, weightVec2 );

mData.setFromWeightVecs( weightVec1,
                         weightVec2 );


}




private void setupNetTopology( int columns )
{
// Check WebPageDct.neuralSearch() for the
// minimum story text length.

// Plus 1 for the bias at zero.
int layerSize = columns + 1;
int hiddenSize = layerSize * 2;

inputLayer.setSize( layerSize );
hiddenLayer1.setSize( hiddenSize );
// hiddenLayer2.setSize( hiddenSize );

// One for the bias at zero, and two more.
outputLayer.setSize( 3 );

inputLayer.setLayers( null, hiddenLayer1 );
hiddenLayer1.setLayers( inputLayer,
                        outputLayer );
                        // hiddenLayer2 );

// hiddenLayer2.setLayers( hiddenLayer1,
//                         outputLayer );

outputLayer.setLayers( hiddenLayer1, null );

errorOutVec.setSize( 3 );
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
// hiddenLayer2.setRandomWeights( randMax );
outputLayer.setRandomWeights( randMax );
}




private void setInputRow( int row )
{
// Set the input layer neurons (activation
// value) from data matrix.

int col = batch.getColumns();

// Plus 1 for the bias at zero.
int layerSize = col + 1;

if( layerSize != inputLayer.getSize())
  {
  throw new Exception(
             "layerSize != inputLayer size." );
  }

for( int count = 0; count < col; count++ )
  {
  float val = batch.getVal( row, count );
  inputLayer.setActivationAt( count + 1, val );
  }
}




private void forwardPass()
{
hiddenLayer1.calcZ();
hiddenLayer1.calcActSigmoid();

// hiddenLayer2.calcZ();
// hiddenLayer2.calcActSigmoid();

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
  float showAct1 = actVec.getVal( 1 );
  float showAct2 = actVec.getVal( 2 );
  float showLabel1 = labelVec.getVal( 1 );
  float showLabel2 = labelVec.getVal( 2 );

  mData.showStatus( " " );
  mData.showStatus( "Act1: " +
                 showAct1.ToString( "N4" ) );
  mData.showStatus( "Act2: " +
                 showAct2.ToString( "N4" ) );
  mData.showStatus( "Label1: " +
                 showLabel1.ToString( "N0" ) );
  mData.showStatus( "Label2: " +
                 showLabel2.ToString( "N0" ) );
  }

// label - act
errorVec.subtract( labelVec, actVec );
errorVec.setVal( 0, 0 );

// For a Cost function:
// float errorNormSqr = errorVec.normSquared();

// delta is dError / dZ.
// Which is (dError / dA) * (dA / dZ)
// That is the Chain Rule for those
// derivatives.

float z1 = outputLayer.getZSumAt( 1 );
float z2 = outputLayer.getZSumAt( 2 );

// The value of derivSigmoid() can be
// between 0 and 0.25.

// dError / dA = dErrorA1
float dErrorA1 = errorVec.getVal( 1 );
float dErrorA2 = errorVec.getVal( 2 );

float delta1 = dErrorA1 *
                      Activ.drvSigmoid( z1 );
float delta2 = dErrorA2 *
                      Activ.drvSigmoid( z2 );

// Delta is on the z side of the activation
// function.  It is the z for this output
// layer.

outputLayer.setDeltaAt( 1, delta1 );
outputLayer.setDeltaAt( 2, delta2 );

// outputLayer.addToDeltaAvgAt( 1, delta1 );
// outputLayer.addToDeltaAvgAt( 2, delta2 );
}




private void setDeltaForHidden()
{
// The later layer goes first.

// hiddenLayer2.setDeltaForHidden();
hiddenLayer1.setDeltaForHidden();
}




} // Class

*/
