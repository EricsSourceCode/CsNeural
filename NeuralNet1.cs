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

// The stepSize is the learning rate.
// The Greek letter Eta is often used
// to represent the stepSize.
// Different layers might have different
// step sizes.

// float stepSize = 0.05F;

// An epoch is one complete pass through
// the entire training set.

int epoch = 2;
int batchSize = 40; // Mini-batch


private NeuronLayer1 inputLayer;
private NeuronLayer1 hiddenLayer;
private NeuronLayer1 outputLayer;
private VectorFlt errorOutVec;
private VectorArray labelArray;
private VectorArray batchArray;




private NeuralNet1()
{
}



internal NeuralNet1( MainData useMainData )
{
mData = useMainData;

inputLayer = new NeuronLayer1( mData );
hiddenLayer = new NeuronLayer1( mData );
outputLayer = new NeuronLayer1( mData );
errorOutVec = new VectorFlt( mData );
labelArray = new VectorArray( mData );
batchArray = new VectorArray( mData );
}



internal void test( VectorArray demParagArray,
                    VectorArray repubParagArray )
{
mData.showStatus( "NeuralNet1.test()." );

int columns = demParagArray.getColumns();
if( columns != repubParagArray.getColumns() )
  {
  throw new Exception(
           "dem and repub columns." );
  }

setupNetTopology( columns );

setRandomWeights();

for( int count = 0; count < epoch; count++ )
  {
  if( !mData.checkEvents())
    return;

  mData.showStatus( "Epoch: " + count );

  if( !oneEpoch( demParagArray,
                 repubParagArray ))
    return; // No more batches.

  }

mData.showStatus( "NeuralNet.test() end." );
}




private bool oneEpoch(
                  VectorArray demParagArray,
                  VectorArray repubParagArray )
{
// Do all batches for one epoch.

// Start for the entire set of data.

int startRow = 0;

// Make this wCount go to a huge number
// so it just runs out of batches.

for( int wCount = 0; wCount < 3;
                         wCount++ )
  {
  mData.showStatus( " " );
  mData.showStatus( "Batch: " + wCount );

  if( !makeOneBatch( startRow,
                     demParagArray,
                     repubParagArray ))
    return false; // No more batches.

  outputLayer.clearAllDeltaAvg();
  hiddenLayer.clearAllDeltaAvg();

  // makeOneBatch() only makes full size
  // batches.

  for( int count = 0; count < batchSize; count++ )
    {
    // labelArray.getVal( count, 1 )

    // From the row in the batch array.
    setInputRow( count );
    forwardPass();
    backprop( count );
    }

/*
======
  adjustBias( outputLayer, stepSize );
      adjustBias( hiddenLayer, stepSize );
      adjustWeights( hiddenLayer, // fromLayer
                     outputLayer, // toSetLayer
                     stepSize );

      adjustWeights( inputLayer, // fromLayer
                     hiddenLayer, // toSetLayer
                     stepSize );

      // Clear to zero to start new mini-batch.
      outputLayer.clearAllDeltaAvg();
      hiddenLayer.clearAllDeltaAvg();
  */

  startRow += batchSize;
  }

return true;
}




private bool makeOneBatch( int startAt,
                   VectorArray demParagArray,
                   VectorArray repubParagArray )
{
int copyAt = startAt;
int lastRow = demParagArray.getLastAppend();
int lastRepubRow = repubParagArray.
                          getLastAppend();
if( lastRepubRow < lastRow )
  lastRow = lastRepubRow;

int columns = demParagArray.getColumns();

// If the size is not already set.
batchArray.setSize( batchSize + 2, columns );
labelArray.setSize( batchSize + 2, 3 );

batchArray.clearLastAppend();
labelArray.clearLastAppend();

VectorFlt copyVec = new VectorFlt( mData );

VectorFlt labelVec = new VectorFlt( mData );
labelVec.setSize( 3 );
labelVec.setVal( 0, 0 ); // Bias is not used.

for( int count = 0; count < (batchSize / 2);
                                count++ )
  {
  if( copyAt >= lastRow )
    return false; // Not a full batch.

  demParagArray.copyVecAt( copyVec, copyAt );
  batchArray.appendVecCopy( copyVec );
  labelVec.setVal( 1, 0 ); // Democrat
  labelVec.setVal( 2, 1 );
  labelArray.appendVecCopy( labelVec );

  repubParagArray.copyVecAt( copyVec, copyAt );
  batchArray.appendVecCopy( copyVec );
  labelVec.setVal( 1, 1 ); // Republican
  labelVec.setVal( 2, 0 );
  labelArray.appendVecCopy( labelVec );

  copyAt++;
  }

return true;
}




private void setupNetTopology( int columns )
{
// Plus 1 for the bias at zero.
int layerSize = columns + 1;

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

hiddenLayer.setRandomWeights( randMax );
outputLayer.setRandomWeights( randMax );
}



private void setInputRow( int row )
{
// Set the input layer neurons (activation
// value) from data matrix.

int col = batchArray.getColumns();

// Plus 1 for the bias at zero.
int layerSize = col + 1;

if( layerSize != inputLayer.getSize())
  {
  throw new Exception(
             "layerSize != inputLayer size." );
  }

for( int count = 0; count < col; count++ )
  {
  float val = batchArray.getVal( row, count );
  inputLayer.setActivationAt( count + 1, val );
  }
}




private void forwardPass()
{
hiddenLayer.calcZ( inputLayer );
hiddenLayer.calcActReLU();

outputLayer.calcZ( hiddenLayer );
outputLayer.calcActSigmoid();
}



private void backprop( int row )
{
setDeltaAtOutput( row );
setDeltaAtHidden( outputLayer,
                  hiddenLayer, row );
}



private void setDeltaAtOutput( int row )
{
VectorFlt actVec = new VectorFlt( mData );
outputLayer.getActivationVec( actVec );

VectorFlt labelVec = new VectorFlt( mData );
labelArray.copyVecAt( labelVec, row );

VectorFlt errorVec = new VectorFlt( mData );

// label - act
errorVec.subtract( labelVec, actVec );
errorVec.setVal( 0, 0 );


// For a Cost function:
// float errorNormSqr = errorVec.normSquared();

// delta is dError / dZ.
// Which is (dError / dA) * (dA / dZ)
// That is the Chain Rule for those
// derivatives.
// dError / dA = dErrorA1


float z1 = outputLayer.getZSumAt( 1 );
float z2 = outputLayer.getZSumAt( 2 );


// The value of derivSigmoid() can be
// between 0 and 0.25.


// dError / dA = dErrorA1
float dErrorA1 = errorVec.getVal( 1 );
float dErrorA2 = errorVec.getVal( 2 );

float delta1 = dErrorA1 *
                 Activation.derivSigmoid( z1 );
float delta2 = dErrorA2 *
                 Activation.derivSigmoid( z2 );

// mData.showStatus( "delta1: " +
//                 delta1.ToString( "N4" ) );
// mData.showStatus( "delta2: " +
//                 delta2.ToString( "N4" ) );

outputLayer.setDeltaAt( 1, delta1 );
outputLayer.setDeltaAt( 2, delta2 );

outputLayer.addToDeltaAvgAt( 1, delta1 );
outputLayer.addToDeltaAvgAt( 2, delta2 );
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

  // mData.showStatus( " " );
  // mData.showStatus( "weightAt: " + weightAt );

  float sumToSet = 0;

  for( int fromNeuron = 1;
           fromNeuron < maxFrom; fromNeuron++ )
    {
    // mData.showStatus( "  fromNeuron: " +
    //                           fromNeuron );

    float deltaFrom = fromLayer.getDeltaAt(
                                  fromNeuron );

    // mData.showStatus( "  deltaFrom: " +
    //             deltaFrom.ToString( "N4" ) );

    // What is the weight between these
    // two neurons?
    // Here is a Matrix.  Row and column.

    float weight = fromLayer.getWeight(
                       fromNeuron, weightAt );

    // mData.showStatus( "  weight: " +
    //               weight.ToString( "N4" ) );

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

  toSetLayer.addToDeltaAvgAt( weightAt,
                              sumToSet );

  // mData.showStatus( "To set delta: " +
  //              sumToSet.ToString( "N4" ) );
  }
}




private void adjustBias( NeuronLayer1 layer,
                         float stepSize )
{
// dCost / dBias = delta

int max = layer.getSize();

for( int count = 1; count < max; count++ )
  {
  // float delta = layer.getDeltaAt( count );
  float delta = layer.getDeltaAvgAt( count );
  delta = delta / batchSize;

  float bias = layer.getBias( count );
  float biasAdj = delta * stepSize;
  // mData.showStatus( "biasAdj: " +
  //                 biasAdj.ToString( "N4" ));

  bias += biasAdj;

  layer.setBias( count, bias );
  }
}




// If you are calculating with matrices, which
// happens in more optimized code in later
// versions, then you'd calculate all
// derivatives at once on a mini-batch.
// You can't change any weights in the network
// while you are doing that.  So with the
// matrix results for a whole mini-batch,
// you'd average rates of change for that
// batch.




private void adjustWeights(
                    NeuronLayer1 fromLayer,
                    NeuronLayer1 toSetLayer,
                    float stepSize )
{
// dError / dW = activation * delta

int maxFrom = fromLayer.getSize();
int maxToSet = toSetLayer.getSize();

for( int countFrom = 1; countFrom < maxFrom;
                        countFrom++ )
  {
  // The is the activation of the neuron
  // in the from layer, the input layer,
  // which is going to output to the
  // toSet neuron through the weight.

  float act = fromLayer.
               getActivationAt( countFrom );

  for( int countToSet = 1;
            countToSet < maxToSet; countToSet++ )
    {

    // float delta = toSetLayer.getDeltaAt(
    //                          countToSet );
    float delta = toSetLayer.getDeltaAvgAt(
                                countToSet );
    delta = delta / batchSize;


    float weight = toSetLayer.getWeight(
                 countToSet, countFrom );

    float wAdjust = stepSize * delta * act;
    // mData.showStatus( "wAdjust: " +
    //               wAdjust.ToString( "N4" ));

    weight += wAdjust;
    toSetLayer.setWeight( countToSet, countFrom,
                          weight );
    }
  }
}




} // Class
