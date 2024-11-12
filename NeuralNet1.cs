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

// The Greek letter Eta is often used
// to represent the learning rate.
// Different layers might have different
// learning rates.

private float learnRate = 0.1F;

private int batchSize = 50;

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

// One epoch is one complete pass through
// the entire training set.

int epoch = 2;

setupNetTopology( columns );

setRandomWeights();

for( int count = 0; count < epoch; count++ )
  {
  if( !mData.checkEvents())
    return;

  mData.showStatus( "Epoch: " + count );

  // Also do oneEpochAvg().

  if( !oneEpoch( demParagArray,
                 repubParagArray ))
    return; // No more batches.

  }

mData.showStatus( "NeuralNet.test() end." );
}


// Copy this function to do it as an average
// for a batch.  oneEpochAvg()

private bool oneEpoch(
                  VectorArray demParagArray,
                  VectorArray repubParagArray )
{
// Do all batches for one epoch.

// Start row for the entire set of data.

int startRow = 0;

// while( there is still data... )

for( int batchCount = 0; batchCount < 1000;
                         batchCount++ )
  {
  if( !makeOneBatch( startRow,
                     demParagArray,
                     repubParagArray ))
    return false; // No more batches.

  // outputLayer.clearAllDeltaAvg();
  // hiddenLayer.clearAllDeltaAvg();

  for( int rowCount = 0; rowCount < batchSize;
                                    rowCount++ )
    {
    // From the row in the batch array.
    setInputRow( rowCount );
    forwardPass();
    if( !backprop( rowCount ))
      return false;

    // Not doing an average for the batch here.
    // }

    // Adjusting weights and biases comes
    // after the backProp() gets the
    // delta values.

    adjustBias( outputLayer, learnRate );
    adjustBias( hiddenLayer, learnRate );

    if( !adjustWeights(
                   hiddenLayer, // fromLayer
                   outputLayer, // toSetLayer
                   learnRate ))
      return false;

    if( !adjustWeights( inputLayer, // fromLayer
                   hiddenLayer, // toSetLayer
                   learnRate ))
      return false;

    }

  startRow += batchSize;
  }

return true;
}



====
All Ds?  All Rs?  All zeros in activation
makes weights irrelevant.  So it is just bias
So bias must be right?


private void setTestVec( VectorFlt testVec,
                            int testLength,
                            string pattern )
{
string testStr = pattern;

// while( true )
for( int count = 0; count < testLength; count++ )
  {
  testStr += pattern;
  if( testStr.Length >= testLength )
    break;

  }

// mData.showStatus( "Test Vec: " + testStr );

testVec.setFromString( testStr );
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
if( columns != repubParagArray.getColumns())
  {
  throw new Exception(
          "dem and repub columns not equal." );
  }

// If the size is not already set.
batchArray.setSize( batchSize + 2, columns );
labelArray.setSize( batchSize + 2, 3 );

batchArray.clearLastAppend();
labelArray.clearLastAppend();

VectorFlt copyVec = new VectorFlt( mData );
VectorFlt labelVec = new VectorFlt( mData );
VectorFlt testDemVec = new VectorFlt( mData );
VectorFlt testRepubVec = new VectorFlt( mData );

testDemVec.setSize( columns );
testRepubVec.setSize( columns );

setTestVec( testDemVec, columns,
                              "MSNBC News " );
setTestVec( testRepubVec, columns,
                              "Fox News " );

labelVec.setSize( 3 );
labelVec.setVal( 0, 0 ); // Bias is not used.

for( int count = 0; count < (batchSize / 2);
// for( int count = 0; count < batchSize;
                                count++ )
  {
  if( copyAt >= lastRow )
    return false; // Not a full batch.

  demParagArray.copyVecAt( copyVec, copyAt );

  // Test:
  // copyVec.clearTo( 0 );
  copyVec.copy( testDemVec );

  batchArray.appendVecCopy( copyVec );
  labelVec.setVal( 1, 0 ); // Democrat
  labelVec.setVal( 2, 1 );
  labelArray.appendVecCopy( labelVec );


  // Repub:
  repubParagArray.copyVecAt( copyVec, copyAt );

  // Test:
  copyVec.clearTo( 1 );
  // copyVec.copy( testRepubVec );

  batchArray.appendVecCopy( copyVec );

  labelVec.setVal( 1, 1 ); // Republican
  labelVec.setVal( 2, 0 );
  labelArray.appendVecCopy( labelVec );

  copyAt++;
  }

if( labelArray.getLastAppend() != batchArray.
                   getLastAppend())
  {
  throw new Exception(
             "label last != batch last" );
  }

return true;
}




private void setupNetTopology( int columns )
{
// Check WebPageDct.neuralSearch() for the
// minimum story text length.

// Plus 1 for the bias at zero.
int layerSize = columns + 1;
int hiddenSize = layerSize * 2;
inputLayer.setSize( layerSize );

// The weights aren't used here.
inputLayer.setWeightArSize( 1 );


hiddenLayer.setSize( hiddenSize );
hiddenLayer.setWeightArSize( layerSize );

// One for the bias at zero, and two more.
outputLayer.setSize( 3 );

outputLayer.setWeightArSize( hiddenSize );

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



private bool backprop( int row )
{
// The row in the batch array.

if( !setDeltaAtOutput( row ))
  return false;

if( !setDeltaAtHidden( outputLayer,
                       hiddenLayer, row ))
  return false;

return true;
}



private bool setDeltaAtOutput( int row )
{
VectorFlt actVec = new VectorFlt( mData );
VectorFlt labelVec = new VectorFlt( mData );
VectorFlt errorVec = new VectorFlt( mData );

outputLayer.getActivationVec( actVec );
labelArray.copyVecAt( labelVec, row );

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
                 Activation.derivSigmoid( z1 );
float delta2 = dErrorA2 *
                 Activation.derivSigmoid( z2 );

outputLayer.setDeltaAt( 1, delta1 );
outputLayer.setDeltaAt( 2, delta2 );

// outputLayer.addToDeltaAvgAt( 1, delta1 );
// outputLayer.addToDeltaAvgAt( 2, delta2 );

return true;
}



// setDeltaAtHidden( outputLayer,
//                   hiddenLayer, row )

private bool setDeltaAtHidden(
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
  // Big exponential loops.

  // if( (weightAt % 10) == 0 )
    {
    if( !mData.checkEvents())
      return false;
    }

  // weightAt is also the index of the neuron
  // in the layer that is about to be set.

  float sumToSet = 0;

  for( int fromNeuron = 1;
           fromNeuron < maxFrom; fromNeuron++ )
    {
    float deltaFrom = fromLayer.getDeltaAt(
                                  fromNeuron );

    // Here is a Matrix.  Row and column.

    // If the fromNeuron is in the output
    // layer, then weightAt refers to which
    // neuron the weight refers to in the
    // hidden layer.  This would be throwing
    // an exception if the rows/columns
    // were backwards here.

    float weight = fromLayer.getWeight(
                       fromNeuron, weightAt );

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

  // toSetLayer.addToDeltaAvgAt( weightAt,
  //                            sumToSet );
  }

return true;
}




private void adjustBias( NeuronLayer1 layer,
                         float rate )
{
// dCost / dBias = delta

int max = layer.getSize();

for( int count = 1; count < max; count++ )
  {
  float delta = layer.getDeltaAt( count );
  // float delta = layer.getDeltaAvgAt( count );
  // delta = delta / batchSize;

  float bias = layer.getBias( count );
  float biasAdj = delta * rate;
  bias += biasAdj;
  layer.setBias( count, bias );
  }
}




private bool adjustWeights(
                    NeuronLayer1 fromLayer,
                    NeuronLayer1 toSetLayer,
                    float rate )
{
// dError / dW = activation * delta

int maxFrom = fromLayer.getSize();
int maxToSet = toSetLayer.getSize();

for( int countFrom = 1; countFrom < maxFrom;
                        countFrom++ )
  {
  if( (countFrom % 10) == 0 )
    {
    if( !mData.checkEvents())
      return false;
    }

  // The activation of the neuron
  // in the from layer, the input layer,
  // which is going to output to the
  // toSet neuron through the weight.

  float act = fromLayer.
               getActivationAt( countFrom );

  for( int countToSet = 1;
            countToSet < maxToSet; countToSet++ )
    {
    float delta = toSetLayer.getDeltaAt(
                              countToSet );
    // float delta = toSetLayer.getDeltaAvgAt(
    //                            countToSet );
    // delta = delta / batchSize;

    // CountToSet is the neuron, countFrom
    // is where the weight is in that
    // neuron's array.

    float weight = toSetLayer.getWeight(
                 countToSet, countFrom );

    float wAdjust = rate * delta * act;
    weight += wAdjust;
    toSetLayer.setWeight( countToSet, countFrom,
                          weight );
    }
  }

return true;
}




} // Class
