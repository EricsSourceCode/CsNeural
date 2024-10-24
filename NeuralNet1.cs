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

// A byte could have a value up to 255.
// But most letters are 97 'a' to 122 'z'.

randMax = randMax / 100.0F;

mData.showStatus( "randMax: " +
                    randMax.ToString( "N8" ));

setRandomWeights( randMax );

// Different layers might have different
// step sizes.

// The stepSize is the learning rate.
float stepSize = 0.05F;

// An epoch is one complete pass through
// the entire training set.

int epoch = 2;
int maxrow = 500;
int batchSize = 20; // Mini-batch
int maxLabel = labelMatrix.getLastAppend();
if( maxrow >= maxLabel )
  maxrow = maxLabel - 1;

for( int count = 0; count < epoch; count++ )
  {
  if( !mData.checkEvents())
    return;

  mData.showStatus( " " );
  int dems = 0;
  int repub = 0;
  int batchCount = 0;
  outputLayer.clearAllDeltaAvg();
  hiddenLayer.clearAllDeltaAvg();

  for( int row = 0; row < maxrow; row++ )
    {
    // if( labelMatrix.getVal( row, 1 ) == 1)
      // continue; // Repub only.

    if( labelMatrix.getVal( row, 1 ) == 1)
      dems++;
    else
      repub++;

    setInputRow( row );
    forwardPass( row );
    backprop( row, maxrow );

    batchCount++;
    if( batchCount > batchSize )
      {
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
      }
    }

  mData.showStatus( "dems: " + dems );
  mData.showStatus( "repub: " + repub );
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
// mData.showStatus( "InputLayer size: " +
//                   layerSize );

if( layerSize != inputLayer.getSize())
  {
  throw new Exception(
             "layerSize != inputLayer size." );
  }

for( int count = 0; count < col; count++ )
  {
  float val = inputMatrix.getVal( row, count );
  inputLayer.setActivationAt( count + 1, val );
  // if( row < 3 )
    // {
    // mData.showStatus( "Input: " +
    //                   val.ToString( "N1" ));
    //}
  }
}




private void forwardPass( int row )
{
hiddenLayer.calcZ( inputLayer );
hiddenLayer.calcActReLU();

outputLayer.calcZ( hiddenLayer );
outputLayer.calcActSigmoid();
}



private void backprop( int row, int maxrow )
{
setDeltaAtOutput( row, maxrow );
setDeltaAtHidden( outputLayer,
                  hiddenLayer, row );
}



private void setDeltaAtOutput( int row,
                               int maxrow )
{
VectorFlt actVec = new VectorFlt( mData );
outputLayer.getActivationVec( actVec );

// The value at zero is the bias.
float aOut1 = outputLayer.getActivationAt( 1 );
float aOut2 = outputLayer.getActivationAt( 2 );

VectorFlt labelVec = new VectorFlt( mData );
labelMatrix.copyVecAt( labelVec, row );

VectorFlt errorVec = new VectorFlt( mData );

// label - act
errorVec.subtract( labelVec, actVec );
errorVec.setVal( 0, 0 );
float errorNormSqr = errorVec.normSquared();

float label1 = labelMatrix.getVal( row, 1 );
float label2 = labelMatrix.getVal( row, 2 );

// delta is dError / dZ.
// Which is (dError / dA) * (dA / dZ)
// That is the Chain Rule for those
// derivatives.


// dError / dA = dErrorA1
float dErrorA1 = label1 - aOut1;
float dErrorA2 = label2 - aOut2;

if( (row < 5) || (row > (maxrow - 5)))
  {
  mData.showStatus( " " );
  mData.showStatus( "aOut1: " + aOut1 );
  mData.showStatus( "aOut2: " + aOut2 );
  mData.showStatus( "label1: " + label1 );
  mData.showStatus( "label2: " + label2 );
  // mData.showStatus( "dErrorA1: " +
  //                dErrorA1.ToString( "N4" ));
  // mData.showStatus( "dErrorA2: " +
  //                dErrorA2.ToString( "N4" ));
  }

float errorSqr = (dErrorA1 * dErrorA1) +
                 (dErrorA2 * dErrorA2);

if( errorSqr != errorNormSqr )
  throw new Exception( "errorNormSqr" );

// mData.showStatus( "errorSqr: " +
//                   errorSqr.ToString( "N6" ));

float z1 = outputLayer.getZSumAt( 1 );
float z2 = outputLayer.getZSumAt( 2 );


// The value of derivSigmoid() can be
// between 0 and 0.25.


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

==== So then what?
  toSetLayer.addToDeltaAvgAt( weightAt, 
                              sumToSet );

  // mData.showStatus( "To set delta: " +
  //              sumToSet.ToString( "N4" ) );
  }
}



// The old way.
private void adjustBias( NeuronLayer1 layer,
                         float stepSize )
{
// dCost / dBias = delta

int max = layer.getSize();

for( int count = 1; count < max; count++ )
  {
  float delta = layer.getDeltaAt( count );
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




// The old way.
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

    float delta = toSetLayer.getDeltaAt(
                                 countToSet );
    float weight = toSetLayer.getWeight(
                 countToSet, countFrom );

    // The Greek letter Eta is often used
    // to represent the stepSize.

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
