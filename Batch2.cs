/*

// Copyright Eric Chauvin 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html



using System;



// namespace



// A mini batch.

// There are a lot of different ways to make
// a batch.



public class Batch1
{
private MainData mData;
private const int batchSize = 50;
private VectorArray labelArray;
private VectorArray batchArray;
private VectorFlt copyVec;
private VectorFlt labelVec;
private VectorFlt testDemVec;
private VectorFlt testRepubVec;



private Batch1()
{
}



internal Batch1( MainData useMainData )
{
mData = useMainData;

labelArray = new VectorArray( mData );
batchArray = new VectorArray( mData );
copyVec = new VectorFlt( mData );
labelVec = new VectorFlt( mData );
testDemVec = new VectorFlt( mData );
testRepubVec = new VectorFlt( mData );
}



internal int getSize()
{
return batchSize;
}


internal int getColumns()
{
return batchArray.getColumns();
}


internal void copyLabelVecAt(
                       VectorFlt labelVec,
                       int row )
{
labelArray.copyVecAt( labelVec, row );
}



internal bool makeOneBatch( int startAt,
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

// The batch array is going to become a
// Matrix in later versions.

// If the size is not already set.
batchArray.setSize( batchSize + 2, columns );
labelArray.setSize( batchSize + 2, 3 );

batchArray.clearLastAppend();
labelArray.clearLastAppend();

testDemVec.setSize( columns );
testRepubVec.setSize( columns );

setTestVec( testDemVec, columns,
                              "MSNBC News " );
setTestVec( testRepubVec, columns,
                              "Fox News " );

labelVec.setSize( 3 );
labelVec.setVal( 0, 0 ); // Bias is not used.

for( int count = 0; count < (batchSize / 2);
                                count++ )
  {
  if( copyAt >= lastRow )
    return false; // Not a full batch.

  demParagArray.copyVecAt( copyVec, copyAt );

  // Copy a simple pattern for part of it.

  int testCopySize = columns - 800;

  // Test:
  copyVec.copyUpTo( testDemVec, testCopySize );

  batchArray.appendVecCopy( copyVec );
  labelVec.setVal( 1, 0 ); // Democrat
  labelVec.setVal( 2, 1 );
  labelArray.appendVecCopy( labelVec );


  // Repub:
  repubParagArray.copyVecAt( copyVec, copyAt );

  // Test:
  copyVec.copyUpTo( testRepubVec, testCopySize );

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



internal float getVal( int row, int column )
{
return batchArray.getVal( row, column );
}



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



} // Class

*/
