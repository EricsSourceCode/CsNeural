// Copyright Eric Chauvin 2021 - 2024.



// This is licensed under the GNU General
// Public License (GPL).  It is the
// same license that Linux has.
// https://www.gnu.org/licenses/gpl-3.0.html





#pragma once


#include "../CppBase/BasicTypes.h"
// #include "../CppBase/FileIO.h"
#include "../Network/SocketsApi.h"
#include "../CryptoBase/Rsa.h"
#include "../Crt/Garner.h"
#include "../Crt/GarnerCrt.h"



class MainApp
  {
  private:
  // Constructors that take a while:
  IntegerMath intMath;
  SPrimes sPrimes; // makeArray in constructor.
  Mod mod;
  QuadRes quadRes;
  MultInv multInv;
  FindFacSm findFacSm; // Makes arrays here.
  // FindFacQr findFacQr;
  Rsa rsa;
  CrtMath crtMath;
  Garner garner;
  GarnerCrt garnerCrt;


  // The constructor for SocketsApi does
  // WSAStartup() and the destructor does
  // WSACleanup(). So the Windows dlls stay
  // going for the life of this app.

  SocketsApi socketsApi;

  public:
  inline static const char* getVersionStr( void )
    {
    return "3/5/2024";
    }

  // Int32 mainLoop( Int32 argc, char* argv[] );
  Int32 mainLoop( void );
  void testSockets( void );
  void testTls( void );

  };
