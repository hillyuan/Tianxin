// @HEADER
// *******************************************************************
//
//           TianXin: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//  Copyright (2022) YUAN Xi
// ******************************************************************* 
// @HEADER

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include "Panzer_STK_Interface.hpp"
#include "PanzerSTK_UnitTest_BuildMesh.hpp"

#include <string>

/*
   Unit test for function STK_Interface::PeriodicGhosting
*/

TEUCHOS_UNIT_TEST(parallelPeriodicMesh, 1D_mesh)
{
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  Teuchos::RCP<panzer_stk::STK_Interface> mesh = panzer_stk::buildParallelMesh({5},{2},{2},{2.},{0});
  
  // Make sure there are two blocks (eblock-0, and eblock-1)
  TEST_EQUALITY(mesh->getNumElementBlocks(), 2);
  TEST_EQUALITY(mesh->getNumNodesets(), 2);
  std::vector<std::string> name;
  mesh->getNodesetNames(name);

  std::vector<stk::mesh::Entity> elements;
  std::vector<stk::mesh::Entity> neighbors;
  mesh->getMyElements(elements);
  mesh->getNeighborElements(neighbors);
//  for( const auto& ele : neighbors )
//			  std::cout << myRank << ", " << mesh->EntityGlobalId(ele) << std::endl;
  if(myRank == 0) {
	TEST_EQUALITY(elements.size(), 6);   // element 1,2,3,6,7,8
	TEST_EQUALITY(neighbors.size(), 3);  // element 4,5,9
  } else {
	TEST_EQUALITY(elements.size(), 4);   // element 4,5,9,10
    TEST_EQUALITY(neighbors.size(), 3);  // element 3,6,8
  }
  
  auto periodic = std::make_tuple("NodeSet",name[0],name[1]);
  mesh->find_periodic_nodes();   
  TEST_EQUALITY(mesh->num_pbc_search(), 0);  // cannot find one-to-one peoridic?
//  mesh->PeriodicGhosting();
 // mesh->getNeighborElements(neighbors);
 // for( const auto& ele : neighbors )
//			  std::cout << myRank << ", " << mesh->EntityGlobalId(ele) << std::endl;
}

TEUCHOS_UNIT_TEST(parallelPeriodicMesh, 2D_mesh)
{
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  Teuchos::RCP<panzer_stk::STK_Interface> mesh = panzer_stk::buildParallelMesh({6,2},{1,1},{2,1},{1.,1.},{});
  
  std::vector<stk::mesh::Entity> elements;
  std::vector<stk::mesh::Entity> neighbors;
  mesh->getMyElements(elements);
  mesh->getNeighborElements(neighbors);
  //for( const auto& ele : neighbors )
//			  std::cout << myRank << ", " << mesh->EntityGlobalId(ele) << std::endl;
  if(myRank == 0) {
	TEST_EQUALITY(elements.size(), 6);   // element 1,2,3,7,8,9
	TEST_EQUALITY(neighbors.size(), 2);  // element 4,10
  } else {
	TEST_EQUALITY(elements.size(), 6);   // element 4,5,6,10,11,12
    TEST_EQUALITY(neighbors.size(), 2);  // element 3,9
  }
  
  auto periodic = std::make_tuple("NodeSet","left","right");
  mesh->addPeriodicBC( periodic );
  mesh->find_periodic_nodes();
  mesh->PeriodicGhosting();
  mesh->getNeighborElements(neighbors);
  //for( const auto& ele : neighbors )
  //		  std::cout << myRank << ", " << mesh->EntityGlobalId(ele) << std::endl;
  if(myRank == 0) {
	TEST_EQUALITY(neighbors.size(), 4);  // element 4,10 + 6,12
  } else {
    TEST_EQUALITY(neighbors.size(), 4);  // element 3,9 + 1,7
  }
}

TEUCHOS_UNIT_TEST(parallelPeriodicMesh, 3D_mesh)
{
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  Teuchos::RCP<panzer_stk::STK_Interface> mesh = panzer_stk::buildParallelMesh({6,3,2},{1,1,1},{2,1,1},{1.,1.,1.},{});
  
  std::vector<stk::mesh::Entity> elements;
  std::vector<stk::mesh::Entity> neighbors;
  mesh->getMyElements(elements);
  mesh->getNeighborElements(neighbors);
  if(myRank == 0) {
	TEST_EQUALITY(elements.size(), 18);   // element 1,2,3,7,8,9,13,14,15
	TEST_EQUALITY(neighbors.size(), 6);   // element 4,10,16
  } else {
	TEST_EQUALITY(elements.size(), 18);   // element 4,5,6,10,11,12,16,17,18
    TEST_EQUALITY(neighbors.size(), 6);   // element 3,9,15
  }
  
  auto periodic = std::make_tuple("SideSet","left","right");
  mesh->addPeriodicBC( periodic );
  mesh->find_periodic_nodes();
  mesh->PeriodicGhosting();
  mesh->getNeighborElements(neighbors);
 // for( const auto& ele : neighbors )
 // 		  std::cout << myRank << ", " << mesh->EntityGlobalId(ele) << std::endl;
  if(myRank == 0) {
	TEST_EQUALITY(neighbors.size(), 12);   // element 4,10,16 + 6,12,18
  } else {
    TEST_EQUALITY(neighbors.size(), 12);   // element 3,9,15 + 1,7,13
  }
}
