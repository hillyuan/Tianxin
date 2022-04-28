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
#include "Panzer_STK_LocalMeshUtilities.hpp"
#include "PanzerSTK_UnitTest_BuildMesh.hpp"

#include <string>

/*
   Unit test for function STK_Interface::getMyElements, getNeighborElements, getMySides, getElementSideRelation
*/

TEUCHOS_UNIT_TEST(parallelLocalMesh, 1D_mesh)
{

  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  std::vector<int> p;
  Teuchos::RCP<panzer_stk::STK_Interface> mesh = panzer_stk::buildParallelMesh({3},{2},{2},{2.}, p);

  // Make sure there are two blocks (eblock-0, and eblock-1)
  TEST_EQUALITY(mesh->getNumElementBlocks(), 2);
  TEST_EQUALITY(mesh->getNumSidesets(), 2);

  std::vector<stk::mesh::Entity> elements;
  std::vector<stk::mesh::Entity> neighbors;
  std::vector<stk::mesh::Entity> sides;
  std::vector<panzer::LocalOrdinal> elids;
  std::vector<panzer::LocalOrdinal> side2ele, ele2side;
  if(myRank == 0){
	  
	  {
		  mesh->getMyElements(elements);
		  TEST_EQUALITY(elements.size(), 4);   // element 1,2,4,5
		  mesh->getNeighborElements(neighbors);
		  TEST_EQUALITY(neighbors.size(), 2);  // element 3,6
	      mesh->getMySides(sides);             // node 1,2,3,4,5,6
		  TEST_EQUALITY(sides.size(), 6);
		  for( const auto& ele : elements ) {
			elids.emplace_back(mesh->elementLocalId(ele));
		  }
		  mesh->getElementSideRelation( elids, side2ele, ele2side);
		  TEST_EQUALITY(side2ele.size(), 12);
		  TEST_EQUALITY(ele2side.size(), 8);
	  }

    {
		mesh->getMyElements("eblock-0",elements);
		TEST_EQUALITY(elements.size(), 2);   // element 1,2
		TEST_EQUALITY(mesh->elementLocalId(elements[0]), 0);   
        TEST_EQUALITY(mesh->elementLocalId(elements[1]), 1);   
		mesh->getNeighborElements("eblock-0",neighbors);
		TEST_EQUALITY(neighbors.size(), 1);  // element 3
		TEST_EQUALITY(mesh->EntityGlobalId(neighbors[0]), 3); 
    }

    {
		mesh->getMyElements("eblock-1",elements);
		TEST_EQUALITY(elements.size(), 2);   // element 4,5
		mesh->getNeighborElements("eblock-1",neighbors);
		TEST_EQUALITY(neighbors.size(), 1);  // element 6
    }

    {
		mesh->getMySides("left", sides);
		TEST_EQUALITY(sides.size(), 1); 
		mesh->getMySides("left","eblock-0", sides);
		TEST_EQUALITY(sides.size(), 1);      // node 1
    }

  } else {
	  {
		  mesh->getMyElements(elements);
		  TEST_EQUALITY(elements.size(), 2);   // element 3,6
		  mesh->getNeighborElements(neighbors);
		  TEST_EQUALITY(neighbors.size(), 3);  // element 2,4,5
		  mesh->getMySides(sides);             // node 7
		  TEST_EQUALITY(sides.size(), 1);
		  for( const auto& ele : elements ) {
			elids.emplace_back(mesh->elementLocalId(ele));
		  }; 
		  mesh->getElementSideRelation( elids, side2ele, ele2side);
		  TEST_EQUALITY(side2ele.size(), 8);   // there four nodes (3,4,6,7) relates to element 3,6
		  TEST_EQUALITY(ele2side.size(), 4);
	  }
    {
		mesh->getMyElements("eblock-0",elements);
		TEST_EQUALITY(elements.size(), 1);    // element 3
		mesh->getNeighborElements("eblock-0",neighbors);
		TEST_EQUALITY(neighbors.size(), 1);   // element 2
    }

    {
        mesh->getMyElements("eblock-1",elements);
		TEST_EQUALITY(elements.size(), 1);    // element 6
		mesh->getNeighborElements("eblock-1",neighbors);
		TEST_EQUALITY(neighbors.size(), 2);   // element 4, 5
    }

    {
        mesh->getMySides("right","eblock-1", sides);
		TEST_EQUALITY(sides.size(), 1);       // node 7
	//	for( const auto& side : sides )
	//		  std::cout << mesh->EntityGlobalId(side) << std::endl;
    }
  }

}

TEUCHOS_UNIT_TEST(parallelLocalMeshUtilities, 2D_mesh)
{

  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  std::vector<int> p;
  Teuchos::RCP<panzer_stk::STK_Interface> mesh = panzer_stk::buildParallelMesh({2,2},{2,1},{2,1},{2.,2.},p);

  // Make sure there are two blocks (eblock-0_0, and eblock-1_0)
  TEST_EQUALITY(mesh->getNumElementBlocks(), 2);
  // There are four sideset "left, right, top, bottom, center"
  TEST_EQUALITY(mesh->getNumSidesets(), 5);

  std::vector<stk::mesh::Entity> elements;
  std::vector<stk::mesh::Entity> neighbors;
  std::vector<stk::mesh::Entity> sides;
  std::vector<panzer::LocalOrdinal> elids;
  std::vector<panzer::LocalOrdinal> side2ele, ele2side;
  if(myRank == 0){
	  
	  {
		  mesh->getMyElements(elements);
		  TEST_EQUALITY(elements.size(), 4);   // element 1,3,5,7
		  mesh->getNeighborElements(neighbors);
		  TEST_EQUALITY(neighbors.size(), 4);  // element 2,4,6,8
	      mesh->getMySides(sides);             // side
		//    for( const auto& ele : sides )
		//	  std::cout << mesh->EntityGlobalId(ele) << std::endl;
		//  TEST_EQUALITY(sides.size(), 6);
		  for( const auto& ele : elements ) {
			elids.emplace_back(mesh->elementLocalId(ele));
		  }
		  mesh->getElementSideRelation( elids, side2ele, ele2side);
		//  TEST_EQUALITY(side2ele.size(), 12);
		  TEST_EQUALITY(ele2side.size(), 16);
	  }

    {
		mesh->getMyElements("eblock-0_0",elements);
		TEST_EQUALITY(elements.size(), 2);   // element 1,5
		TEST_EQUALITY(mesh->EntityGlobalId(elements[0]), 1);   
        TEST_EQUALITY(mesh->EntityGlobalId(elements[1]), 5);   
		mesh->getNeighborElements("eblock-0_0",neighbors);
		TEST_EQUALITY(neighbors.size(), 2);  // element 2,6
		TEST_EQUALITY(mesh->EntityGlobalId(neighbors[0]), 2);
    }

    {
		mesh->getMyElements("eblock-1_0",elements);
		TEST_EQUALITY(elements.size(), 2);   // element 3,7
		TEST_EQUALITY(mesh->EntityGlobalId(elements[0]), 3);   
        TEST_EQUALITY(mesh->EntityGlobalId(elements[1]), 7);   
		mesh->getNeighborElements("eblock-1_0",neighbors);
		TEST_EQUALITY(neighbors.size(), 2);  // element 4,8
		TEST_EQUALITY(mesh->EntityGlobalId(neighbors[0]), 4);
    }

    {
		mesh->getMySides("left","eblock-0_0", sides);
		TEST_EQUALITY(sides.size(), 2);
		mesh->getMySides("left", sides);
		TEST_EQUALITY(sides.size(), 2);
    }

    {
		mesh->getMySides("top","eblock-0_0", sides);
		TEST_EQUALITY(sides.size(), 1);   // one side in eblock_0_0 in current cpu
		mesh->getMySides("top", sides);
		TEST_EQUALITY(sides.size(), 2);   // two sides in current cpu
    }

    {
        mesh->getMySides("bottom","eblock-1_0", sides);
		TEST_EQUALITY(sides.size(), 1);   // one side in eblock_1_0 in current cpu
		mesh->getMySides("bottom", sides);
		TEST_EQUALITY(sides.size(), 2);   // two sides in current cpu
    }

  } else {
	  
	  {
		mesh->getMyElements(elements);
		TEST_EQUALITY(elements.size(), 4);   // element 2,4,6,8
		mesh->getNeighborElements(neighbors);
		TEST_EQUALITY(neighbors.size(), 4);  // element 1,3,5,7
		//  for( const auto& ele : neighbors )
		//	  std::cout << mesh->EntityGlobalId(ele) << std::endl;
	  }

	{
		mesh->getMyElements("eblock-0_0",elements);
		TEST_EQUALITY(elements.size(), 2);   // element 2,6
		TEST_EQUALITY(mesh->EntityGlobalId(elements[0]), 2);   
        TEST_EQUALITY(mesh->EntityGlobalId(elements[1]), 6);   
		mesh->getNeighborElements("eblock-0_0",neighbors);
		TEST_EQUALITY(neighbors.size(), 2);  // element 1,5
		TEST_EQUALITY(mesh->EntityGlobalId(neighbors[0]), 1);
    }

    {
        mesh->getMyElements("eblock-1_0",elements);
		TEST_EQUALITY(elements.size(), 2);   // element 4,8
		TEST_EQUALITY(mesh->EntityGlobalId(elements[0]), 4);   
        TEST_EQUALITY(mesh->EntityGlobalId(elements[1]), 8);   
		mesh->getNeighborElements("eblock-1_0",neighbors);
		TEST_EQUALITY(neighbors.size(), 2);  // element 3,7
		TEST_EQUALITY(mesh->EntityGlobalId(neighbors[0]), 3);
    }

    {
		mesh->getMySides("right","eblock-1_0", sides);
		TEST_EQUALITY(sides.size(), 2);
		mesh->getMySides("right", sides);
		TEST_EQUALITY(sides.size(), 2);
    }

    {
		mesh->getMySides("top","eblock-0_0", sides);
		TEST_EQUALITY(sides.size(), 1);   // one side in eblock_0_0 in current cpu
		mesh->getMySides("top", sides);
		TEST_EQUALITY(sides.size(), 2);   // two sides in current cpu
    }

    {
        mesh->getMySides("bottom","eblock-1_0", sides);
		TEST_EQUALITY(sides.size(), 1);   // one side in eblock_1_0 in current cpu
		mesh->getMySides("bottom", sides);
		TEST_EQUALITY(sides.size(), 2);   // two sides in current cpu
    }
  }

}


TEUCHOS_UNIT_TEST(parallelLocalMeshUtilities, 3D_mesh)
{

  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  std::vector<int> p;
  Teuchos::RCP<panzer_stk::STK_Interface> mesh= panzer_stk::buildParallelMesh({2,2,2},{2,1,1},{2,1,1},{2.,2.,2.},p);

  // Make sure there are two blocks (eblock-0_0_0, and eblock-1_0_0)
  TEST_EQUALITY(mesh->getNumElementBlocks(), 2);
  // There are six sideset "left, right, top, bottom, front, back"
  TEST_EQUALITY(mesh->getNumSidesets(), 6);
  
  std::vector<stk::mesh::Entity> elements;
  std::vector<stk::mesh::Entity> neighbors;
  std::vector<stk::mesh::Entity> sides;
  std::vector<panzer::LocalOrdinal> elids;
  std::vector<panzer::LocalOrdinal> side2ele, ele2side;
  if(myRank == 0){
	  
	  {
		  mesh->getMyElements(elements);
		  TEST_EQUALITY(elements.size(), 8);   // element 1,3,5,7,9,11,13,15
		  mesh->getNeighborElements(neighbors);
		  TEST_EQUALITY(neighbors.size(), 8);  // element 2,4,6,8,10,12,14,16
	      mesh->getMySides(sides);             // node 
		//    for( const auto& ele : sides )
		//	  std::cout << mesh->EntityGlobalId(ele) << std::endl;
		//  TEST_EQUALITY(sides.size(), 6);
		  for( const auto& ele : elements ) {
			elids.emplace_back(mesh->elementLocalId(ele));
		  }
		  mesh->getElementSideRelation( elids, side2ele, ele2side);
		//  TEST_EQUALITY(side2ele.size(), 12);
		  TEST_EQUALITY(ele2side.size(), 48);
	  }

    {
		mesh->getMyElements("eblock-0_0_0",elements);
		TEST_EQUALITY(elements.size(), 4);   // element 1,5,9,13
		TEST_EQUALITY(mesh->EntityGlobalId(elements[0]), 1);   
        TEST_EQUALITY(mesh->EntityGlobalId(elements[1]), 5);   
		mesh->getNeighborElements("eblock-0_0_0",neighbors);
		TEST_EQUALITY(neighbors.size(), 4);  // element 2,6,10,14
		TEST_EQUALITY(mesh->EntityGlobalId(neighbors[0]), 2);
		TEST_EQUALITY(mesh->EntityGlobalId(neighbors[3]), 14);
    }

    {
		mesh->getMyElements("eblock-1_0_0",elements);
		TEST_EQUALITY(elements.size(), 4);   // element 3,7,11,15
		TEST_EQUALITY(mesh->EntityGlobalId(elements[0]), 3);   
        TEST_EQUALITY(mesh->EntityGlobalId(elements[1]), 7);   
		mesh->getNeighborElements("eblock-1_0_0",neighbors);
		TEST_EQUALITY(neighbors.size(), 4);  // element 4,8,12,16
		TEST_EQUALITY(mesh->EntityGlobalId(neighbors[0]), 4);
		TEST_EQUALITY(mesh->EntityGlobalId(neighbors[3]), 16);
    }

    {
		mesh->getMySides("left","eblock-0_0_0", sides);
		TEST_EQUALITY(sides.size(), 4);
		mesh->getMySides("left", sides);
		TEST_EQUALITY(sides.size(), 4);
    }

    {
		mesh->getMySides("top","eblock-0_0_0", sides);
		TEST_EQUALITY(sides.size(), 2);
		mesh->getMySides("top", sides);
		TEST_EQUALITY(sides.size(), 4);
    }

    {
		mesh->getMySides("front","eblock-0_0_0", sides);
		TEST_EQUALITY(sides.size(), 2);
		mesh->getMySides("front", sides);
		TEST_EQUALITY(sides.size(), 4);
    }

   {
		mesh->getMySides("top","eblock-1_0_0", sides);
		TEST_EQUALITY(sides.size(), 2);
		mesh->getMySides("top", sides);
		TEST_EQUALITY(sides.size(), 4);
    }

    {
		mesh->getMySides("back","eblock-1_0_0", sides);
		TEST_EQUALITY(sides.size(), 2);
		mesh->getMySides("back", sides);
		TEST_EQUALITY(sides.size(), 4);
    }

  } else {
	  
	  {
		  mesh->getMyElements(elements);
		  TEST_EQUALITY(elements.size(), 8);   // element 2,4,6,8,10,12,14,16
		  mesh->getNeighborElements(neighbors);
		  TEST_EQUALITY(neighbors.size(), 8);  // element 1,3,5,7,9,11,13,15
	      mesh->getMySides(sides);             // face 
		//    for( const auto& ele : sides )
		//	  std::cout << mesh->EntityGlobalId(ele) << std::endl;
		//  TEST_EQUALITY(sides.size(), 6);
		  for( const auto& ele : elements ) {
			elids.emplace_back(mesh->elementLocalId(ele));
		  }
		  mesh->getElementSideRelation( elids, side2ele, ele2side);
		//  TEST_EQUALITY(side2ele.size(), 12);
		  TEST_EQUALITY(ele2side.size(), 48);
	  }

    {
		mesh->getMyElements("eblock-0_0_0",elements);
		TEST_EQUALITY(elements.size(), 4);   // element 2,6,10,14
		TEST_EQUALITY(mesh->EntityGlobalId(elements[0]), 2);   
        TEST_EQUALITY(mesh->EntityGlobalId(elements[1]), 6);   
		mesh->getNeighborElements("eblock-0_0_0",neighbors);
		TEST_EQUALITY(neighbors.size(), 4);  // element 1,5,9,13
		TEST_EQUALITY(mesh->EntityGlobalId(neighbors[0]), 1);
		TEST_EQUALITY(mesh->EntityGlobalId(neighbors[3]), 13);
    }

    {
		mesh->getMyElements("eblock-1_0_0",elements);
		TEST_EQUALITY(elements.size(), 4);   // element 4,8,12,16
		TEST_EQUALITY(mesh->EntityGlobalId(elements[0]), 4);   
        TEST_EQUALITY(mesh->EntityGlobalId(elements[1]), 8);   
		mesh->getNeighborElements("eblock-1_0_0",neighbors);
		TEST_EQUALITY(neighbors.size(), 4);  // element 3,7,11,15
		TEST_EQUALITY(mesh->EntityGlobalId(neighbors[0]), 3);
		TEST_EQUALITY(mesh->EntityGlobalId(neighbors[3]), 15);
    }


    {
		mesh->getMySides("right","eblock-1_0_0", sides);
		TEST_EQUALITY(sides.size(), 4);
		mesh->getMySides("right", sides);
		TEST_EQUALITY(sides.size(), 4);
    }

    {
		mesh->getMySides("bottom","eblock-0_0_0", sides);
		TEST_EQUALITY(sides.size(), 2);
		mesh->getMySides("bottom", sides);
		TEST_EQUALITY(sides.size(), 4);
    }


    {
		mesh->getMySides("back","eblock-0_0_0", sides);
		TEST_EQUALITY(sides.size(), 2);
		mesh->getMySides("back", sides);
		TEST_EQUALITY(sides.size(), 4);
    }
	
	{
		mesh->getMySides("top","eblock-1_0_0", sides);
		TEST_EQUALITY(sides.size(), 2);
		mesh->getMySides("top", sides);
		TEST_EQUALITY(sides.size(), 4);
    }


    {
		mesh->getMySides("front","eblock-1_0_0", sides);
		TEST_EQUALITY(sides.size(), 2);
		mesh->getMySides("front", sides);
		TEST_EQUALITY(sides.size(), 4);
    }
  }

}

