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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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
#include "Teuchos_ParameterList.hpp"

#include "Panzer_STK_Interface.hpp"
#include "PanzerSTK_UnitTest_BuildMesh.hpp"

#include <string>

/*
   Unit test for function STK_Interface::getElementSideRelation
*/

namespace panzer_stk {

TEUCHOS_UNIT_TEST(localMesh, localSides)
{
	std::vector<std::string>  names;
	std::vector<stk::mesh::Entity>  elements;
	std::vector<panzer::LocalOrdinal> side2ele, ele2side;
	// 1D mesh
	{
		Teuchos::RCP<panzer_stk::STK_Interface> mesh = buildMesh({3},{2},{2.});
		mesh->getElementBlockNames( names );
		mesh->getMyElements(names[1], elements);
		std::vector<panzer::LocalOrdinal> elids;
		for( const auto& ele : elements ) {
			elids.emplace_back(mesh->elementLocalId(ele));
		}
		mesh->getElementSideRelation( elids, side2ele, ele2side);

	   // Three elements, four nodes per block
		TEST_EQUALITY(side2ele.size(), 8);
		TEST_EQUALITY(ele2side.size(), 6);
	}
	
	// 2D Mesh test
    {
		Teuchos::RCP<panzer_stk::STK_Interface> mesh = buildMesh({3,5},{4,3},{2.,6.});
		
		mesh->getElementBlockNames( names );
        TEST_EQUALITY(names.size(), 4*3);
		
		mesh->getMyElements(names[0], elements); 
		TEST_EQUALITY(elements.size(), 3*5);
		
		std::vector<panzer::LocalOrdinal> elids;
		for( const auto& ele : elements ) {
			elids.emplace_back(mesh->elementLocalId(ele));
		}
		mesh->getElementSideRelation( elids, side2ele, ele2side);
		TEST_EQUALITY(side2ele.size(), 76);  // 3*6 + 4*5 sides
		TEST_EQUALITY(ele2side.size(), 60);  // 15 elements with 4 sides per element
	}
	
	// 3D Mesh test
	{
		Teuchos::RCP<panzer_stk::STK_Interface> mesh = buildMesh({3,5,2},{4,3,5},{2.,6.,1.});
		
		mesh->getElementBlockNames( names );
        TEST_EQUALITY(names.size(), 4*3*5);
		
		mesh->getMyElements(names[0], elements); 
		TEST_EQUALITY(elements.size(), 3*5*2);
		
		std::vector<panzer::LocalOrdinal> elids;
		for( const auto& ele : elements ) {
			elids.emplace_back(mesh->elementLocalId(ele));
		}
		mesh->getElementSideRelation( elids, side2ele, ele2side);
		TEST_EQUALITY(side2ele.size(), 242);   // 15*3 + 10*4 + 6*6 faces
		TEST_EQUALITY(ele2side.size(), 180);   // 30 elements with 6 faces per element
	}
}

}
