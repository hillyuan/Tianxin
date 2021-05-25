// @HEADER
// ***********************************************************************
//
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2011) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// Questions? Contact Roger P. Pawlowski (rppawlo@sandia.gov) and
// Eric C. Cyr (eccyr@sandia.gov)
// ***********************************************************************
// @HEADER

#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

using Teuchos::RCP;
using Teuchos::rcp;

#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include "Panzer_STK_Version.hpp"
#include "PanzerAdaptersSTK_config.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SquareTriMeshFactory.hpp"
#include "Panzer_STK_QuadTriMeshFactory.hpp"
#include "Panzer_STK_CubeTetMeshFactory.hpp"
#include "Panzer_Workset_Builder.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_PhysicsBlock.hpp"
#include "Panzer_GlobalData.hpp"
#include "Panzer_BC.hpp"
#include "Kokkos_ViewFactory.hpp"
#include "user_app_EquationSetFactory.hpp"

namespace panzer {

  void testInitialzation(const Teuchos::RCP<Teuchos::ParameterList>& ipb,
			 std::vector<panzer::BC>& bcs);


  TEUCHOS_UNIT_TEST(mixelement_builder, volume)
  {

    RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
    pl->set("X Blocks",2);
    pl->set("Y Blocks",1);
    pl->set("X Elements",2);  // in each block
    pl->set("Y Elements",1);  // in each block

    panzer_stk::QuadTriMeshFactory factory;
    factory.setParameterList(pl);
    RCP<panzer_stk::STK_Interface> mesh = factory.buildMesh(MPI_COMM_WORLD);
    if(mesh->isWritable())
      mesh->writeToExodus("mixed_mesh.exo");

    std::vector<std::string> element_blocks;
    mesh->getElementBlockNames(element_blocks);
    const std::size_t workset_size = 20;

    Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList("Physics Blocks");
    std::vector<panzer::BC> bcs;
    testInitialzation(ipb, bcs);

    // build physics blocks
    //////////////////////////////////////////////////////////////
    std::vector<Teuchos::RCP<panzer::PhysicsBlock> > physicsBlocks;
    {
      const int default_integration_order = 1;

      Teuchos::RCP<user_app::MyFactory> eqset_factory = Teuchos::rcp(new user_app::MyFactory);
    
      std::map<std::string,std::string> block_ids_to_physics_ids;
      block_ids_to_physics_ids["eblock-0_0"] = "test physics";
      block_ids_to_physics_ids["eblock-1_0"] = "test physics";

      std::map<std::string,Teuchos::RCP<const shards::CellTopology> > block_ids_to_cell_topo;
      block_ids_to_cell_topo["eblock-0_0"] = mesh->getCellTopology("eblock-0_0");
      block_ids_to_cell_topo["eblock-1_0"] = mesh->getCellTopology("eblock-1_0");
      
      Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();

      panzer::buildPhysicsBlocks(block_ids_to_physics_ids,
				 block_ids_to_cell_topo,
				 ipb,
				 default_integration_order,
				 workset_size,
				 eqset_factory,
				 gd,
				 false,
				 physicsBlocks);
    }

    std::vector< Teuchos::RCP<std::vector<panzer::Workset> > > worksets;

    for (std::vector<std::string>::size_type i=0; i < element_blocks.size(); ++i) {

      std::vector<std::size_t> local_cell_ids;
      Kokkos::DynRankView<double,PHX::Device> cell_vertex_coordinates;

      panzer_stk::workset_utils::getIdsAndVertices(*mesh, element_blocks[i], local_cell_ids, 
				cell_vertex_coordinates);

      Teuchos::RCP<shards::CellTopology> topo
         = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));

      Teuchos::RCP<const panzer::PhysicsBlock> pb = panzer::findPhysicsBlock(element_blocks[i],physicsBlocks);
      worksets.push_back(panzer::buildWorksets(pb->getWorksetNeeds(),pb->elementBlockID(),
					       local_cell_ids,
					       cell_vertex_coordinates));
    
      TEST_EQUALITY((*worksets[i])[0].cell_vertex_coordinates(0,0,0), cell_vertex_coordinates(0,0,0));
      TEST_EQUALITY((*worksets[i])[0].cell_vertex_coordinates(1,2,1), cell_vertex_coordinates(1,2,1));

      TEST_ASSERT((*worksets[i])[0].cell_local_ids==local_cell_ids);

      TEST_EQUALITY((*worksets[i])[0](0).cell_vertex_coordinates(0,0,0), cell_vertex_coordinates(0,0,0));
      TEST_EQUALITY((*worksets[i])[0](0).cell_vertex_coordinates(1,2,1), cell_vertex_coordinates(1,2,1));
    }
    

    TEST_EQUALITY(worksets.size(), 2);
    TEST_EQUALITY(worksets[0]->size(), 1);
    TEST_EQUALITY(worksets[1]->size(), 1);
	  
	TEST_EQUALITY((*worksets[0])[0](0).cell_vertex_coordinates.extent(1), 4);
	TEST_EQUALITY((*worksets[1])[0](0).cell_vertex_coordinates.extent(1), 3);

    TEST_EQUALITY((*worksets[0])[0].num_cells, 2);
    TEST_EQUALITY((*worksets[1])[0].num_cells, 4);
    
    TEST_EQUALITY((*worksets[0])[0].block_id, element_blocks[0]);
    TEST_EQUALITY((*worksets[1])[0].block_id, element_blocks[1]);
    
  }
	
  TEUCHOS_UNIT_TEST(workset_builder, edge)
  {

    RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
    pl->set("X Blocks",2);
    pl->set("Y Blocks",1);
    pl->set("X Elements",2);  // in each block
    pl->set("Y Elements",1);  // in each block

    panzer_stk::QuadTriMeshFactory factory;
    factory.setParameterList(pl);
    RCP<panzer_stk::STK_Interface> mesh = factory.buildMesh(MPI_COMM_WORLD);

    std::vector<std::string> element_blocks;
    mesh->getElementBlockNames(element_blocks);
    const std::size_t workset_size = 20;

    Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList("Physics Blocks");
    std::vector<panzer::BC> bcs;
    testInitialzation(ipb, bcs);

    // build physics blocks
    //////////////////////////////////////////////////////////////
    std::vector<Teuchos::RCP<panzer::PhysicsBlock> > physicsBlocks;
    {
      const int default_integration_order = 1;

      Teuchos::RCP<user_app::MyFactory> eqset_factory = Teuchos::rcp(new user_app::MyFactory);
    
      std::map<std::string,std::string> block_ids_to_physics_ids;
      block_ids_to_physics_ids["eblock-0_0"] = "test physics";
      block_ids_to_physics_ids["eblock-1_0"] = "test physics";

      std::map<std::string,Teuchos::RCP<const shards::CellTopology> > block_ids_to_cell_topo;
      block_ids_to_cell_topo["eblock-0_0"] = mesh->getCellTopology("eblock-0_0");
      block_ids_to_cell_topo["eblock-1_0"] = mesh->getCellTopology("eblock-1_0");
      
      Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();

      panzer::buildPhysicsBlocks(block_ids_to_physics_ids,
				 block_ids_to_cell_topo,
				 ipb,
				 default_integration_order,
				 workset_size,
				 eqset_factory,
				 gd,
				 false,
				 physicsBlocks);
    }

    Teuchos::RCP<std::vector<panzer::Workset> > worksets;

    {
      std::vector<std::size_t> local_cell_ids_a, local_cell_ids_b;
      std::vector<std::size_t> local_side_ids_a, local_side_ids_b;

      local_cell_ids_a.push_back(0);
      local_cell_ids_b.push_back(4);

      local_side_ids_a.push_back(3);
      local_side_ids_b.push_back(1);

      Kokkos::DynRankView<double,PHX::Device> cell_vertex_coordinates_a, cell_vertex_coordinates_b;
      mesh->getElementVertices(local_cell_ids_a,cell_vertex_coordinates_a);
      mesh->getElementVertices(local_cell_ids_b,cell_vertex_coordinates_b);

      Teuchos::RCP<const panzer::PhysicsBlock> pb_a = panzer::findPhysicsBlock(element_blocks[0],physicsBlocks);
      Teuchos::RCP<const panzer::PhysicsBlock> pb_b = panzer::findPhysicsBlock(element_blocks[1],physicsBlocks);
      worksets = panzer::buildEdgeWorksets( pb_a->getWorksetNeeds(),pb_a->elementBlockID(),
 	                         local_cell_ids_a, local_side_ids_a, cell_vertex_coordinates_a,
                                 pb_b->getWorksetNeeds(),pb_b->elementBlockID(),
			                 local_cell_ids_b, local_side_ids_b, cell_vertex_coordinates_b);

     
      TEST_EQUALITY((*worksets).size(),1);
      TEST_EQUALITY((*worksets)[0].num_cells,1);
      TEST_EQUALITY((*worksets)[0].subcell_dim,1);

      // this is identical to (*worksets)[0](0)
      TEST_EQUALITY((*worksets)[0].cell_vertex_coordinates(0,0,0), cell_vertex_coordinates_a(0,0,0));
      TEST_EQUALITY((*worksets)[0].cell_vertex_coordinates(0,3,1), cell_vertex_coordinates_a(0,3,1));
      TEST_EQUALITY((*worksets)[0].subcell_index, 3);
      TEST_EQUALITY((*worksets)[0].block_id, "eblock-0_0");
      TEST_EQUALITY((*worksets)[0].cell_local_ids.size(),1);
      TEST_EQUALITY((*worksets)[0].cell_local_ids[0],0);
      TEST_EQUALITY((*worksets)[0].ir_degrees->size(),1);
      TEST_EQUALITY((*worksets)[0].int_rules.size(),1);
      TEST_EQUALITY((*worksets)[0].basis_names->size(),2);
      TEST_EQUALITY((*worksets)[0].bases.size(),2);
      
      TEST_EQUALITY((*worksets)[0](0).cell_vertex_coordinates(0,0,0), cell_vertex_coordinates_a(0,0,0));
      TEST_EQUALITY((*worksets)[0](0).cell_vertex_coordinates(0,3,1), cell_vertex_coordinates_a(0,3,1));
      TEST_EQUALITY((*worksets)[0](0).subcell_index, 3);
      TEST_EQUALITY((*worksets)[0](0).block_id, "eblock-0_0");
      TEST_EQUALITY((*worksets)[0](0).cell_local_ids.size(),1);
      TEST_EQUALITY((*worksets)[0](0).cell_local_ids[0],0);
      TEST_EQUALITY((*worksets)[0](0).ir_degrees->size(),1);
      TEST_EQUALITY((*worksets)[0](0).int_rules.size(),1);
      TEST_EQUALITY((*worksets)[0](0).basis_names->size(),2);
      TEST_EQUALITY((*worksets)[0](0).bases.size(),2);

      TEST_EQUALITY((*worksets)[0](1).cell_vertex_coordinates(0,0,0), cell_vertex_coordinates_b(0,0,0));
      TEST_EQUALITY((*worksets)[0](1).cell_vertex_coordinates(0,2,1), cell_vertex_coordinates_b(0,2,1));
      TEST_EQUALITY((*worksets)[0](1).subcell_index, 1);
      TEST_EQUALITY((*worksets)[0](1).block_id, "eblock-1_0");
      TEST_EQUALITY((*worksets)[0](1).cell_local_ids[0],4);
      TEST_EQUALITY((*worksets)[0](1).ir_degrees->size(),1);
      TEST_EQUALITY((*worksets)[0](1).int_rules.size(),1);
      TEST_EQUALITY((*worksets)[0](1).basis_names->size(),2);
      TEST_EQUALITY((*worksets)[0](1).bases.size(),2);
    }
    
  }

  void testInitialzation(const Teuchos::RCP<Teuchos::ParameterList>& ipb,
			 std::vector<panzer::BC>& bcs)
  {
    // Physics block
    Teuchos::ParameterList& physics_block = ipb->sublist("test physics");
    {
      Teuchos::ParameterList& p = physics_block.sublist("a");
      p.set("Type","Energy");
      p.set("Prefix","");
      p.set("Model ID","solid");
      p.set("Basis Type","HGrad");
      p.set("Basis Order",2);
      p.set("Integration Order",1);
    }
    {
      Teuchos::ParameterList& p = physics_block.sublist("b");
      p.set("Type","Energy");
      p.set("Prefix","ION_");
      p.set("Model ID","ion solid");
      p.set("Basis Type","HGrad");
      p.set("Basis Order",1);
      p.set("Integration Order",1);
    }

    {
      std::size_t bc_id = 0;
      panzer::BCType neumann = BCT_Dirichlet;
      std::string sideset_id = "left";
      std::string element_block_id = "eblock-0_0";
      std::string dof_name = "UX";
      std::string strategy = "Constant";
      double value = 5.0;
      Teuchos::ParameterList p;
      p.set("Value",value);
      panzer::BC bc(bc_id, neumann, sideset_id, element_block_id, dof_name, 
		    strategy, p);
      bcs.push_back(bc);
    }    
    {
      std::size_t bc_id = 0;
      panzer::BCType neumann = BCT_Dirichlet;
      std::string sideset_id = "right";
      std::string element_block_id = "eblock-1_0";
      std::string dof_name = "UX";
      std::string strategy = "Constant";
      double value = 5.0;
      Teuchos::ParameterList p;
      p.set("Value",value);
      panzer::BC bc(bc_id, neumann, sideset_id, element_block_id, dof_name, 
		    strategy, p);
      bcs.push_back(bc);
    }   
    {
      std::size_t bc_id = 0;
      panzer::BCType neumann = BCT_Dirichlet;
      std::string sideset_id = "top";
      std::string element_block_id = "eblock-1_0";
      std::string dof_name = "UX";
      std::string strategy = "Constant";
      double value = 5.0;
      Teuchos::ParameterList p;
      p.set("Value",value);
      panzer::BC bc(bc_id, neumann, sideset_id, element_block_id, dof_name, 
		    strategy, p);
      bcs.push_back(bc);
    }
  }

}
