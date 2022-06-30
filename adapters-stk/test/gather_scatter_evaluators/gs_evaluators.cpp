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

#include "Panzer_WorksetContainer.hpp"
#include "Panzer_STK_WorksetFactory.hpp"
#include "Panzer_Workset_Builder.hpp"
#include "Panzer_FieldManagerBuilder.hpp"
#include "Panzer_TpetraLinearObjFactory.hpp"
#include "Panzer_AssemblyEngine.hpp"
#include "Panzer_AssemblyEngine_InArgs.hpp"
#include "Panzer_AssemblyEngine_TemplateManager.hpp"
#include "Panzer_AssemblyEngine_TemplateBuilder.hpp"
#include "Panzer_DOFManagerFactory.hpp"
#include "Panzer_BasisIRLayout.hpp"
#include "Panzer_GlobalData.hpp"

#include "PanzerAdaptersSTK_config.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STK_GatherFields.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_ParameterLibraryUtilities.hpp"

#include "user_app_EquationSetFactory.hpp"
#include "user_app_ClosureModel_Factory_TemplateBuilder.hpp"

#include "Teuchos_DefaultMpiComm.hpp"
#include "Teuchos_OpaqueWrapper.hpp"

#include <cstdio> // for get char
#include <vector>
#include <string>

namespace panzer {

  Teuchos::RCP<panzer::BasisIRLayout> buildLinearBasis(std::size_t worksetSize);

  Teuchos::RCP<panzer_stk::STK_Interface> buildMesh(int elemX,int elemY);

  TEUCHOS_UNIT_TEST(gs_evaluators, gather_constr)
  {

    const std::size_t workset_size = 20;
    Teuchos::RCP<panzer::BasisIRLayout> linBasis = buildLinearBasis(workset_size);

    Teuchos::RCP<std::vector<std::string> > fieldNames
        = Teuchos::rcp(new std::vector<std::string>);
    fieldNames->push_back("dog");

    Teuchos::ParameterList pl;
    pl.set("Basis",linBasis);
    pl.set("Field Names",fieldNames);

    Teuchos::RCP<panzer_stk::STK_Interface> mesh = buildMesh(2,2);

    Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList("Physics Blocks");
    // Physics block
    Teuchos::ParameterList& physics_block = ipb->sublist("test physics");
	physics_block.set("Material","Cu");
    {
      Teuchos::ParameterList& p = physics_block.sublist("a");
      p.set("Type","Energy");
      p.set("Prefix","");
      p.set("Model ID","solid");
      p.set("Basis Type","HGrad");
      p.set("Basis Order",2);
    }
    {
      Teuchos::ParameterList& p = physics_block.sublist("b");
      p.set("Type","Energy");
      p.set("Prefix","ION_");
      p.set("Model ID","ion solid");
      p.set("Basis Type","HGrad");
      p.set("Basis Order",1);
    }
	
	Teuchos::ParameterList material_models("Material");
	Teuchos::ParameterList& Cu = material_models.sublist("Cu");
	{
		Teuchos::ParameterList& therm = Cu.sublist("Thermal Conductivity");
		therm.set("Value Type","Constant");
		Teuchos::ParameterList& fn = therm.sublist("Constant");
		fn.set<Teuchos::Array<double> >("Value",Teuchos::tuple<double>( 1.0 ));
		
		Teuchos::ParameterList& dens = Cu.sublist("Density");
		dens.set("Value Type","Constant");
		Teuchos::ParameterList& fn1 = dens.sublist("Constant");
		fn1.set<Teuchos::Array<double> >("Value",Teuchos::tuple<double>( 1.0 ));
		
		Teuchos::ParameterList& hc = Cu.sublist("Heat Capacity");
		hc.set("Value Type","Constant");
		Teuchos::ParameterList& fn2 = hc.sublist("Constant");
		fn2.set<Teuchos::Array<double> >("Value",Teuchos::tuple<double>( 1.0 ));
		
		Teuchos::ParameterList& therm1 = Cu.sublist("ION_Thermal Conductivity");
		therm1.set("Value Type","Constant");
		Teuchos::ParameterList& fn3 = therm1.sublist("Constant");
		fn3.set<Teuchos::Array<double> >("Value",Teuchos::tuple<double>( 1.0 ));
		
		Teuchos::ParameterList& dens1 = Cu.sublist("ION_Density");
		dens1.set("Value Type","Constant");
		Teuchos::ParameterList& fn4 = dens1.sublist("Constant");
		fn4.set<Teuchos::Array<double> >("Value",Teuchos::tuple<double>( 1.0 ));
		
		Teuchos::ParameterList& hc1 = Cu.sublist("ION_Heat Capacity");
		hc1.set("Value Type","Constant");
		Teuchos::ParameterList& fn5 = hc1.sublist("Constant");
		fn5.set<Teuchos::Array<double> >("Value",Teuchos::tuple<double>( 1.0 ));
	}
	
	Teuchos::ParameterList pldiric("Dirichlet");
    {	
	   Teuchos::ParameterList& p0 = pldiric.sublist("a");  // noname sublist
	   p0.set("SideSet Name","left");
	   p0.set("Value Type","Constant");
       p0.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "TEMPERATURE" ));
	   Teuchos::ParameterList pl_sub("Constant");
	   pl_sub.set("Value",5.0);
	   p0.set("Constant",pl_sub);
	   
	   Teuchos::ParameterList& p1 = pldiric.sublist("b");  // noname sublist
	   p1.set("SideSet Name","top");
	   p1.set("Value Type","Constant");
       p1.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "TEMPERATURE" ));
	   p1.set("Constant",pl_sub);
	   
	   Teuchos::ParameterList& p2 = pldiric.sublist("c");  // noname sublist
	   p2.set("SideSet Name","right");
	   p2.set("Value Type","Constant");
       p2.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "TEMPERATURE" ));
	   p2.set("Constant",pl_sub);
   }
   //pldiric.print();

    Teuchos::RCP<panzer::FieldManagerBuilder> fmb =
      Teuchos::rcp(new panzer::FieldManagerBuilder);

    // build physics blocks
    //////////////////////////////////////////////////////////////
    Teuchos::RCP<user_app::MyFactory> eqset_factory = Teuchos::rcp(new user_app::MyFactory);
    std::vector<Teuchos::RCP<panzer::PhysicsBlock> > physicsBlocks;

    {
      std::map<std::string,std::string> block_ids_to_physics_ids;
      block_ids_to_physics_ids["eblock-0_0"] = "test physics";
      block_ids_to_physics_ids["eblock-1_0"] = "test physics";

      std::map<std::string,Teuchos::RCP<const shards::CellTopology> > block_ids_to_cell_topo;
      block_ids_to_cell_topo["eblock-0_0"] = mesh->getCellTopology("eblock-0_0");
      block_ids_to_cell_topo["eblock-1_0"] = mesh->getCellTopology("eblock-1_0");

      Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();
	  panzer::createAndRegisterFunctor<double>(material_models,gd->functors);

      int default_integration_order = 1;

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

    Teuchos::RCP<panzer_stk::WorksetFactory> wkstFactory
       = Teuchos::rcp(new panzer_stk::WorksetFactory(mesh)); // build STK workset factory
    Teuchos::RCP<panzer::WorksetContainer> wkstContainer     // attach it to a workset container (uses lazy evaluation)
       = Teuchos::rcp(new panzer::WorksetContainer);
    wkstContainer->setFactory(wkstFactory);
    for(size_t i=0;i<physicsBlocks.size();i++)
      wkstContainer->setNeeds(physicsBlocks[i]->elementBlockID(),physicsBlocks[i]->getWorksetNeeds());
    wkstContainer->setWorksetSize(workset_size);

    // build DOF Manager
    /////////////////////////////////////////////////////////////

    // build the connection manager
    const Teuchos::RCP<panzer::ConnManager>
      conn_manager = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));

    panzer::DOFManagerFactory globalIndexerFactory;
    RCP<panzer::GlobalIndexer> dofManager
         = globalIndexerFactory.buildGlobalIndexer(Teuchos::opaqueWrapper(MPI_COMM_WORLD),physicsBlocks,conn_manager);

    Teuchos::RCP<const Teuchos::MpiComm<int> > tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
    Teuchos::RCP<panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal> > eLinObjFactory
          = Teuchos::rcp(new panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal>(tComm.getConst(),dofManager));
    Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits> > linObjFactory = eLinObjFactory;

    // setup field manager build
    /////////////////////////////////////////////////////////////

    // Add in the application specific closure model factory
    user_app::MyModelFactory_TemplateBuilder cm_builder;
    panzer::ClosureModelFactory_TemplateManager<panzer::Traits> cm_factory;
    cm_factory.buildObjects(cm_builder);

    Teuchos::ParameterList closure_models("Closure Models");
    closure_models.sublist("solid").sublist("SOURCE_TEMPERATURE").set<double>("Value",1.0);
    closure_models.sublist("ion solid").sublist("SOURCE_ION_TEMPERATURE").set<double>("Value",1.0);

    Teuchos::ParameterList user_data("User Data");

    fmb->setWorksetContainer(wkstContainer);
    fmb->setupVolumeFieldManagers(physicsBlocks,cm_factory,closure_models,*eLinObjFactory,user_data);
    //fmb->setupBCFieldManagers(bcs,physicsBlocks,*eqset_factory,cm_factory,bc_factory,closure_models,*eLinObjFactory,user_data);
	fmb->setupDiricheltFieldManagers(pldiric,mesh,dofManager);

    fmb->writeVolumeGraphvizDependencyFiles("field_manager",physicsBlocks);

    panzer::AssemblyEngine_TemplateManager<panzer::Traits> ae_tm;
    panzer::AssemblyEngine_TemplateBuilder builder(fmb,eLinObjFactory);
    ae_tm.buildObjects(builder);

    RCP<panzer::LinearObjContainer> ghostCont
       = (eLinObjFactory->buildGhostedLinearObjContainer());
    RCP<panzer::LinearObjContainer> container
       = (eLinObjFactory->buildLinearObjContainer());
    eLinObjFactory->initializeGhostedContainer(panzer::LinearObjContainer::X |
											 panzer::LinearObjContainer::DxDt |
                                             panzer::LinearObjContainer::F |
                                             panzer::LinearObjContainer::Mat,*ghostCont);
    eLinObjFactory->initializeContainer(panzer::LinearObjContainer::X |
											 panzer::LinearObjContainer::DxDt |
                                             panzer::LinearObjContainer::F |
                                             panzer::LinearObjContainer::Mat,*container);
    panzer::AssemblyEngineInArgs input(ghostCont,container);

    ae_tm.getAsObject<panzer::Traits::Residual>()->evaluate(input);
    ae_tm.getAsObject<panzer::Traits::Jacobian>()->evaluate(input);
  }

  Teuchos::RCP<panzer::BasisIRLayout> buildLinearBasis(std::size_t worksetSize)
  {
     Teuchos::RCP<shards::CellTopology> topo =
        Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));

     panzer::CellData cellData(worksetSize,topo);
     panzer::IntegrationRule intRule(1,cellData);

     return Teuchos::rcp(new panzer::BasisIRLayout("Q1",1,intRule));
  }

  Teuchos::RCP<panzer_stk::STK_Interface> buildMesh(int elemX,int elemY)
  {
    typedef panzer_stk::STK_Interface::SolutionFieldType VariableField;
    typedef panzer_stk::STK_Interface::VectorFieldType CoordinateField;

    RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
    pl->set("X Blocks",2);
    pl->set("Y Blocks",1);
    pl->set("X Elements",elemX);
    pl->set("Y Elements",elemY);

    panzer_stk::SquareQuadMeshFactory factory;
    factory.setParameterList(pl);
    RCP<panzer_stk::STK_Interface> mesh = factory.buildUncommitedMesh(MPI_COMM_WORLD);

    // add in some fields
    mesh->addSolutionField("dog","eblock-0_0");
    mesh->addSolutionField("dog","eblock-1_0");

    factory.completeMeshConstruction(*mesh,MPI_COMM_WORLD);

    VariableField * field = mesh->getMetaData()->get_field<VariableField>(stk::topology::NODE_RANK, "dog");
    CoordinateField * cField = mesh->getMetaData()->get_field<CoordinateField>(stk::topology::NODE_RANK, "coordinates");
    TEUCHOS_ASSERT(field!=0);
    TEUCHOS_ASSERT(cField!=0);

    // fill the fields with data
    const std::vector<stk::mesh::Bucket*> nodeData
        = mesh->getBulkData()->buckets(mesh->getNodeRank());
    for(std::size_t b=0;b<nodeData.size();++b) {
       stk::mesh::Bucket * bucket = nodeData[b];

       // build all buckets
       for(stk::mesh::Bucket::iterator itr=bucket->begin();
           itr!=bucket->end();++itr) {

          double* coordinates = stk::mesh::field_data(*cField,*itr);
          double* dog_array   = stk::mesh::field_data(*field,*itr);

          double x = coordinates[0];
          double y = coordinates[1];

          *dog_array = 4.0*x*x+y;
       }
    }

    if(mesh->isWritable())
       mesh->writeToExodus("output.exo");

    return mesh;
  }

}
