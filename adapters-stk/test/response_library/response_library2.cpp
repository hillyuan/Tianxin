// @HEADER
// ***********************************************************************
//
//           TianXin: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2022) YUAN Xi
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
// ***********************************************************************
// @HEADER

#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include "PanzerAdaptersSTK_config.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STK_WorksetFactory.hpp"
#include "Panzer_Workset_Builder.hpp"
#include "Panzer_FieldManagerBuilder.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_DOFManagerFactory.hpp"
#include "Panzer_BlockedEpetraLinearObjFactory.hpp"
#include "Panzer_GlobalData.hpp"
#include "Panzer_ResponseEvaluatorFactory_Functional.hpp"
#include "user_app_EquationSetFactory.hpp"
#include "user_app_STKClosureModel_Factory_TemplateBuilder.hpp"

#include "Panzer_WorksetContainer.hpp"
#include "TianXin_Response_Integral.hpp"

#include "TestEvaluators.hpp"

#include <vector>
#include <map>
#include <string>

#include "Sacado_mpl_vector.hpp"

using Teuchos::RCP;

namespace panzer {

  void testInitialzation(const Teuchos::RCP<Teuchos::ParameterList>& ipb);

  RCP<LinearObjFactory<panzer::Traits> > buildModel(
                                        std::vector<Teuchos::RCP<panzer::PhysicsBlock> > & physics_blocks,
                                        panzer::ClosureModelFactory_TemplateManager<panzer::Traits> & cm_factory,
                                        Teuchos::ParameterList & closure_models,
                                        Teuchos::ParameterList & user_data,
										RCP<panzer_stk::STK_Interface> mesh,
										Teuchos::RCP<panzer::WorksetContainer> wkstContainer,
										RCP<panzer::WorksetContainer> wkstContainer2);

  TEUCHOS_UNIT_TEST(response_library, test_surface)
  {

    std::vector<Teuchos::RCP<panzer::PhysicsBlock> > physics_blocks;
    panzer::ClosureModelFactory_TemplateManager<panzer::Traits> cm_factory;
    Teuchos::ParameterList closure_models("Closure Models");
	Teuchos::ParameterList res_pl("Response");
    Teuchos::ParameterList user_data("User Data");

    // setup and evaluate ResponseLibrary
    ///////////////////////////////////////////////////

    out << "Adding responses" << std::endl;

    RCP<panzer_stk::STK_Interface> mesh;
	Teuchos::RCP<panzer::WorksetContainer> wkstContainer = Teuchos::rcp(new panzer::WorksetContainer);
	Teuchos::RCP<panzer::WorksetContainer> wkstContainer2 = Teuchos::rcp(new panzer::WorksetContainer);
    RCP<panzer::LinearObjFactory<panzer::Traits> > lof
          = buildModel(physics_blocks,cm_factory,closure_models,user_data,mesh,wkstContainer,wkstContainer2);
    RCP<const panzer::GlobalIndexer> globalIndexer
        = user_data.sublist("Panzer Data").get<RCP<panzer::GlobalIndexer> >("DOF Manager");

    Teuchos::RCP<panzer::LinearObjContainer> loc = lof->buildLinearObjContainer();
    lof->initializeContainer(panzer::LinearObjContainer::X,*loc);
    Teuchos::RCP<panzer::LinearObjContainer> gloc = lof->buildGhostedLinearObjContainer();
    lof->initializeGhostedContainer(panzer::LinearObjContainer::X,*gloc);

    double iValue = -2.3;
    double tValue = 82.9;
	
    // Following new response
	Teuchos::ParameterList& response0 = res_pl.sublist("response0");
	{
	  response0.set("Type","Integral");
      response0.set<Teuchos::Array<std::string> >("Element Block Name",Teuchos::tuple<std::string>("eblock-0_0") );
      response0.set("Integrand Name","FIELD_A");
	  response0.set("DOF Name","FIELD_A");
	}
	
	Teuchos::ParameterList& response1 = res_pl.sublist("response1");
	{
	  response1.set("Type","Integral");
      response1.set<Teuchos::Array<std::string> >("Element Block Name",Teuchos::tuple<std::string>("eblock-0_0","eblock-0_0","eblock-1_0") );
      response1.set<Teuchos::Array<std::string> >("SideSet Name",Teuchos::tuple<std::string>("bottom","top","right"));
      response1.set("Integrand Name","FIELD_B");
	  response1.set("DOF Name","FIELD_B");
	}

	Teuchos::RCP<panzer::FieldManagerBuilder> fmb = Teuchos::rcp(new panzer::FieldManagerBuilder);
	std::unordered_map<std::string, std::vector<TianXin::TemplatedResponse>> respContainer;
	fmb->setWorksetContainer2(wkstContainer);
	fmb->setupResponseFieldManagers(res_pl,mesh,physics_blocks,*lof,cm_factory,closure_models,user_data,respContainer);

	for( const auto& resps : respContainer )
	{
		const auto& ev0 = resps.second[0].get<panzer::Traits::Residual>();
		const auto& map= ev0->getMap();
		Teuchos::RCP<Tpetra::Vector<double, int, panzer::GlobalOrdinal>> rvec = Teuchos::rcp(new Tpetra::Vector<double, int, panzer::GlobalOrdinal>(map));
		for( auto& ev : resps.second )
			ev.get<panzer::Traits::Residual>()->setVector(rvec);
	}
	
	const std::vector< std::shared_ptr< PHX::FieldManager<panzer::Traits> > >
		rfm = fmb->getResponseFieldManager();
	const std::vector<WorksetDescriptor> & wkstDesc = fmb->getResponseWorksetDescriptors();
	// Loop over response field managers
	for (std::size_t block = 0; block < rfm.size(); ++block) {
		const WorksetDescriptor& wd = wkstDesc[block];
		std::shared_ptr< PHX::FieldManager<panzer::Traits> > fm = rfm[block];
		Teuchos::RCP<panzer::Workset> workset;
		if( wd.useSideset() ) {
		  workset = wkstContainer->getSideWorkset(wd);
	    } else {
			Teuchos::RCP<std::vector<Workset> > wksts = wkstContainer->getWorksets(wd);
			workset = Teuchos::rcpFromRef((*wksts)[0]);
		}
		TEUCHOS_TEST_FOR_EXCEPTION(workset == Teuchos::null, std::logic_error,
                         "Failed to find corresponding bc workset!");

		panzer::Traits::PED ped;
		fm->template preEvaluate<panzer::Traits::Residual>(ped);
		fm->evaluateFields<panzer::Traits::Residual>(*workset);
		fm->postEvaluate<panzer::Traits::Residual>(0);
		
		//for(PHX::FieldManager<panzer::Traits>::iterator fd=fm->begin(); fd!=fm->end(); ++fd) {
		//	fd->print(std::cout);
		//}
	}

	{ 
		const auto& resp0 = respContainer["RESPONSE_FIELD_A"];
		const auto& ev = resp0[0].get<panzer::Traits::Residual>();
		const auto& vec= ev->getVector();
		const auto& array = vec->getData(0);
		TEST_FLOATING_EQUALITY(array[0],0.5*tValue,1e-14);
	}
	{ 
		const auto& resp0 = respContainer["RESPONSE_FIELD_B"];
		const auto& ev = resp0[0].get<panzer::Traits::Residual>();
		const auto& vec= ev->getVector();
		const auto& array = vec->getData(0);
		TEST_FLOATING_EQUALITY(array[0],2.0*iValue,1e-14);
	}
  }

  void testInitialzation(const Teuchos::RCP<Teuchos::ParameterList>& ipb)
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
  }

  RCP<LinearObjFactory<panzer::Traits> > buildModel(
                                        std::vector<Teuchos::RCP<panzer::PhysicsBlock> > & physics_blocks,
                                        panzer::ClosureModelFactory_TemplateManager<panzer::Traits> & cm_factory,
                                        Teuchos::ParameterList & closure_models,
                                        Teuchos::ParameterList & user_data, 
										RCP<panzer_stk::STK_Interface> mesh,
										RCP<panzer::WorksetContainer> wkstContainer,
										RCP<panzer::WorksetContainer> wkstContainer2 )
  {
    using Teuchos::RCP;

  #ifdef HAVE_MPI
     Teuchos::RCP<Teuchos::Comm<int> > tcomm = Teuchos::rcp(new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(MPI_COMM_WORLD)));
  #else
     Teuchos::RCP<Teuchos::Comm<int> > tcomm = Teuchos::rcp(new Teuchos::SerialComm<int>);
  #endif

    panzer_stk::SquareQuadMeshFactory mesh_factory;
    Teuchos::RCP<user_app::MyFactory> eqset_factory = Teuchos::rcp(new user_app::MyFactory);
    const std::size_t workset_size = 20;

    // setup mesh
    /////////////////////////////////////////////
   // RCP<panzer_stk::STK_Interface> mesh;
    {
       RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
       pl->set("X Blocks",2);
       pl->set("Y Blocks",1);
       pl->set("X Elements",4);
       pl->set("Y Elements",4);
       mesh_factory.setParameterList(pl);
       mesh = mesh_factory.buildMesh(MPI_COMM_WORLD);
    }

    // setup physic blocks
    /////////////////////////////////////////////
    Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList("Physics Blocks");
    {
       testInitialzation(ipb);

       std::map<std::string,std::string> block_ids_to_physics_ids;
       block_ids_to_physics_ids["eblock-0_0"] = "test physics";
       block_ids_to_physics_ids["eblock-1_0"] = "test physics";

       std::map<std::string,Teuchos::RCP<const shards::CellTopology> > block_ids_to_cell_topo;
       block_ids_to_cell_topo["eblock-0_0"] = mesh->getCellTopology("eblock-0_0");
       block_ids_to_cell_topo["eblock-1_0"] = mesh->getCellTopology("eblock-1_0");

       Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();

      int default_integration_order = 1;

       panzer::buildPhysicsBlocks(block_ids_to_physics_ids,
                                  block_ids_to_cell_topo,
				  ipb,
				  default_integration_order,
				  workset_size,
                                  eqset_factory,
				  gd,
		    	          false,
                                  physics_blocks);
    }

    // build WorksetContainer & setup worksets
    Teuchos::RCP<panzer_stk::WorksetFactory> wkstFactory
       = Teuchos::rcp(new panzer_stk::WorksetFactory(mesh)); // build STK workset factory
    //Teuchos::RCP<panzer::WorksetContainer> wkstContainer     // attach it to a workset container (uses lazy evaluation)
    //   = Teuchos::rcp(new panzer::WorksetContainer);
    wkstContainer->setFactory(wkstFactory);
    for(size_t i=0;i<physics_blocks.size();i++)
      wkstContainer->setNeeds(physics_blocks[i]->elementBlockID(),physics_blocks[i]->getWorksetNeeds());
    wkstContainer->setWorksetSize(workset_size);

	wkstContainer2->setFactory(wkstFactory);
    for(size_t i=0;i<physics_blocks.size();i++)
      wkstContainer2->setNeeds(physics_blocks[i]->elementBlockID(),physics_blocks[i]->getWorksetNeeds());
    wkstContainer2->setWorksetSize(workset_size);

    // setup DOF manager
    /////////////////////////////////////////////
    const Teuchos::RCP<panzer::ConnManager> conn_manager
           = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));

    Teuchos::RCP<const panzer::GlobalIndexerFactory > indexerFactory
          = Teuchos::rcp(new panzer::DOFManagerFactory);
    const Teuchos::RCP<panzer::GlobalIndexer> dofManager
          = indexerFactory->buildGlobalIndexer(Teuchos::opaqueWrapper(MPI_COMM_WORLD),physics_blocks,conn_manager);

    // and linear object factory
    Teuchos::RCP<const Teuchos::MpiComm<int> > tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
    Teuchos::RCP<panzer::BlockedEpetraLinearObjFactory<panzer::Traits,int> > elof
          = Teuchos::rcp(new panzer::BlockedEpetraLinearObjFactory<panzer::Traits,int>(tComm.getConst(),dofManager));

    Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits> > lof = elof;

    // setup field manager builder
    /////////////////////////////////////////////

    // Add in the application specific closure model factory
    user_app::STKModelFactory_TemplateBuilder cm_builder;
    cm_factory.buildObjects(cm_builder);

    double iValue = -2.3;
    double tValue = 82.9;

    closure_models.sublist("solid").sublist("SOURCE_TEMPERATURE").set<double>("Value",1.0);
    closure_models.sublist("solid").sublist("FIELD_A").set<double>("Value",tValue);
	closure_models.sublist("solid").sublist("Thermal Conductivity").set<double>("Value",1.0);
    closure_models.sublist("ion solid").sublist("SOURCE_ION_TEMPERATURE").set<double>("Value",1.0);
    closure_models.sublist("ion solid").sublist("FIELD_B").set<double>("Value",iValue);
	closure_models.sublist("ion solid").sublist("ION_Thermal Conductivity").set<double>("Value",1.0);

    user_data.sublist("Panzer Data").set("Mesh", mesh);
    user_data.sublist("Panzer Data").set("DOF Manager", dofManager);
    user_data.sublist("Panzer Data").set("Linear Object Factory", lof);
    user_data.set<int>("Workset Size",workset_size);

    return lof;
  }

}
