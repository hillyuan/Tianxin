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

#include "PanzerAdaptersSTK_config.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_WorksetContainer.hpp"
#include "Panzer_Workset_Builder.hpp"
#include "Panzer_STK_WorksetFactory.hpp"
#include "Panzer_FieldManagerBuilder.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_TpetraLinearObjFactory.hpp"
#include "Panzer_AssemblyEngine.hpp"
#include "Panzer_AssemblyEngine_InArgs.hpp"
#include "Panzer_AssemblyEngine_TemplateManager.hpp"
#include "Panzer_AssemblyEngine_TemplateBuilder.hpp"
#include "Panzer_DOFManagerFactory.hpp"
#include "Panzer_GlobalData.hpp"
#include "Panzer_PauseToAttach.hpp"
#include "Panzer_ParameterLibraryUtilities.hpp"

#include "user_app_EquationSetFactory.hpp"
#include "user_app_ClosureModel_Factory_TemplateBuilder.hpp"
#include "user_app_BCStrategy_Factory.hpp"

#ifdef HAVE_MPI
   #include "Teuchos_DefaultMpiComm.hpp"
#else
   #include "NO_SERIAL_BUILD.h"
#endif

#include "Teuchos_DefaultMpiComm.hpp"
#include "Teuchos_OpaqueWrapper.hpp"

#include "Thyra_TpetraThyraWrappers.hpp"
#include "Thyra_TpetraVector.hpp"
#include "Thyra_TpetraVectorSpace.hpp"

#include "Thyra_LinearOpTester.hpp"
#include "Thyra_TestingTools.hpp"

#include <cstdio> // for get char

namespace panzer {

  void testInitialzation(const Teuchos::RCP<Teuchos::ParameterList>& ipb);

  Teuchos::RCP<const Thyra::LinearOpBase<double> >  tLinearOp;
  Teuchos::RCP<const Thyra::VectorBase<double> >  tVector;

  TEUCHOS_UNIT_TEST(assembly_engine, basic_tpetra_in_stages)
  {
    // build global communicator
    Teuchos::RCP<Teuchos::Comm<int> > comm = Teuchos::rcp(new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(MPI_COMM_WORLD)));

    RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
    pl->set("X Blocks",2);
    pl->set("Y Blocks",1);
    pl->set("X Elements",6);
    pl->set("Y Elements",4);

    panzer_stk::SquareQuadMeshFactory factory;
    factory.setParameterList(pl);
    RCP<panzer_stk::STK_Interface> mesh = factory.buildMesh(MPI_COMM_WORLD);
	
	// construct Dirichlet boundary condition
   ////////////////////////////////////////////////////////
   out << "BUILD DIRICHELET BC" << std::endl;
   Teuchos::ParameterList pldiric;
   {
	   Teuchos::ParameterList& p0 = pldiric.sublist("a");  // noname sublist
	   p0.set("ElementSet Name","eblock-0_0");
	   p0.set("NodeSet Name","left");
	   p0.set("Value Type","Constant");
       p0.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "TEMPERATURE" ));
	   Teuchos::ParameterList pl_sub("Constant");
	   pl_sub.set("Value",5.0);
	   p0.set("Constant",pl_sub);
	   
	   Teuchos::ParameterList& p1 = pldiric.sublist("b");  // noname sublist
	   p1.set("ElementSet Name","eblock-0_0");
	   p1.set("NodeSet Name","top");
	   p1.set("Value Type","Constant");
       p1.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "TEMPERATURE" ));
	   Teuchos::ParameterList pl_sub1("Constant");
	   pl_sub1.set("Value",5.0);
	   p1.set("Constant",pl_sub1);
	   
	   Teuchos::ParameterList& p3 = pldiric.sublist("d");  // noname sublist
	   p3.set("ElementSet Name","eblock-1_0");
	   p3.set("NodeSet Name","right");
	   p3.set("Value Type","Constant");
       p3.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "ION_TEMPERATURE" ));
	   Teuchos::ParameterList pl_sub3("Constant");
	   pl_sub3.set("Value",5.0);
	   p3.set("Constant",pl_sub3);
   }

    Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList("Physics Blocks");
    testInitialzation(ipb);

    Teuchos::RCP<panzer::FieldManagerBuilder> fmb =
      Teuchos::rcp(new panzer::FieldManagerBuilder);

    // build physics blocks
    //////////////////////////////////////////////////////////////
    const std::size_t workset_size = 20;
    Teuchos::RCP<user_app::MyFactory> eqset_factory = Teuchos::rcp(new user_app::MyFactory);
    std::vector<Teuchos::RCP<panzer::PhysicsBlock> > physicsBlocks;
    {
      const int default_integration_order = 1;

      std::map<std::string,std::string> block_ids_to_physics_ids;
      block_ids_to_physics_ids["eblock-0_0"] = "test physics";
      block_ids_to_physics_ids["eblock-1_0"] = "test physics";

      std::map<std::string,Teuchos::RCP<const shards::CellTopology> > block_ids_to_cell_topo;
      block_ids_to_cell_topo["eblock-0_0"] = mesh->getCellTopology("eblock-0_0");
      block_ids_to_cell_topo["eblock-1_0"] = mesh->getCellTopology("eblock-1_0");

      Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();
	  
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
	  panzer::createAndRegisterFunctor<double>(material_models,gd->functors);

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

    // build worksets
    //////////////////////////////////////////////////////////////
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

    Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits> > linObjFactory
          = Teuchos::rcp(new panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal>(comm,dofManager));

    // setup field manager build
    /////////////////////////////////////////////////////////////

    // Add in the application specific closure model factory
    panzer::ClosureModelFactory_TemplateManager<panzer::Traits> cm_factory;
    user_app::MyModelFactory_TemplateBuilder cm_builder;
    cm_factory.buildObjects(cm_builder);

    Teuchos::ParameterList closure_models("Closure Models");
    closure_models.sublist("solid").sublist("SOURCE_TEMPERATURE").set<double>("Value",1.0);
    closure_models.sublist("ion solid").sublist("SOURCE_ION_TEMPERATURE").set<double>("Value",1.0);

    Teuchos::ParameterList user_data("User Data");

    fmb->setWorksetContainer(wkstContainer);
    fmb->setupVolumeFieldManagers(physicsBlocks,cm_factory,closure_models,*linObjFactory,user_data);
    //fmb->setupBCFieldManagers(bcs,physicsBlocks,*eqset_factory,cm_factory,bc_factory,closure_models,*linObjFactory,user_data);
	fmb->setupDiricheltFieldManagers(pldiric,mesh,dofManager);

    panzer::AssemblyEngine_TemplateManager<panzer::Traits> ae_tm;
    panzer::AssemblyEngine_TemplateBuilder builder(fmb,linObjFactory);
    ae_tm.buildObjects(builder);

    RCP<panzer::LinearObjContainer> tGhosted = linObjFactory->buildGhostedLinearObjContainer();
    RCP<panzer::LinearObjContainer> tGlobal = linObjFactory->buildLinearObjContainer();
    linObjFactory->initializeGhostedContainer(panzer::LinearObjContainer::X |
                                              panzer::LinearObjContainer::DxDt |
                                              panzer::LinearObjContainer::F |
                                              panzer::LinearObjContainer::Mat,*tGhosted);
    linObjFactory->initializeContainer(panzer::LinearObjContainer::X |
                                              panzer::LinearObjContainer::DxDt |
                                              panzer::LinearObjContainer::F |
                                              panzer::LinearObjContainer::Mat,*tGlobal);

    // panzer::pauseToAttach();
    tGhosted->initialize();
    tGlobal->initialize();

    panzer::AssemblyEngineInArgs input(tGhosted,tGlobal);
    input.alpha = 0.0;
    input.beta = 1.0;

    int flags1 = panzer::AssemblyEngine<panzer::Traits::Residual>::EvaluationFlags::Initialize +panzer::AssemblyEngine<panzer::Traits::Residual>::EvaluationFlags::VolumetricFill;
    int flags2 = panzer::AssemblyEngine<panzer::Traits::Residual>::EvaluationFlags::All - flags1;
    ae_tm.getAsObject<panzer::Traits::Residual>()->evaluate(input, panzer::AssemblyEngine<panzer::Traits::Residual>::EvaluationFlags(flags1));
    ae_tm.getAsObject<panzer::Traits::Residual>()->evaluate(input, panzer::AssemblyEngine<panzer::Traits::Residual>::EvaluationFlags(flags2));
    ae_tm.getAsObject<panzer::Traits::Jacobian>()->evaluate(input);

    RCP<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal> > globalCont
       = Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal> >(tGlobal);

    Teuchos::RCP<const Tpetra::Operator<double,int,panzer::GlobalOrdinal> > baseOp = globalCont->get_A();
    Teuchos::RCP<const Thyra::VectorSpaceBase<double> > rangeSpace = Thyra::createVectorSpace<double>(baseOp->getRangeMap());
    Teuchos::RCP<const Thyra::VectorSpaceBase<double> > domainSpace = Thyra::createVectorSpace<double>(baseOp->getDomainMap());

    tLinearOp = Thyra::constTpetraLinearOp<double,int,panzer::GlobalOrdinal>(rangeSpace, domainSpace, baseOp);
    tVector = Thyra::constTpetraVector<double,int,panzer::GlobalOrdinal>(Thyra::tpetraVectorSpace<double,int,panzer::GlobalOrdinal>(baseOp->getRangeMap()).getConst(),
                                                       globalCont->get_f().getConst());
  }

  TEUCHOS_UNIT_TEST(assembly_engine, basic_tpetra)
  {
    // build global communicator
    Teuchos::RCP<Teuchos::Comm<int> > comm = Teuchos::rcp(new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(MPI_COMM_WORLD)));

    RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
    pl->set("X Blocks",2);
    pl->set("Y Blocks",1);
    pl->set("X Elements",6);
    pl->set("Y Elements",4);

    panzer_stk::SquareQuadMeshFactory factory;
    factory.setParameterList(pl);
    RCP<panzer_stk::STK_Interface> mesh = factory.buildMesh(MPI_COMM_WORLD);
	
	Teuchos::ParameterList pldiric;
   {
	   Teuchos::ParameterList& p0 = pldiric.sublist("a");  // noname sublist
	   p0.set("ElementSet Name","eblock-0_0");
	   p0.set("NodeSet Name","left");
	   p0.set("Value Type","Constant");
       p0.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "TEMPERATURE" ));
	   Teuchos::ParameterList pl_sub("Constant");
	   pl_sub.set("Value",5.0);
	   p0.set("Constant",pl_sub);
	   
	   Teuchos::ParameterList& p1 = pldiric.sublist("b");  // noname sublist
	   p1.set("ElementSet Name","eblock-0_0");
	   p1.set("NodeSet Name","top");
	   p1.set("Value Type","Constant");
       p1.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "TEMPERATURE" ));
	   Teuchos::ParameterList pl_sub1("Constant");
	   pl_sub1.set("Value",5.0);
	   p1.set("Constant",pl_sub1);
	   
	   Teuchos::ParameterList& p3 = pldiric.sublist("d");  // noname sublist
	   p3.set("ElementSet Name","eblock-1_0");
	   p3.set("NodeSet Name","right");
	   p3.set("Value Type","Constant");
       p3.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "ION_TEMPERATURE" ));
	   Teuchos::ParameterList pl_sub3("Constant");
	   pl_sub3.set("Value",5.0);
	   p3.set("Constant",pl_sub3);
   }

    Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList("Physics Blocks");
    testInitialzation(ipb);

    Teuchos::RCP<panzer::FieldManagerBuilder> fmb =
      Teuchos::rcp(new panzer::FieldManagerBuilder);

    // build physics blocks
    //////////////////////////////////////////////////////////////
    const std::size_t workset_size = 20;
    Teuchos::RCP<user_app::MyFactory> eqset_factory = Teuchos::rcp(new user_app::MyFactory);
    user_app::BCFactory bc_factory;
    std::vector<Teuchos::RCP<panzer::PhysicsBlock> > physicsBlocks;
    {
      const int default_integration_order = 1;

      std::map<std::string,std::string> block_ids_to_physics_ids;
      block_ids_to_physics_ids["eblock-0_0"] = "test physics";
      block_ids_to_physics_ids["eblock-1_0"] = "test physics";

      std::map<std::string,Teuchos::RCP<const shards::CellTopology> > block_ids_to_cell_topo;
      block_ids_to_cell_topo["eblock-0_0"] = mesh->getCellTopology("eblock-0_0");
      block_ids_to_cell_topo["eblock-1_0"] = mesh->getCellTopology("eblock-1_0");

      Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();
	  
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
	  panzer::createAndRegisterFunctor<double>(material_models,gd->functors);

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

    // build worksets
    //////////////////////////////////////////////////////////////
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

    Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits> > linObjFactory
          = Teuchos::rcp(new panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal>(comm,dofManager));

    // setup field manager build
    /////////////////////////////////////////////////////////////

    // Add in the application specific closure model factory
    panzer::ClosureModelFactory_TemplateManager<panzer::Traits> cm_factory;
    user_app::MyModelFactory_TemplateBuilder cm_builder;
    cm_factory.buildObjects(cm_builder);

    Teuchos::ParameterList closure_models("Closure Models");
    closure_models.sublist("solid").sublist("SOURCE_TEMPERATURE").set<double>("Value",1.0);
    closure_models.sublist("ion solid").sublist("SOURCE_ION_TEMPERATURE").set<double>("Value",1.0);

    Teuchos::ParameterList user_data("User Data");

    fmb->setWorksetContainer(wkstContainer);
    fmb->setupVolumeFieldManagers(physicsBlocks,cm_factory,closure_models,*linObjFactory,user_data);
    //fmb->setupBCFieldManagers(bcs,physicsBlocks,*eqset_factory,cm_factory,bc_factory,closure_models,*linObjFactory,user_data);
	fmb->setupDiricheltFieldManagers(pldiric,mesh,dofManager);

    panzer::AssemblyEngine_TemplateManager<panzer::Traits> ae_tm;
    panzer::AssemblyEngine_TemplateBuilder builder(fmb,linObjFactory);
    ae_tm.buildObjects(builder);

    RCP<panzer::LinearObjContainer> tGhosted = linObjFactory->buildGhostedLinearObjContainer();
    RCP<panzer::LinearObjContainer> tGlobal = linObjFactory->buildLinearObjContainer();
    linObjFactory->initializeGhostedContainer(panzer::LinearObjContainer::X |
                                              panzer::LinearObjContainer::DxDt |
                                              panzer::LinearObjContainer::F |
                                              panzer::LinearObjContainer::Mat,*tGhosted);
    linObjFactory->initializeContainer(panzer::LinearObjContainer::X |
                                              panzer::LinearObjContainer::DxDt |
                                              panzer::LinearObjContainer::F |
                                              panzer::LinearObjContainer::Mat,*tGlobal);

    // panzer::pauseToAttach();
    tGhosted->initialize();
    tGlobal->initialize();

    panzer::AssemblyEngineInArgs input(tGhosted,tGlobal);
    input.alpha = 0.0;
    input.beta = 1.0;

    ae_tm.getAsObject<panzer::Traits::Residual>()->evaluate(input);
    ae_tm.getAsObject<panzer::Traits::Jacobian>()->evaluate(input);

    RCP<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal> > globalCont
       = Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal> >(tGlobal);

    Teuchos::RCP<const Tpetra::Operator<double,int,panzer::GlobalOrdinal> > baseOp = globalCont->get_A();
    Teuchos::RCP<const Thyra::VectorSpaceBase<double> > rangeSpace = Thyra::createVectorSpace<double>(baseOp->getRangeMap());
    Teuchos::RCP<const Thyra::VectorSpaceBase<double> > domainSpace = Thyra::createVectorSpace<double>(baseOp->getDomainMap());

    tLinearOp = Thyra::constTpetraLinearOp<double,int,panzer::GlobalOrdinal>(rangeSpace, domainSpace, baseOp);
    tVector = Thyra::constTpetraVector<double,int,panzer::GlobalOrdinal>(Thyra::tpetraVectorSpace<double,int,panzer::GlobalOrdinal>(baseOp->getRangeMap()).getConst(),
                                                       globalCont->get_f().getConst());
  }

  TEUCHOS_UNIT_TEST(assembly_engine, z_basic_epetra_vtpetra)
  {

     TEUCHOS_ASSERT(tLinearOp!=Teuchos::null);
     //TEUCHOS_ASSERT(eLinearOp!=Teuchos::null);

     TEUCHOS_ASSERT(tVector!=Teuchos::null);
     //TEUCHOS_ASSERT(eVector!=Teuchos::null);

     Thyra::LinearOpTester<double> tester;
     tester.set_all_error_tol(1e-14);
     tester.show_all_tests(true);
     tester.dump_all(true);
     tester.num_random_vectors(200);
/*
     {
        const bool result = tester.compare( *tLinearOp, *eLinearOp, Teuchos::ptrFromRef(out) );
        TEST_ASSERT(result);
     }

     {
        const bool result = Thyra::testRelNormDiffErr(
           "Tpetra",*tVector,
           "Epetra",*eVector,
           "linear_properties_error_tol()", 1e-14,
           "linear_properties_warning_tol()", 1e-14,
           &out);
        TEST_ASSERT(result);
     }
*/

     // Need to kill global objects so that memory leaks on kokkos ref
     // count pointing doesn't trigger test failure.
    // eLinearOp = Teuchos::null;
     tLinearOp = Teuchos::null;
    // eVector = Teuchos::null;
     tVector = Teuchos::null;

  }

  void testInitialzation(const Teuchos::RCP<Teuchos::ParameterList>& ipb)
  {
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

    // BCs
   /* {
      std::size_t bc_id = 0;
      panzer::BCType neumann = BCT_Dirichlet;
      std::string sideset_id = "left";
      std::string element_block_id = "eblock-0_0";
      std::string dof_name = "TEMPERATURE";
      std::string strategy = "Constant";
      double value = 5.0;
      Teuchos::ParameterList p;
      p.set("Value",value);
      panzer::BC bc(bc_id, neumann, sideset_id, element_block_id, dof_name,
		    strategy, p);
      bcs.push_back(bc);
    }
    {
      std::size_t bc_id = 1;
      panzer::BCType neumann = BCT_Dirichlet;
      std::string sideset_id = "right";
      std::string element_block_id = "eblock-1_0";
      std::string dof_name = "TEMPERATURE";
      std::string strategy = "Constant";
      double value = 5.0;
      Teuchos::ParameterList p;
      p.set("Value",value);
      panzer::BC bc(bc_id, neumann, sideset_id, element_block_id, dof_name,
		    strategy, p);
      bcs.push_back(bc);
    }
    {
      std::size_t bc_id = 2;
      panzer::BCType neumann = BCT_Dirichlet;
      std::string sideset_id = "top";
      std::string element_block_id = "eblock-1_0";
      std::string dof_name = "TEMPERATURE";
      std::string strategy = "Constant";
      double value = 5.0;
      Teuchos::ParameterList p;
      p.set("Value",value);
      panzer::BC bc(bc_id, neumann, sideset_id, element_block_id, dof_name,
		    strategy, p);
      bcs.push_back(bc);
    }*/
  }

}
