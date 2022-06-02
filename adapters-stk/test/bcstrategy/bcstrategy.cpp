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
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Assert.hpp>

#include "Panzer_Traits.hpp"
#include "Panzer_BC.hpp"
#include "Panzer_BCStrategy.hpp"
#include "user_app_BCStrategy_Factory.hpp"
#include <iostream>

#include "PanzerAdaptersSTK_config.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STK_WorksetFactory.hpp"
#include "Panzer_Workset_Builder.hpp"
#include "Panzer_WorksetContainer.hpp"
#include "Panzer_FieldManagerBuilder.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_TpetraLinearObjFactory.hpp"
#include "Panzer_AssemblyEngine.hpp"
#include "Panzer_AssemblyEngine_InArgs.hpp"
#include "Panzer_AssemblyEngine_TemplateManager.hpp"
#include "Panzer_AssemblyEngine_TemplateBuilder.hpp"
#include "Panzer_DOFManagerFactory.hpp"
#include "Panzer_GlobalData.hpp"
#include "TianXin_Dirichlet.hpp"

#include "user_app_EquationSetFactory.hpp"
#include "user_app_ClosureModel_Factory_TemplateBuilder.hpp"
#include "Tpetra_Core.hpp"

namespace panzer {

  void testInitialzation(const Teuchos::RCP<Teuchos::ParameterList>& ipb,
			 std::vector<panzer::BC>& bcs);

  TEUCHOS_UNIT_TEST(bcstrategy, constant_Dirichlet_strategy)
  {

    using lids_type = typename Tpetra::CrsMatrix<double, int, panzer::GlobalOrdinal,panzer::TpetraNodeType>::nonconst_local_inds_host_view_type;
    using vals_type = typename Tpetra::CrsMatrix<double, int, panzer::GlobalOrdinal,panzer::TpetraNodeType>::nonconst_values_host_view_type;
    using Teuchos::RCP;

    // pause_to_attach();

    RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
    pl->set("X Blocks",1);
    pl->set("Y Blocks",1);
    pl->set("X Elements",1);
    pl->set("Y Elements",1);

    panzer_stk::SquareQuadMeshFactory factory;
    factory.setParameterList(pl);
    RCP<panzer_stk::STK_Interface> mesh = factory.buildMesh(MPI_COMM_WORLD);
    //RCP<Epetra_Comm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
	//RCP<const Teuchos::Comm<int> > Comm = Tpetra::getDefaultComm();

    Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList("Physics Blocks");
    std::vector<panzer::BC> bcs;
    testInitialzation(ipb, bcs);
	
	Teuchos::ParameterList pl_dirichlet("Dirichlet BC");
	{
      pl_dirichlet.set("NodeSet Name","bottom");
	  pl_dirichlet.set("Value Type","Constant");
      pl_dirichlet.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "TEMPERATURE" ));
	  Teuchos::ParameterList pl_sub("Constant");
	  pl_sub.set("Value",5.0);
	  pl_dirichlet.set("Constant",pl_sub);
    }
	pl_dirichlet.print();

    Teuchos::RCP<panzer::FieldManagerBuilder> fmb = Teuchos::rcp(new panzer::FieldManagerBuilder);

    // build physics blocks
    //////////////////////////////////////////////////////////////
    const std::size_t workset_size = 1;
    Teuchos::RCP<user_app::MyFactory> eqset_factory = Teuchos::rcp(new user_app::MyFactory);
    user_app::BCFactory bc_factory;
    std::vector<Teuchos::RCP<panzer::PhysicsBlock> > physicsBlocks;
    {
      std::map<std::string,std::string> block_ids_to_physics_ids;
      block_ids_to_physics_ids["eblock-0_0"] = "test physics";

      std::map<std::string,Teuchos::RCP<const shards::CellTopology> > block_ids_to_cell_topo;
      block_ids_to_cell_topo["eblock-0_0"] = mesh->getCellTopology("eblock-0_0");

      Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();

      const int default_integration_order = 1;

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

    Teuchos::RCP<const Teuchos::MpiComm<int> > tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
    panzer::DOFManagerFactory globalIndexerFactory;
    RCP<panzer::GlobalIndexer> dofManager
         = globalIndexerFactory.buildGlobalIndexer(Teuchos::opaqueWrapper(MPI_COMM_WORLD),physicsBlocks,conn_manager);

    auto dfm = std::shared_ptr<PHX::FieldManager<panzer::Traits>>( new PHX::FieldManager<panzer::Traits>() );		 
	Teuchos::RCP< TianXin::DirichletEvalautor<panzer::Traits::Residual, panzer::Traits> > dirichlet_res =
      Teuchos::rcp( new TianXin::DirichletEvalautor<panzer::Traits::Residual, panzer::Traits>(pl_dirichlet,mesh,dofManager) );
    dfm->registerEvaluator<panzer::Traits::Residual>(dirichlet_res);
    dfm->requireField<panzer::Traits::Residual>(*dirichlet_res->evaluatedFields()[0]);
	
    Teuchos::RCP< TianXin::DirichletEvalautor<panzer::Traits::Jacobian, panzer::Traits> > dirichlet_jac =
      Teuchos::rcp( new TianXin::DirichletEvalautor<panzer::Traits::Jacobian, panzer::Traits>(pl_dirichlet,mesh,dofManager) );
    dfm->registerEvaluator<panzer::Traits::Jacobian>(dirichlet_jac);
    dfm->requireField<panzer::Traits::Jacobian>(*dirichlet_jac->evaluatedFields()[0]);
	

    Teuchos::RCP<panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal> > tLinObjFactory
          = Teuchos::rcp(new panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal>(tComm.getConst(),dofManager));
    Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits> > linObjFactory = tLinObjFactory;

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
    fmb->setupBCFieldManagers(bcs,physicsBlocks,*eqset_factory,cm_factory,bc_factory,closure_models,*linObjFactory,user_data);

    panzer::AssemblyEngine_TemplateManager<panzer::Traits> ae_tm;
    panzer::AssemblyEngine_TemplateBuilder builder(fmb,linObjFactory);
    ae_tm.buildObjects(builder);

    RCP< panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal> > eGhosted
       = Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal> >(linObjFactory->buildGhostedLinearObjContainer());
    RCP<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal> > eGlobal
       = Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(linObjFactory->buildLinearObjContainer());
    tLinObjFactory->initializeGhostedContainer(panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::X |
                                               panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::DxDt |
                                               panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::F |
                                               panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::Mat,*eGhosted);
    tLinObjFactory->initializeContainer(panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::X |
                                        panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::DxDt |
                                        panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::F |
                                        panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::Mat,*eGlobal);
    panzer::AssemblyEngineInArgs input(eGhosted,eGlobal);

    RCP<Tpetra::Vector<double,int,panzer::GlobalOrdinal,panzer::TpetraNodeType>> x = 
		Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(input.container_)->get_x();

    x->putScalar(1.0);
    input.beta = 1.0;

    ae_tm.getAsObject<panzer::Traits::Residual>()->evaluate(input);
    ae_tm.getAsObject<panzer::Traits::Jacobian>()->evaluate(input);

    // Check residual values.  Evaluation should have put (x - 5.0)
    // into each residual.  With initial guess of 1.0, check to make
    // sure each entry in residual has -4.0.  Note that we are using
    // one element with same dirichlet bc on each side, so all nodes
    // have same dirichlet bc applied to it.

    RCP<Tpetra::Vector<double,int,panzer::GlobalOrdinal,panzer::TpetraNodeType>> f = 
		Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(input.container_)->get_f();
    double tol = 10.0*std::numeric_limits<double>::epsilon();
	auto f_2d = f->getLocalViewHost(Tpetra::Access::ReadOnly);
	auto f_1d = Kokkos::subview(f_2d, Kokkos::ALL (), 0);
    for (std::size_t i=0; i < 2; ++i) {  // i < f->getLocalLength()
      TEST_FLOATING_EQUALITY(f_1d(i), -4.0, tol );
    }

    // Check Jacobian values. The constrained two (0,1) have one on diagonal and zero elsewhere.
    RCP<Tpetra::CrsMatrix<double,int,panzer::GlobalOrdinal>> jac = 
		Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(input.container_)->get_A();
    for (std::size_t i=0; i < 2; ++i) {  // i < jac->getLocalNumRows()
      std::size_t numEntries = 0;
	  std::size_t sz = jac->getNumEntriesInLocalRow(i);
	  lids_type local_column_indices("ind",sz);
      vals_type values("val",sz);
	  //Teuchos::Array<int> local_column_indices(sz);
      //Teuchos::Array<double> values(sz);
      jac->getLocalRowCopy(i, local_column_indices, values, numEntries);
      for (std::size_t j=0; j < numEntries; j++) {
	    std::cout << "J(" << i << "," << local_column_indices[j] << ") = " << values[j] << std::endl;
	    if ( i == local_column_indices[j] ) {
	      TEST_FLOATING_EQUALITY(values[j], 1.0, tol);
	    }
	    else {
	      TEST_FLOATING_EQUALITY(values[j], 0.0, tol);
	    }
      }
    }

    jac->print(std::cout);

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
      p.set("Basis Order",1);
    }

    {
      std::size_t bc_id = 3;
      panzer::BCType dirichlet = BCT_Dirichlet;
      std::string sideset_id = "bottom";
      std::string element_block_id = "eblock-0_0";
      std::string dof_name = "TEMPERATURE";
      std::string strategy = "Constant";
      double value = 5.0;
      Teuchos::ParameterList p;
      p.set("Value",value);
      panzer::BC bc(bc_id, dirichlet, sideset_id, element_block_id, dof_name,
		    strategy, p);
      bcs.push_back(bc);
    }
  }

}
