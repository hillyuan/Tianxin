// @HEADER
// ***********************************************************************
//
//           TianXin: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2022) Xi Yuan
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
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Assert.hpp>

#include "Panzer_Traits.hpp"
#include <iostream>

#include "PanzerAdaptersSTK_config.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STK_WorksetFactory.hpp"
#include "Panzer_Workset_Builder.hpp"
#include "Panzer_WorksetContainer.hpp"
#include "Panzer_FieldManagerBuilder.hpp"
#include "Panzer_GlobalData.hpp"
#include "TianXin_Neumann.hpp"
#include "Panzer_ParameterLibraryUtilities.hpp"

#include "user_app_EquationSetFactory.hpp"
#include "Tpetra_Core.hpp"

namespace panzer {

  TEUCHOS_UNIT_TEST(bcstrategy, constant_Neumman_evaluator)
  {
    using Teuchos::RCP;

    RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
    pl->set("X Blocks",1);
    pl->set("Y Blocks",1);
    pl->set("X Elements",2);
    pl->set("Y Elements",1);
	pl->set("X0",0.0);
    pl->set("Y0",0.0);
	pl->set("Xf",2.0);
    pl->set("Yf",1.0);

    panzer_stk::SquareQuadMeshFactory factory;
    factory.setParameterList(pl);
    RCP<panzer_stk::STK_Interface> mesh = factory.buildMesh(MPI_COMM_WORLD);
	//mesh->writeToExodus("output.exo");
//mesh->print(std::cout);
    Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList("Physics Blocks");
    Teuchos::ParameterList& physics_block = ipb->sublist("test physics");
	physics_block.set("Material","Cu");
    {
      Teuchos::ParameterList& p = physics_block.sublist("a");
      p.set("Type","Energy");
      p.set("Prefix","");
      p.set("Model ID","solid");
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
	}

    Teuchos::RCP<panzer::FieldManagerBuilder> fmb = Teuchos::rcp(new panzer::FieldManagerBuilder);

    // build physics blocks
    //////////////////////////////////////////////////////////////
    const std::size_t workset_size = 1;
    Teuchos::RCP<user_app::MyFactory> eqset_factory = Teuchos::rcp(new user_app::MyFactory);
    std::vector<Teuchos::RCP<panzer::PhysicsBlock> > physicsBlocks;
    {
      std::map<std::string,std::string> block_ids_to_physics_ids;
      block_ids_to_physics_ids["eblock-0_0"] = "test physics";

      std::map<std::string,Teuchos::RCP<const shards::CellTopology> > block_ids_to_cell_topo;
      block_ids_to_cell_topo["eblock-0_0"] = mesh->getCellTopology("eblock-0_0");

      Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();
	  panzer::createAndRegisterFunctor<double>(material_models,gd->functors);

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
  /*  const Teuchos::RCP<panzer::ConnManager>
      conn_manager = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));

    Teuchos::RCP<const Teuchos::MpiComm<int> > tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
    panzer::DOFManagerFactory globalIndexerFactory;
    RCP<panzer::GlobalIndexer> dofManager
         = globalIndexerFactory.buildGlobalIndexer(Teuchos::opaqueWrapper(MPI_COMM_WORLD),physicsBlocks,conn_manager);*/
		 
    // Numann BC
	Teuchos::ParameterList pl_neumann("Neumann BC");
	{
	  pl_neumann.set<std::string>("Type","Flux");
	  pl_neumann.set("Element Block Name","eblock-0_0");
      pl_neumann.set("SideSet Name","left");
	  pl_neumann.set("Value Type","Constant");
      pl_neumann.set<std::string>("DOF Name","TEMPERATURE");
	  pl_neumann.set<std::string>("Residual Name","Residual");
	  Teuchos::ParameterList pl_sub("Constant");
	  pl_sub.set("Value",5.0);
	  pl_neumann.set("Constant",pl_sub);
    }
	//pl_neumann.print();

    std::shared_ptr<PHX::FieldManager<panzer::Traits> > nfm
          = std::shared_ptr<PHX::FieldManager<panzer::Traits>>( new PHX::FieldManager<panzer::Traits>());
	WorksetDescriptor wd(pl_neumann);
	const Teuchos::RCP<panzer::Workset> wkst = wkstContainer->getSideWorkset(wd);

    Teuchos::RCP<const shards::CellTopology> volume_cell_topology = physicsBlocks[0]->cellData().getCellTopology();
	const panzer::CellData side_cell_data(wkst->num_cells,1,volume_cell_topology);
	Teuchos::RCP<panzer::PhysicsBlock> side_pb = physicsBlocks[0]->copyWithCellData(side_cell_data);
	
	const std::string dof_name= pl_neumann.get<std::string>("DOF Name");
    Teuchos::RCP<const panzer::PureBasis> basis = side_pb->getBasisForDOF(dof_name);
	std::cout << basis->cardinality()<< ", " << basis->numCells()<< ", " << basis->dimension() << ", " << basis->type() << std::endl;
    const int integration_order = side_pb->getIntegrationOrder();
	Teuchos::RCP<panzer::IntegrationRule> ir = Teuchos::rcp(new panzer::IntegrationRule(integration_order,side_cell_data));
	ir->print(std::cout);std::cout << std::endl;
	pl_neumann.set<Teuchos::RCP<const panzer::PureBasis>>("Basis", basis);
	pl_neumann.set<Teuchos::RCP<const panzer::IntegrationRule>>("IR", ir.getConst());
	const std::string Identifier= pl_neumann.get<std::string>("Type");
	std::unique_ptr<TianXin::NeumannBase<panzer::Traits::Residual, panzer::Traits>> evalr = 
			TianXin::NeumannResidualFactory::Instance().Create(Identifier, pl_neumann);
	const PHX::FieldTag & ftr = evalr->getFieldTag();std::cout << ftr << std::endl;
	PHX::MDField<typename panzer::Traits::Residual::ScalarT> res_r(ftr.name(),basis->functional);
	Teuchos::RCP<TianXin::NeumannBase<panzer::Traits::Residual, panzer::Traits> > re = Teuchos::rcp(evalr.release());
	nfm->template registerEvaluator<panzer::Traits::Residual>(re);
	nfm->requireField<panzer::Traits::Residual>(*re->evaluatedFields()[0]);
	
	std::unique_ptr<TianXin::NeumannBase<panzer::Traits::Jacobian, panzer::Traits>> evalj = 
			TianXin::NeumannJacobianFactory::Instance().Create(Identifier, pl_neumann);
	std::cout << evalj->getFieldTag() << std::endl;
	Teuchos::RCP<PHX::Evaluator<panzer::Traits> > je = Teuchos::rcp(evalj.release());
	nfm->template registerEvaluator<panzer::Traits::Jacobian>(je);
	nfm->requireField<panzer::Traits::Jacobian>(*je->evaluatedFields()[0]);
	
	Traits::SD setupData;
	Teuchos::RCP<std::vector<panzer::Workset> > worksets = Teuchos::rcp(new(std::vector<panzer::Workset>));
	worksets->push_back(*wkst);
	setupData.worksets_ = worksets;
	std::vector<PHX::index_size_type> derivative_dimensions;
	derivative_dimensions.push_back(basis->cardinality());   
	nfm->setKokkosExtendedDataTypeDimensions<panzer::Traits::Jacobian>(derivative_dimensions);
	
	nfm->postRegistrationSetup(setupData);

    panzer::Traits::PED preEvalData;
    nfm->preEvaluate<panzer::Traits::Residual>(preEvalData);
    nfm->evaluateFields<panzer::Traits::Residual>(*wkst);
    nfm->postEvaluate<panzer::Traits::Residual>(0);
	
	nfm->getFieldData<panzer::Traits::Residual>(res_r);
	res_r.print(std::cout,false);std::cout << std::endl;
	TEST_EQUALITY(res_r.rank(),2);
    TEST_EQUALITY(static_cast<int>(res_r.size()),basis->numCells()*basis->cardinality());
    auto res_v = res_r.get_static_view();
    auto res_h = Kokkos::create_mirror_view ( res_v);
    Kokkos::deep_copy(res_h, res_v);
	
	// check cell 0
	//for(int v=0;v<basis->cardinality();v++)
    //{
	//	std::cout << "i= "<< v << "," << res_h(0,v) << std::endl;
    //}
	
	TEST_FLOATING_EQUALITY(Sacado::scalarValue(res_h(0,0)),2.5,1e-15);
	TEST_FLOATING_EQUALITY(Sacado::scalarValue(res_h(0,1)),0,1e-15);
	TEST_FLOATING_EQUALITY(Sacado::scalarValue(res_h(0,2)),0,1e-15);
	TEST_FLOATING_EQUALITY(Sacado::scalarValue(res_h(0,3)),2.5,1e-15);
  }

}
