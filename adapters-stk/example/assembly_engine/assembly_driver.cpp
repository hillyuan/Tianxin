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
#include <Teuchos_FancyOStream.hpp>

using Teuchos::RCP;
using Teuchos::rcp;

#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include "PanzerAdaptersSTK_config.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_Workset_Builder.hpp"
#include "Panzer_WorksetContainer.hpp"
#include "Panzer_STK_WorksetFactory.hpp"
#include "Panzer_FieldManagerBuilder.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_AssemblyEngine.hpp"
#include "Panzer_AssemblyEngine_InArgs.hpp"
#include "Panzer_AssemblyEngine_TemplateManager.hpp"
#include "Panzer_AssemblyEngine_TemplateBuilder.hpp"
#include "Panzer_TpetraLinearObjFactory.hpp"
#include "Panzer_DOFManagerFactory.hpp"
#include "Panzer_GlobalData.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_ParameterLibraryUtilities.hpp"
#include "user_app_EquationSetFactory.hpp"
#include "user_app_ClosureModel_Factory_TemplateBuilder.hpp"

#include "MatrixMarket_Tpetra.hpp"

#include "Thyra_VectorSpaceBase.hpp"
#include "Thyra_VectorBase.hpp"
#include "Thyra_LinearOpBase.hpp"
#include "Thyra_TpetraLinearOp.hpp"

#include "TianXin_STK_Utilities.hpp"

#include "Stratimikos_DefaultLinearSolverBuilder.hpp"

// calls MPI_Init and MPI_Finalize
int main(int argc,char * argv[])
{
   using Teuchos::RCP;
   using panzer::StrPureBasisPair;
   using panzer::StrPureBasisComp;
   using TpetraCrsMatrix = Tpetra::CrsMatrix<double,int,panzer::GlobalOrdinal>;
   using LOC = panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>;

   Teuchos::GlobalMPISession mpiSession(&argc,&argv);
   Kokkos::initialize(argc,argv);
   Teuchos::FancyOStream out(Teuchos::rcpFromRef(std::cout));
   out.setOutputToRootOnly(0);
   out.setShowProcRank(true);

   // variable declarations
   ////////////////////////////////////////////////////

   // factory definitions
   Teuchos::RCP<user_app::MyFactory> eqset_factory = Teuchos::rcp(new user_app::MyFactory);
   panzer_stk::SquareQuadMeshFactory mesh_factory;

   // other declarations
   const std::size_t workset_size = 20;
   Teuchos::RCP<panzer::FieldManagerBuilder> fmb = 
         Teuchos::rcp(new panzer::FieldManagerBuilder);

   RCP<panzer_stk::STK_Interface> mesh;

   // construction of uncommitted (no elements) mesh 
   ////////////////////////////////////////////////////////

   // set mesh factory parameters
   RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
   pl->set("X Blocks",2);
   pl->set("Y Blocks",1);
   pl->set("X Elements",10);
   pl->set("Y Elements",10);
   mesh_factory.setParameterList(pl);
   mesh = mesh_factory.buildUncommitedMesh(MPI_COMM_WORLD);
   
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
	   pl_sub1.set("Value",-5.0);
	   p1.set("Constant",pl_sub1);
	   
	   Teuchos::ParameterList& p2 = pldiric.sublist("c");  // noname sublist
	   p2.set("ElementSet Name","eblock-1_0");
	   p2.set("NodeSet Name","top");
	   p2.set("Value Type","Constant");
       p2.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "ION_TEMPERATURE" ));
	   Teuchos::ParameterList pl_sub2("Constant");
	   pl_sub2.set("Value",20.0);
	   p2.set("Constant",pl_sub2);
	   
	   Teuchos::ParameterList& p3 = pldiric.sublist("d");  // noname sublist
	   p3.set("ElementSet Name","eblock-1_0");
	   p3.set("NodeSet Name","right");
	   p3.set("Value Type","Constant");
       p3.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "ION_TEMPERATURE" ));
	   Teuchos::ParameterList pl_sub3("Constant");
	   pl_sub3.set("Value",-10.0);
	   p3.set("Constant",pl_sub3);
   }
   //pldiric->print();
  
   // construct input physics and physics block
   ////////////////////////////////////////////////////////

   out << "BUILD PHYSICS" << std::endl;
   Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList("Physics Blocks");
   std::vector<Teuchos::RCP<panzer::PhysicsBlock> > physicsBlocks;
   {
      std::map<std::string,std::string> block_ids_to_physics_ids;
      std::map<std::string,Teuchos::RCP<const shards::CellTopology> > block_ids_to_cell_topo;

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
		p.set("Integration Order",2);
      }
	  {
		Teuchos::ParameterList& p = physics_block.sublist("b");
		p.set("Type","Energy");
		p.set("Prefix","ION_");
		p.set("Model ID","ion solid");
		p.set("Basis Type","HGrad");
		p.set("Basis Order",1);
		p.set("Integration Order",2);
	  }

      block_ids_to_physics_ids["eblock-0_0"] = "test physics";
      block_ids_to_physics_ids["eblock-1_0"] = "test physics";

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

      int default_integration_order = 1;
      
      // build physicsBlocks map
      panzer::buildPhysicsBlocks(block_ids_to_physics_ids,
                                 block_ids_to_cell_topo,
				 ipb,
				 default_integration_order,
				 workset_size,
			         eqset_factory,
				 gd,
			         true,
			         physicsBlocks);
   }

   // finish building mesh, set required field variables and mesh bulk data
   ////////////////////////////////////////////////////////////////////////

   {
      std::vector<Teuchos::RCP<panzer::PhysicsBlock> >::const_iterator physIter;
      for(physIter=physicsBlocks.begin();physIter!=physicsBlocks.end();++physIter) {
         Teuchos::RCP<const panzer::PhysicsBlock> pb = *physIter;
         const std::vector<StrPureBasisPair> & blockFields = pb->getProvidedDOFs();

         // insert all fields into a set
         std::set<StrPureBasisPair,StrPureBasisComp> fieldNames;
         fieldNames.insert(blockFields.begin(),blockFields.end());

         // add basis to DOF manager: block specific
         std::set<StrPureBasisPair,StrPureBasisComp>::const_iterator fieldItr;
         for (fieldItr=fieldNames.begin();fieldItr!=fieldNames.end();++fieldItr) {
            mesh->addSolutionField(fieldItr->first,pb->elementBlockID());
         }
      }

      mesh_factory.completeMeshConstruction(*mesh,MPI_COMM_WORLD);
   }
   panzer::ConstructElementalPhysics(physicsBlocks,mesh);

   // build worksets
   ////////////////////////////////////////////////////////

   // build worksets
   out << "BUILD WORKSETS" << std::endl;

   Teuchos::RCP<panzer_stk::WorksetFactory> wkstFactory
      = Teuchos::rcp(new panzer_stk::WorksetFactory(mesh)); // build STK workset factory
   Teuchos::RCP<panzer::WorksetContainer> wkstContainer     // attach it to a workset container (uses lazy evaluation)
      = Teuchos::rcp(new panzer::WorksetContainer);
   wkstContainer->setFactory(wkstFactory);
   for(size_t i=0;i<physicsBlocks.size();i++) 
     wkstContainer->setNeeds(physicsBlocks[i]->elementBlockID(),physicsBlocks[i]->getWorksetNeeds());
   wkstContainer->setWorksetSize(workset_size);

   std::vector<std::string> elementBlockNames;
   mesh->getElementBlockNames(elementBlockNames);
   std::map<std::string,Teuchos::RCP<std::vector<panzer::Workset> > > volume_worksets;
   panzer::getVolumeWorksetsFromContainer(*wkstContainer,elementBlockNames,volume_worksets);

   out << "block count = " << volume_worksets.size() << std::endl;
   out << "workset count = " << volume_worksets["eblock-0_0"]->size() << std::endl;
   
   // build DOF Manager
   /////////////////////////////////////////////////////////////
 
   out << "BUILD CONN MANAGER" << std::endl;
   // build the connection manager 
   const Teuchos::RCP<panzer::ConnManager> conn_manager =
     Teuchos::rcp(new panzer_stk::STKConnManager(mesh));

   panzer::DOFManagerFactory globalIndexerFactory;
   RCP<panzer::GlobalIndexer> dofManager 
         = globalIndexerFactory.buildGlobalIndexer(Teuchos::opaqueWrapper(MPI_COMM_WORLD),physicsBlocks,conn_manager);

   // construct some linear algebra object, build object to pass to evaluators
   Teuchos::RCP<const Teuchos::MpiComm<int> > tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
   Teuchos::RCP<panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal> > eLinObjFactory
         = Teuchos::rcp(new panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal>(tComm.getConst(),dofManager));
   Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits> > linObjFactory = eLinObjFactory;

   // setup field manager build
   /////////////////////////////////////////////////////////////
   out << "SETUP FMB" << std::endl;
 
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

   // setup assembly engine
   /////////////////////////////////////////////////////////////

   // build assembly engine
   panzer::AssemblyEngine_TemplateManager<panzer::Traits> ae_tm;
   panzer::AssemblyEngine_TemplateBuilder builder(fmb,linObjFactory);
   ae_tm.buildObjects(builder);

   // setup linear algebra and solve 
   /////////////////////////////////////////////////////////////

   // build ghosted variables
   out << "BUILD LA" << std::endl;
   RCP<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>> ghostCont 
         = Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(linObjFactory->buildGhostedLinearObjContainer());
   RCP<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>> container 
         = Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(linObjFactory->buildLinearObjContainer());
   eLinObjFactory->initializeContainer(panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::X |
                                       panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::DxDt |
                                       panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::F |
                                       panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::Mat,*container);
   eLinObjFactory->initializeGhostedContainer(panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::X |
                                              panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::DxDt |
                                              panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::F |
                                              panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::Mat,*ghostCont);

   panzer::AssemblyEngineInArgs input(ghostCont,container);
   input.alpha = 0;
   input.beta = 1;

   // evaluate physics
   out << "EVALUTE" << std::endl;
   ae_tm.getAsObject<panzer::Traits::Residual>()->evaluate(input);
   ae_tm.getAsObject<panzer::Traits::Jacobian>()->evaluate(input);

   out << "RAN SUCCESSFULLY!" << std::endl;

   out << "SOLVE" << std::endl;

   // notice that this should be called by the assembly driver!
   // linObjFactory->ghostToGlobalContainer(*ghostCont,*container);

   Teuchos::RCP<Thyra::VectorBase<double> > th_x = Thyra::createVector(container->get_x()); 
   Teuchos::RCP<Thyra::VectorBase<double> > th_f = Thyra::createVector(container->get_f());

   // solve with amesos
   Stratimikos::DefaultLinearSolverBuilder solverBuilder;
   Teuchos::RCP<Teuchos::ParameterList> validList = Teuchos::rcp(new Teuchos::ParameterList(*solverBuilder.getValidParameters()));
   solverBuilder.setParameterList(validList);
  
   RCP<Thyra::LinearOpWithSolveFactoryBase<double> > lowsFactory = solverBuilder.createLinearSolveStrategy("Amesos2");
   const Teuchos::RCP<Tpetra::Operator<double,int,panzer::GlobalOrdinal> > baseOp = container->get_A();
   const Teuchos::RCP<const Thyra::VectorSpaceBase<double> > rangeSpace = Thyra::createVectorSpace<double>(baseOp->getRangeMap());
   const Teuchos::RCP<const Thyra::VectorSpaceBase<double> > domainSpace = Thyra::createVectorSpace<double>(baseOp->getDomainMap());
   Teuchos::RCP<Thyra::TpetraLinearOp<double,int,panzer::GlobalOrdinal> > tLinearOp = Thyra::tpetraLinearOp(rangeSpace, domainSpace, baseOp);
   Teuchos::RCP<const Thyra::LinearOpBase<double> > thyraA = Teuchos::rcp_dynamic_cast<const Thyra::LinearOpBase<double>>(tLinearOp);
   RCP<Thyra::LinearOpWithSolveBase<double> > lows = Thyra::linearOpWithSolve(*lowsFactory, thyraA);
   Thyra::solve<double>(*lows, Thyra::NOTRANS, *th_f, th_x.ptr());

   if(false) {
	   Tpetra::MatrixMarket::Writer<TpetraCrsMatrix>::writeSparseFile("a_op.mm",*container->get_A());
	   Tpetra::MatrixMarket::Writer<TpetraCrsMatrix>::writeDenseFile("x_vec.mm",*container->get_x());
	   Tpetra::MatrixMarket::Writer<TpetraCrsMatrix>::writeDenseFile("b_vec.mm",*container->get_f());
   }

   out << "WRITE" << std::endl;

   // redistribute solution vector
   linObjFactory->globalToGhostContainer(*container,*ghostCont,LOC::X | LOC::DxDt); 

   TianXin::write_solution_data(*dofManager,*mesh,*ghostCont->get_x());
   mesh->writeToExodus("output.exo");

   return 0;
}
