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

#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include "PanzerAdaptersSTK_config.hpp"
#include "Panzer_GlobalData.hpp"
#include "Panzer_Workset_Builder.hpp"
#include "Panzer_WorksetContainer.hpp"
#include "Panzer_AssemblyEngine.hpp"
#include "Panzer_AssemblyEngine_InArgs.hpp"
#include "Panzer_AssemblyEngine_TemplateManager.hpp"
#include "Panzer_AssemblyEngine_TemplateBuilder.hpp"
#include "Panzer_LinearObjFactory.hpp"
#include "Panzer_TpetraLinearObjFactory.hpp"
#include "Panzer_DOFManagerFactory.hpp"
#include "Panzer_FieldManagerBuilder.hpp"
#include "Panzer_PureBasis.hpp"
#include "Panzer_GlobalData.hpp"
#include "Panzer_ResponseLibrary.hpp"
#include "Panzer_ResponseEvaluatorFactory_Functional.hpp"
#include "Panzer_Response_Functional.hpp"

#include "PanzerAdaptersSTK_config.hpp"
#include "Panzer_STK_WorksetFactory.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SquareTriMeshFactory.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "TianXin_STK_Utilities.hpp"

#include "Example_ClosureModel_Factory_TemplateBuilder.hpp"
#include "Example_EquationSetFactory.hpp"

#include "MatrixMarket_Tpetra.hpp"

#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosPseudoBlockGmresSolMgr.hpp"

#include <sstream>

using Teuchos::RCP;
using Teuchos::rcp;

int main(int argc,char * argv[])
{
   using Teuchos::RCP;
   using Teuchos::rcp_dynamic_cast;
   using panzer::StrPureBasisPair;
   using panzer::StrPureBasisComp;
   typedef Tpetra::MultiVector<double,int,panzer::GlobalOrdinal> MV;
   typedef Tpetra::Operator<double,int,panzer::GlobalOrdinal> OP;
   using TpetraCrsMatrix = Tpetra::CrsMatrix<double,int,panzer::GlobalOrdinal>;

   Teuchos::GlobalMPISession mpiSession(&argc,&argv);
   Kokkos::initialize(argc,argv);
   RCP<const Teuchos::MpiComm<int> > tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
   Teuchos::FancyOStream out(Teuchos::rcpFromRef(std::cout));
   out.setOutputToRootOnly(0);
   out.setShowProcRank(true);

   // Build command line processor
   ////////////////////////////////////////////////////

   int x_elements=10,y_elements=10,basis_order=1;
   std::string celltype = "Quad"; // or "Tri"
   Teuchos::CommandLineProcessor clp;
   clp.throwExceptions(false);
   clp.setDocString("This example solves Poisson problem with Quad and Tri inline mesh with high order.\n");

   clp.setOption("cell",&celltype);
   clp.setOption("x-elements",&x_elements);
   clp.setOption("y-elements",&y_elements);
   clp.setOption("basis-order",&basis_order); 

   // parse commandline argument
   Teuchos::CommandLineProcessor::EParseCommandLineReturn r_parse= clp.parse( argc, argv );
   if (r_parse == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) return  0;
   if (r_parse != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL  ) return -1;

   // variable declarations
   ////////////////////////////////////////////////////

   // factory definitions
   Teuchos::RCP<Example::EquationSetFactory> eqset_factory = 
     Teuchos::rcp(new Example::EquationSetFactory); // where poison equation is defined

   Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory;
   if      (celltype == "Quad") mesh_factory = Teuchos::rcp(new panzer_stk::SquareQuadMeshFactory);
   else if (celltype == "Tri")  mesh_factory = Teuchos::rcp(new panzer_stk::SquareTriMeshFactory);
   else 
     throw std::runtime_error("not supported celltype argument: try Quad or Tri");
   
   // other declarations
   const std::size_t workset_size = 20;

   // construction of uncommitted (no elements) mesh 
   ////////////////////////////////////////////////////////

   // set mesh factory parameters
   RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
   pl->set("X Blocks",1);
   pl->set("Y Blocks",1);
   pl->set("X Elements",x_elements);
   pl->set("Y Elements",y_elements);
   mesh_factory->setParameterList(pl);

   RCP<panzer_stk::STK_Interface> mesh = mesh_factory->buildUncommitedMesh(MPI_COMM_WORLD);
   
   Teuchos::ParameterList pldiric("Dirichlet");
   {	
	   Teuchos::ParameterList& p0 = pldiric.sublist("a");  // noname sublist
	   p0.set("NodeSet Name","left");
	   p0.set("Value Type","Constant");
       p0.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "TEMPERATURE" ));
	   Teuchos::ParameterList pl_sub("Constant");
	   pl_sub.set("Value",0.0);
	   p0.set("Constant",pl_sub);
	   
	   Teuchos::ParameterList& p1 = pldiric.sublist("b");  // noname sublist
	   p1.set("NodeSet Name","top");
	   p1.set("Value Type","Constant");
       p1.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "TEMPERATURE" ));
	   p1.set("Constant",pl_sub);
	   
	   Teuchos::ParameterList& p2 = pldiric.sublist("c");  // noname sublist
	   p2.set("NodeSet Name","right");
	   p2.set("Value Type","Constant");
       p2.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "TEMPERATURE" ));
	   p2.set("Constant",pl_sub);
	   
	   Teuchos::ParameterList& p3 = pldiric.sublist("d");  // noname sublist
	   p3.set("NodeSet Name","bottom");
	   p3.set("Value Type","Constant");
       p3.set<Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>( "TEMPERATURE" ));
	   p3.set("Constant",pl_sub);
   }

   // construct input physics and physics block
   ////////////////////////////////////////////////////////

   Teuchos::RCP<Teuchos::ParameterList> ipb = Teuchos::parameterList("Physics Blocks");
   std::vector<RCP<panzer::PhysicsBlock> > physicsBlocks;
   {
      bool build_transient_support = false;

      {
		const int integration_order = 10;
		Teuchos::ParameterList& p = ipb->sublist("Poisson Physics");
		p.set("Type","Poisson");
		p.set("Model ID","solid");
		p.set("Basis Type","HGrad");
		p.set("Basis Order",basis_order);
		p.set("Integration Order",integration_order);
	  }
      
      const panzer::CellData volume_cell_data(workset_size, mesh->getCellTopology("eblock-0_0"));

      // GobalData sets ostream and parameter interface to physics
      Teuchos::RCP<panzer::GlobalData> gd = panzer::createGlobalData();

      // Can be overridden by the equation set
      int default_integration_order = 1;
      
      // the physics block nows how to build and register evaluator with the field manager
      RCP<panzer::PhysicsBlock> pb = rcp(new panzer::PhysicsBlock(ipb,
				       "eblock-0_0", 
				       default_integration_order,
				       volume_cell_data,
				       eqset_factory,
				       gd,
				       build_transient_support));

      // we can have more than one physics block, one per element block
      physicsBlocks.push_back(pb);
   }

   // finish building mesh, set required field variables and mesh bulk data
   ////////////////////////////////////////////////////////////////////////

   {
      RCP<panzer::PhysicsBlock> pb = physicsBlocks[0]; // we are assuming only one physics block

      const std::vector<StrPureBasisPair> & blockFields = pb->getProvidedDOFs();

      // insert all fields into a set
      std::set<StrPureBasisPair,StrPureBasisComp> fieldNames;
      fieldNames.insert(blockFields.begin(),blockFields.end());

      // add basis to DOF manager: block specific
      std::set<StrPureBasisPair,StrPureBasisComp>::const_iterator fieldItr;
      for (fieldItr=fieldNames.begin();fieldItr!=fieldNames.end();++fieldItr)
         mesh->addSolutionField(fieldItr->first,pb->elementBlockID());

      mesh_factory->completeMeshConstruction(*mesh,MPI_COMM_WORLD);
   }

   // build DOF Manager and linear object factory
   /////////////////////////////////////////////////////////////
 
   // build the connection manager 
   const Teuchos::RCP<panzer::ConnManager> conn_manager =
     Teuchos::rcp(new panzer_stk::STKConnManager(mesh));

   panzer::DOFManagerFactory globalIndexerFactory;
   RCP<panzer::GlobalIndexer> dofManager 
         = globalIndexerFactory.buildGlobalIndexer(Teuchos::opaqueWrapper(MPI_COMM_WORLD),physicsBlocks,conn_manager);

   // construct some linear algebra object, build object to pass to evaluators
   Teuchos::RCP<panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal> > linObjFactory
         = Teuchos::rcp(new panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal>(tComm.getConst(),dofManager));

   // build worksets
   ////////////////////////////////////////////////////////

   Teuchos::RCP<panzer_stk::WorksetFactory> wkstFactory
      = Teuchos::rcp(new panzer_stk::WorksetFactory(mesh)); // build STK workset factory
   Teuchos::RCP<panzer::WorksetContainer> wkstContainer     // attach it to a workset container (uses lazy evaluation)
       = Teuchos::rcp(new panzer::WorksetContainer);
    wkstContainer->setFactory(wkstFactory);
    for(size_t i=0;i<physicsBlocks.size();i++) 
      wkstContainer->setNeeds(physicsBlocks[i]->elementBlockID(),physicsBlocks[i]->getWorksetNeeds());
    wkstContainer->setWorksetSize(workset_size);
    wkstContainer->setGlobalIndexer(dofManager);

   // Setup response library for checking the error in this manufactered solution
   ////////////////////////////////////////////////////////////////////////

   Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits> > errorResponseLibrary
      = Teuchos::rcp(new panzer::ResponseLibrary<panzer::Traits>(wkstContainer,dofManager,linObjFactory));

   {
     const int integration_order = 10;

     std::vector<std::string> eBlocks;
     mesh->getElementBlockNames(eBlocks);

     panzer::FunctionalResponse_Builder<int,int> builder;
     builder.comm = MPI_COMM_WORLD;
     builder.cubatureDegree = integration_order;
     builder.requiresCellIntegral = true;
     builder.quadPointField = "TEMPERATURE_L2_ERROR";

     errorResponseLibrary->addResponse("L2 Error",eBlocks,builder);

     builder.comm = MPI_COMM_WORLD;
     builder.cubatureDegree = integration_order;
     builder.requiresCellIntegral = true;
     builder.quadPointField = "TEMPERATURE_H1_ERROR";

     errorResponseLibrary->addResponse("H1 Error",eBlocks,builder);
   }


   // setup closure model
   /////////////////////////////////////////////////////////////
 
   // Add in the application specific closure model factory
   panzer::ClosureModelFactory_TemplateManager<panzer::Traits> cm_factory; 
   Example::ClosureModelFactory_TemplateBuilder cm_builder;
   cm_factory.buildObjects(cm_builder);

   Teuchos::ParameterList closure_models("Closure Models");
   {
     closure_models.sublist("solid").sublist("SOURCE_TEMPERATURE").set<std::string>("Type","SIMPLE SOURCE"); // a constant source
      // SOURCE_TEMPERATURE field is required by the PoissonEquationSet
     // required for error calculation
     closure_models.sublist("solid").sublist("TEMPERATURE_L2_ERROR").set<std::string>("Type","L2 ERROR_CALC");
     closure_models.sublist("solid").sublist("TEMPERATURE_L2_ERROR").set<std::string>("Field A","TEMPERATURE");
     closure_models.sublist("solid").sublist("TEMPERATURE_L2_ERROR").set<std::string>("Field B","TEMPERATURE_EXACT");

     closure_models.sublist("solid").sublist("TEMPERATURE_H1_ERROR").set<std::string>("Type","H1 ERROR_CALC");
     closure_models.sublist("solid").sublist("TEMPERATURE_H1_ERROR").set<std::string>("Field A","TEMPERATURE");
     closure_models.sublist("solid").sublist("TEMPERATURE_H1_ERROR").set<std::string>("Field B","TEMPERATURE_EXACT");

     closure_models.sublist("solid").sublist("TEMPERATURE_EXACT").set<std::string>("Type","TEMPERATURE_EXACT");
   }

   Teuchos::ParameterList user_data("User Data"); // user data can be empty here

   // setup field manager builder
   /////////////////////////////////////////////////////////////

   Teuchos::RCP<panzer::FieldManagerBuilder> fmb = 
         Teuchos::rcp(new panzer::FieldManagerBuilder);
   fmb->setWorksetContainer(wkstContainer);
   fmb->setupVolumeFieldManagers(physicsBlocks,cm_factory,closure_models,*linObjFactory,user_data);
   //fmb->setupBCFieldManagers(bcs,physicsBlocks,*eqset_factory,cm_factory,bc_factory,closure_models,
   //                          *linObjFactory,user_data);
   fmb->setupDiricheltFieldManagers(pldiric,mesh,dofManager);

   fmb->writeVolumeGraphvizDependencyFiles("Poisson", physicsBlocks);

   // setup assembly engine
   /////////////////////////////////////////////////////////////

   // build assembly engine: The key piece that brings together everything and 
   //                        drives and controls the assembly process. Just add
   //                        matrices and vectors
   panzer::AssemblyEngine_TemplateManager<panzer::Traits> ae_tm;
   panzer::AssemblyEngine_TemplateBuilder builder(fmb,linObjFactory);
   ae_tm.buildObjects(builder);

   // Finalize construcition of STK writer response library
   /////////////////////////////////////////////////////////////
   {
      user_data.set<int>("Workset Size",workset_size);
      errorResponseLibrary->buildResponseEvaluators(physicsBlocks,
                                                    cm_factory,
                                                    closure_models,
                                                    user_data);
   }

   // assemble linear system
   /////////////////////////////////////////////////////////////

   // build linear algebra objects: Ghost is for parallel assembly, it contains
   //                               local element contributions summed, the global IDs
   //                               are not unique. The non-ghosted or "global"
   //                               container will contain the sum over all processors
   //                               of the ghosted objects. The global indices are unique.
   RCP<panzer::LinearObjContainer> ghostCont = linObjFactory->buildGhostedLinearObjContainer();
   RCP<panzer::LinearObjContainer> container = linObjFactory->buildLinearObjContainer();
   linObjFactory->initializeGhostedContainer(panzer::LinearObjContainer::X |
                                             panzer::LinearObjContainer::F |
                                             panzer::LinearObjContainer::Mat,*ghostCont);
   linObjFactory->initializeContainer(panzer::LinearObjContainer::X |
                                      panzer::LinearObjContainer::F |
                                      panzer::LinearObjContainer::Mat,*container);
   ghostCont->initialize();
   container->initialize();

   panzer::AssemblyEngineInArgs input(ghostCont,container);
   input.alpha = 0;
   input.beta = 1;

   // evaluate physics: This does both the Jacobian and residual at once
   ae_tm.getAsObject<panzer::Traits::Jacobian>()->evaluate(input);
   //linObjFactory->ghostToGlobalContainer(*ghostCont,*container,panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::Mat 
   //                                   | panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::F); 
   
   // write out linear system
   if(false) {
      Tpetra::MatrixMarket::Writer<TpetraCrsMatrix>::writeSparseFile("a_op.mm",
	    *(Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(container)->get_A()));
      Tpetra::MatrixMarket::Writer<TpetraCrsMatrix>::writeDenseFile("b_vec.mm",
	    *(Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(container)->get_f()));
   }

   // solve linear system
   /////////////////////////////////////////////////////////////

   // convert generic linear object container to epetra container
   RCP<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>> ep_container 
         = rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(container);

   // Setup the linear solve: notice A is used directly 
   //Epetra_LinearProblem problem(&*ep_container->get_A(),&*ep_container->get_x(),&*ep_container->get_f()); 
   Belos::LinearProblem<double,MV,OP> problem(ep_container->get_A(), ep_container->get_x(), ep_container->get_f());
   TEUCHOS_ASSERT(problem.setProblem());

   typedef Belos::PseudoBlockGmresSolMgr<double,MV,OP> SolverType;

   Teuchos::ParameterList belosList;
   belosList.set( "Num Blocks", 3000 );            // Maximum number of blocks in Krylov factorization
   belosList.set( "Block Size", 1 );              // Blocksize to be used by iterative solver
   belosList.set( "Maximum Iterations", 50000 );       // Maximum number of iterations allowed
   belosList.set( "Maximum Restarts", 20 );      // Maximum number of restarts allowed
   belosList.set( "Convergence Tolerance", 1e-9 );         // Relative convergence tolerance requested
   belosList.set( "Verbosity", Belos::Errors + Belos::Warnings + Belos::TimingDetails + Belos::StatusTestDetails );
   belosList.set( "Output Frequency", 1 );
   belosList.set( "Output Style", 1 );

   SolverType solver(Teuchos::rcpFromRef(problem), Teuchos::rcpFromRef(belosList));

   Belos::ReturnType result = solver.solve();
   if (result == Belos::Converged)
     std::cout << "Result: Converged." << std::endl;
   else {
    TEUCHOS_ASSERT(false); // FAILURE!
   }

   // we have now solved for the residual correction from
   // zero in the context of a Newton solve.
   //     J*e = -r = -(f - J*0) where f = J*u
   // Therefore we have  J*e=-J*u which implies e = -u
   // thus we will scale the solution vector 
   ep_container->get_x()->scale(-1.0);
  
   // output data (optional)
   /////////////////////////////////////////////////////////////

   // write out linear system
   if(false) {
      Tpetra::MatrixMarket::Writer<TpetraCrsMatrix>::writeSparseFile("a_op.mm",
	    *(Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(container)->get_A()));
      Tpetra::MatrixMarket::Writer<TpetraCrsMatrix>::writeDenseFile("b_vec.mm",
	    *(Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(container)->get_f()));
      Tpetra::MatrixMarket::Writer<TpetraCrsMatrix>::writeDenseFile("x_vec.mm",
	    *(Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(container)->get_x()));
   }

   // write out solution to matrix
   if(true) {
      // redistribute solution vector to ghosted vector
      linObjFactory->globalToGhostContainer(*container,*ghostCont, panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::X 
                                                                 | panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>::DxDt); 

      // get X Epetra_Vector from ghosted container
      RCP<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>> ep_ghostCont = 
		rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(ghostCont);
      TianXin::write_solution_data(*dofManager,*mesh,*ep_ghostCont->get_x());
      // Due to multiple instances of this test being run at the same
      // time (one for each celltype and each order), we need to
      // differentiate output to prevent race conditions on output
      // file. Multiple runs for different mesh refinement levels for
      // the same celltype/order are ok as they are staged one after
      // another in the ADD_ADVANCED_TEST cmake macro.
      std::ostringstream filename;
      filename << "output_" << celltype << "_p" << basis_order << ".exo";
      mesh->writeToExodus(filename.str());
   }

   // compute error norm
   /////////////////////////////////////////////////////////////

   if(true) {
      Teuchos::FancyOStream lout(Teuchos::rcpFromRef(std::cout));
      lout.setOutputToRootOnly(0);

      panzer::AssemblyEngineInArgs respInput(ghostCont,container);
      respInput.alpha = 0;
      respInput.beta = 1;

      Teuchos::RCP<panzer::ResponseBase> l2_resp = errorResponseLibrary->getResponse<panzer::Traits::Residual>("L2 Error");
      Teuchos::RCP<panzer::Response_Functional<panzer::Traits::Residual> > l2_resp_func = 
             Teuchos::rcp_dynamic_cast<panzer::Response_Functional<panzer::Traits::Residual> >(l2_resp);
      Teuchos::RCP<Thyra::VectorBase<double> > l2_respVec = Thyra::createMember(l2_resp_func->getVectorSpace());
      l2_resp_func->setVector(l2_respVec);

      Teuchos::RCP<panzer::ResponseBase> h1_resp = errorResponseLibrary->getResponse<panzer::Traits::Residual>("H1 Error");
      Teuchos::RCP<panzer::Response_Functional<panzer::Traits::Residual> > h1_resp_func = 
             Teuchos::rcp_dynamic_cast<panzer::Response_Functional<panzer::Traits::Residual> >(h1_resp);
      Teuchos::RCP<Thyra::VectorBase<double> > h1_respVec = Thyra::createMember(h1_resp_func->getVectorSpace());
      h1_resp_func->setVector(h1_respVec);

      errorResponseLibrary->addResponsesToInArgs<panzer::Traits::Residual>(respInput);
      errorResponseLibrary->evaluate<panzer::Traits::Residual>(respInput);

      lout << "This is the Basis Order" << std::endl;
      lout << "Basis Order = " << basis_order << std::endl;
      lout << "This is the L2 Error" << std::endl;
      lout << "L2 Error = " << sqrt(l2_resp_func->value) << std::endl;
      lout << "This is the H1 Error" << std::endl;
      lout << "H1 Error = " << sqrt(h1_resp_func->value) << std::endl;
   }

   // all done!
   /////////////////////////////////////////////////////////////
   
   out << "ALL PASSED" << std::endl;

   return 0;
}
