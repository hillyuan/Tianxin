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

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListHelpers.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_Assert.hpp"
#include "Teuchos_as.hpp"

#include "Panzer_NodeType.hpp"

#include "PanzerAdaptersSTK_config.hpp"
#include "Panzer_ClosureModel_Factory_TemplateManager.hpp"
#include "Panzer_ParameterLibraryUtilities.hpp"
#include "Panzer_PauseToAttach.hpp"
#include "Panzer_ResponseLibrary.hpp"
#include "Panzer_String_Utilities.hpp"
#include "Panzer_TpetraLinearObjFactory.hpp"
#include "Panzer_ElementBlockIdToPhysicsIdMap.hpp"
#include "Panzer_DOFManagerFactory.hpp"
#include "Panzer_ModelEvaluator.hpp"

#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SetupLOWSFactory.hpp"
#include "Panzer_STK_WorksetFactory.hpp"
#include "Panzer_STKConnManager.hpp"
#include "TianXin_STK_Utilities.hpp"

#include "NOX_Thyra.H"

#include "user_app_ClosureModel_Factory_TemplateBuilder.hpp"
#include "user_app_EquationSetFactory.hpp"

#include <Ioss_SerializeIO.h>

#include <string>
#include <iostream>

using ST = double;
using LO = panzer::LocalOrdinal;
using GO = panzer::GlobalOrdinal;
using NT = panzer::TpetraNodeType;
using TpetraVector = Tpetra::Vector<ST,LO,GO,NT>;
typedef Thyra::TpetraVector<ST,LO,GO,NT> Thyra_TpetraVector;
typedef Thyra::TpetraOperatorVectorExtraction<ST, LO, GO, NT>   ConverterT;

int main(int argc, char *argv[])
{
  typedef panzer::ModelEvaluator<double> PME;

  using Teuchos::RCP;
  using Teuchos::rcp;

  int status = 0;

  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);
  Kokkos::initialize(argc,argv);

  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::rcp(new Teuchos::FancyOStream(Teuchos::rcp(&std::cout,false)));
  Teuchos::RCP<Teuchos::FancyOStream> pout = Teuchos::rcp(new Teuchos::FancyOStream(Teuchos::rcp(&std::cout,false)));
  if (mpiSession.getNProc() > 1) {
    out->setShowProcRank(true);
    out->setOutputToRootOnly(0);
  }

  try {

    Teuchos::RCP<Teuchos::Time> total_time =
      Teuchos::TimeMonitor::getNewTimer("User App: Total Time");

    Teuchos::TimeMonitor timer(*total_time);

    Teuchos::RCP<const Teuchos::MpiComm<int> > comm
        = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >(Teuchos::DefaultComm<int>::getComm());

    // Parse the command line arguments
    std::string input_file_name = "input.yaml";
    {
      Teuchos::CommandLineProcessor clp;

      clp.setOption("i", &input_file_name, "User_App input yaml filename");

      Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_return =
         clp.parse(argc,argv,&std::cerr);

      TEUCHOS_TEST_FOR_EXCEPTION(parse_return != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL,
                            std::runtime_error, "Failed to parse command line!");
    }

    // Parse the input file and broadcast to other processes
    Teuchos::RCP<Teuchos::ParameterList> input_params = Teuchos::rcp(new Teuchos::ParameterList("User_App Parameters"));
    Teuchos::updateParametersFromYamlFileAndBroadcast(input_file_name, input_params.ptr(), *comm);
    //Teuchos::writeParameterListToYamlFile(*input_params,"input.yaml");

    RCP<Teuchos::ParameterList> mesh_pl             = rcp(new Teuchos::ParameterList(input_params->sublist("Mesh")));
    RCP<Teuchos::ParameterList> physics_blocks_pl   = rcp(new Teuchos::ParameterList(input_params->sublist("Physics Blocks")));
    RCP<Teuchos::ParameterList> lin_solver_pl       = rcp(new Teuchos::ParameterList(input_params->sublist("Linear Solver")));
    Teuchos::ParameterList & block_to_physics_pl    = input_params->sublist("Block ID to Physics ID Mapping");
	Teuchos::ParameterList & dirichelt_pl           = input_params->sublist("Dirichlet Conditions");
	Teuchos::ParameterList & neumann_pl             = input_params->sublist("Neumann Conditions");
    Teuchos::ParameterList & response_pl            = input_params->sublist("Responses");
    Teuchos::ParameterList & closure_models_pl      = input_params->sublist("Closure Models");
    Teuchos::ParameterList & user_data_pl           = input_params->sublist("User Data");
    Teuchos::ParameterList & nonlinsolver_pl        = input_params->sublist("Nonlinear Solver");
	Teuchos::ParameterList & material_pl            = input_params->sublist("Material");

    user_data_pl.set<RCP<const Teuchos::Comm<int> > >("Comm", comm);

    RCP<panzer::GlobalData> globalData = panzer::createGlobalData();
	panzer::createAndRegisterFunctor<double>(material_pl,globalData->functors);
    RCP<user_app::MyFactory> eqset_factory = Teuchos::rcp(new user_app::MyFactory);

    user_app::MyModelFactory_TemplateBuilder cm_builder;
    panzer::ClosureModelFactory_TemplateManager<panzer::Traits> cm_factory;
    cm_factory.buildObjects(cm_builder);

    // read in mesh database, build un committed data
    ////////////////////////////////////////////////////////////////
    RCP<panzer_stk::STK_MeshFactory> mesh_factory = rcp(new panzer_stk::SquareQuadMeshFactory);
    mesh_factory->setParameterList(mesh_pl);

    RCP<panzer_stk::STK_Interface> mesh = mesh_factory->buildUncommitedMesh(MPI_COMM_WORLD);

    // read in physics blocks
    ////////////////////////////////////////////////////////////

    std::map<std::string,std::string> block_ids_to_physics_ids;
    panzer::buildBlockIdToPhysicsIdMap(block_ids_to_physics_ids, block_to_physics_pl);

    std::map<std::string,Teuchos::RCP<const shards::CellTopology> > block_ids_to_cell_topo;
    for(auto itr=block_ids_to_physics_ids.begin();itr!=block_ids_to_physics_ids.end();itr++)
      block_ids_to_cell_topo[itr->first] = mesh->getCellTopology(itr->first);

    std::vector<Teuchos::RCP<panzer::PhysicsBlock> > physicsBlocks;

    // setup some defaults
    int workset_size = 20;
    int default_integration_order = 2;
    bool build_transient_support = false;
    std::vector<std::string> tangentParamNames;

    panzer::buildPhysicsBlocks(block_ids_to_physics_ids,
                               block_ids_to_cell_topo,
                               physics_blocks_pl,
                               default_integration_order,
                               workset_size,
                               eqset_factory,
                               globalData,
                               build_transient_support,
                               physicsBlocks,
                               tangentParamNames);

   // Add fields to the mesh data base (this is a peculiarity of how STK classic requires the
   // fields to be setup)
   //////////////////////////////////////////////////////////////////////////////////////////
   for(std::size_t i=0;i<physicsBlocks.size();i++) {
      RCP<panzer::PhysicsBlock> pb = physicsBlocks[i]; // we are assuming only one physics block
      std::cout << "PhysicsBlock:" << i << ", " << pb->getMaterialName() << std::endl;
      const std::vector<panzer::StrPureBasisPair> & blockFields = pb->getProvidedDOFs();

      // insert all fields into a set
      std::set<panzer::StrPureBasisPair,panzer::StrPureBasisComp> fieldNames;
      fieldNames.insert(blockFields.begin(),blockFields.end());

      // build string for modifiying vectors
      std::vector<std::string> dimenStr(3);
      dimenStr[0] = "X"; dimenStr[1] = "Y"; dimenStr[2] = "Z";

      // add basis to DOF manager: block specific
      std::set<panzer::StrPureBasisPair,panzer::StrPureBasisComp>::const_iterator fieldItr;
      for (fieldItr=fieldNames.begin();fieldItr!=fieldNames.end();++fieldItr) {
         Teuchos::RCP<const panzer::PureBasis> basis = fieldItr->second;
         if(basis->getElementSpace()==panzer::PureBasis::HGRAD)
            mesh->addSolutionField(fieldItr->first,pb->elementBlockID());
         else if(basis->getElementSpace()==panzer::PureBasis::HCURL) {
            for(int dim=0;dim<basis->dimension();++dim)
               mesh->addCellField(fieldItr->first+dimenStr[dim],pb->elementBlockID());
         }
      }

      mesh_factory->completeMeshConstruction(*mesh,MPI_COMM_WORLD);

      mesh->setupExodusFile("output.exo");
    }
	panzer::ConstructElementalPhysics(physicsBlocks,mesh);
	std::cout << "Size of elements=" << mesh->elementPhysics.size() << std::endl;

    // build worksets
    //////////////////////////////////////////////////////////////

    // build WorksetContainer
    Teuchos::RCP<panzer_stk::WorksetFactory> wkstFactory
       = Teuchos::rcp(new panzer_stk::WorksetFactory(mesh)); // build STK workset factory
    Teuchos::RCP<panzer::WorksetContainer> wkstContainer             // attach it to a workset container (uses lazy evaluation)
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

    // build the state dof manager and LOF
    RCP<panzer::GlobalIndexer> dofManager;
    RCP< panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal> > linObjFactory;
    {
      panzer::DOFManagerFactory globalIndexerFactory;
      dofManager = globalIndexerFactory.buildGlobalIndexer(Teuchos::opaqueWrapper(MPI_COMM_WORLD),physicsBlocks,conn_manager);
      linObjFactory = Teuchos::rcp(new panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal>(comm,dofManager));
    }

    // build linear solver
    /////////////////////////////////////////////////////////////

    RCP<Thyra::LinearOpWithSolveFactoryBase<double> > lowsFactory
        = panzer_stk::buildLOWSFactory(false, dofManager, conn_manager,
                                               Teuchos::as<int>(mesh->getDimension()),
                                               comm, lin_solver_pl,Teuchos::null);

    // build and setup model evaluatorlinear solver
    /////////////////////////////////////////////////////////////
    RCP<PME> physics = Teuchos::rcp(new PME(linObjFactory,lowsFactory,globalData,build_transient_support,0.0));
	physics->setupModel(wkstContainer,physicsBlocks,
                   *eqset_factory,
                   cm_factory, mesh, dofManager, dirichelt_pl,
                   neumann_pl, response_pl, closure_models_pl,
                   user_data_pl,false,"");

    // setup the nonlinear solver
    /////////////////////////////////////////////////////////////

    RCP<Thyra::NOXNonlinearSolver> noxSolver_obj = rcp(new Thyra::NOXNonlinearSolver);
    noxSolver_obj->setParameterList(rcp(new Teuchos::ParameterList(nonlinsolver_pl)));

    // do a nonlinear solve
    /////////////////////////////////////////////////////////////

    RCP<Thyra::VectorBase<double> > solution_vec = Thyra::createMember(physics->get_x_space());
    Thyra::assign(solution_vec.ptr(),0.0);

    {
      // set the model to use and the default parameters
      noxSolver_obj->setModel(physics);
      noxSolver_obj->setBasePoint(physics->createInArgs());

      Thyra::SolveCriteria<double> solve_criteria; // this object is ignored
      Thyra::assign(solution_vec.ptr(),0.0);
      const Thyra::SolveStatus<double> solve_status = noxSolver_obj->solve(&*solution_vec,&solve_criteria,NULL);

      TEUCHOS_TEST_FOR_EXCEPTION(
        solve_status.solveStatus != Thyra::SOLVE_STATUS_CONVERGED,
        std::runtime_error,
        "Nonlinear solver failed to converge");
    }

    // write to an exodus file
	auto tmp = Teuchos::rcp_dynamic_cast< Thyra_TpetraVector >(solution_vec,true);
	auto tvec = tmp->getTpetraVector();
	auto gvec = linObjFactory->getGhostedTpetraVector();
	linObjFactory->globalToGhostTpetraVector(*tvec,*gvec,true);
	//auto gc = physics->getGhostedContainer();
	//TianXin::write_solution_data(*dofManager,*mesh,*Teuchos::rcp_dynamic_cast<panzer::TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal>>(gc)->get_x());
    //TianXin::write_solution_data(*dofManager,*mesh,*gvec);
	TianXin::pushSolutionOnFields(*dofManager,*mesh,*gvec);
    mesh->writeToExodus("output.exo");
  }
  catch (std::exception& e) {
    *out << "*********** Caught Exception: Begin Error Report ***********" << std::endl;
    *out << e.what() << std::endl;
    *out << "************ Caught Exception: End Error Report ************" << std::endl;
    status = -1;
  }
  catch (std::string& msg) {
    *out << "*********** Caught Exception: Begin Error Report ***********" << std::endl;
    *out << msg << std::endl;
    *out << "************ Caught Exception: End Error Report ************" << std::endl;
    status = -1;
  }
  catch (...) {
    *out << "*********** Caught Exception: Begin Error Report ***********" << std::endl;
    *out << "Caught UNKOWN exception" << std::endl;
    *out << "************ Caught Exception: End Error Report ************" << std::endl;
    status = -1;
  }

  // Teuchos::TimeMonitor::summarize(*out,false,true,false);

  if (status == 0)
    *out << "TianXin::MainDriver run completed." << std::endl;

  return status;
}

