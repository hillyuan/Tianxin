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

#include <string>
#include <iostream>

#include "Panzer_TpetraLinearObjFactory.hpp"
#include "Panzer_Traits.hpp"

// for testing gather/scatter construction
#include "Panzer_PureBasis.hpp"
#include "Panzer_IntegrationRule.hpp"
#include "Panzer_CellData.hpp"
#include "Panzer_PauseToAttach.hpp"

#include "UnitTest_GlobalIndexer.hpp"

#ifdef HAVE_MPI
   #include "Teuchos_DefaultMpiComm.hpp"
#else
   #include "NO_SERIAL_BUILD.h"
#endif

#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

#include "Thyra_TpetraThyraWrappers.hpp"

using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;
using Teuchos::RCP;
using Teuchos::rcpFromRef;

typedef Tpetra::MultiVector<double,int,panzer::GlobalOrdinal> MultiVector;
typedef Tpetra::Vector<double,int,panzer::GlobalOrdinal> Vector;
typedef Tpetra::CrsMatrix<double,int,panzer::GlobalOrdinal> CrsMatrix;
typedef Tpetra::CrsGraph<int,panzer::GlobalOrdinal> CrsGraph;
typedef Tpetra::Map<int,panzer::GlobalOrdinal> Map;

namespace panzer {

/*
RCP<MultiVector> getTpetraMultiVector(RCP<Thyra::MultiVectorBase<double> > & vec,const Map & map)
{
   return Thyra::get_Tpetra_MultiVector<double,int,int>(map,vec);
}
*/

TEUCHOS_UNIT_TEST(tTpetraLinearObjFactory, gather_scatter_constr)
{

   // build global (or serial communicator)
   #ifdef HAVE_MPI
      Teuchos::RCP<Teuchos::Comm<int> > tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(MPI_COMM_WORLD)));
   #else
      Teuchos::RCP<Teuchos::Comm<int> > failure_comm = THIS_,_SERIAL_BUILDS_,_SHOULD_FAIL;
   #endif

   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::rcp_dynamic_cast;

   int myRank = tComm->getRank();
   int numProc = tComm->getSize();

   // panzer::pauseToAttach();

   RCP<panzer::GlobalIndexer> indexer 
         = rcp(new unit_test::GlobalIndexer(myRank,numProc));
 
   // setup factory
   Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits> > la_factory
         = Teuchos::rcp(new panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal>(tComm.getConst(),indexer));

   // build parameter lists for gather and scatters
   //////////////////////////////////////////////////////////////////
   std::size_t numCells = 10;

   Teuchos::ParameterList gatherParams;
   {
      Teuchos::RCP<shards::CellTopology> topo = 
         Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));

      // auxiliary information needed to construct basis object
      std::string basisType = "Q1";
      panzer::CellData cellData(numCells,topo);

      // build DOF names
      RCP<std::vector<std::string> > dofNames = rcp(new std::vector<std::string>);
      dofNames->push_back("ux"); // in practice these probably would not be gathered together!
      dofNames->push_back("p");

      // build basis
      RCP<panzer::PureBasis> basis = rcp(new panzer::PureBasis(basisType,1,cellData));

      // build gather parameter list
      gatherParams.set<RCP<std::vector<std::string> > >("DOF Names",dofNames);
      gatherParams.set<RCP<std::vector<std::string> > >("Indexer Names",dofNames);
      gatherParams.set<RCP<panzer::PureBasis> >("Basis",basis);
   }

   Teuchos::ParameterList scatterParams;
   {
      Teuchos::RCP<shards::CellTopology> topo = 
         Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));

      std::string basisType = "Q1";
      panzer::CellData cellData(numCells,topo);
   
      // build basis
      RCP<const panzer::PureBasis> basis = rcp(new panzer::PureBasis(basisType,1,cellData));
   
      std::string scatterName = "Residual_NS";
   
      // build DOF names
      RCP<std::vector<std::string> > evaluatedNames = rcp(new std::vector<std::string>);
      evaluatedNames->push_back("Residual_ux"); // in practice these probably would not be scattered together!
      evaluatedNames->push_back("Residual_p");
   
      // build evaluated map
      RCP<std::map<std::string,std::string> > evaluatedMap = rcp(new std::map<std::string,std::string>);
      evaluatedMap->insert(std::make_pair("Residual_ux","ux")); // in practice these probably would not be scattered together!
      evaluatedMap->insert(std::make_pair("Residual_p","p"));

      // build scatter parameter list
      scatterParams.set<std::string>("Scatter Name",scatterName);
      scatterParams.set<RCP<std::vector<std::string> > >("Dependent Names",evaluatedNames);
      scatterParams.set<RCP<std::map<std::string,std::string> > >("Dependent Map",evaluatedMap);
      scatterParams.set("Basis",basis);
   }

   Teuchos::ParameterList scatterDirichletParams;
   {
      Teuchos::RCP<shards::CellTopology> topo = 
         Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));

      std::string basisType = "Q1";
      panzer::CellData cellData(numCells,topo);
   
      // build basis
      RCP<panzer::PureBasis> basis = rcp(new panzer::PureBasis(basisType,1,cellData));
   
      std::string scatterName = "Residual_NS";
   
      // build DOF names
      RCP<std::vector<std::string> > evaluatedNames = rcp(new std::vector<std::string>);
      evaluatedNames->push_back("Residual_ux"); // in practice these probably would not be scattered together!
      evaluatedNames->push_back("Residual_p");
   
      // build evaluated map
      RCP<std::map<std::string,std::string> > evaluatedMap = rcp(new std::map<std::string,std::string>);
      evaluatedMap->insert(std::make_pair("Residual_ux","ux")); // in practice these probably would not be scattered together!
      evaluatedMap->insert(std::make_pair("Residual_p","p"));
   
      // build scatter parameter list
      scatterDirichletParams.set<std::string>("Scatter Name",scatterName);
      scatterDirichletParams.set<RCP<std::vector<std::string> > >("Dependent Names",evaluatedNames);
      scatterDirichletParams.set<RCP<std::map<std::string,std::string> > >("Dependent Map",evaluatedMap);
      scatterDirichletParams.set<RCP<panzer::PureBasis> >("Basis",basis);
      scatterDirichletParams.set<int>("Side Subcell Dimension",1);
      scatterDirichletParams.set<int>("Local Side ID",2);
      scatterDirichletParams.set("Check Apply BC",false);
   }

   // Evaluator construction tests
   ///////////////////////////////////////////////////////
   RCP<PHX::Evaluator<panzer::Traits> > evaluator;

   {
      typedef panzer::Traits::Residual EvalType;

      // gather test
      {
         evaluator = la_factory->buildGather<EvalType>(gatherParams);

         TEST_ASSERT(evaluator!=Teuchos::null);
         RCP<GatherSolution_Tpetra<EvalType,panzer::Traits,int,panzer::GlobalOrdinal> > gatherSolutionEval 
               = rcp_dynamic_cast<GatherSolution_Tpetra<EvalType,panzer::Traits,int,panzer::GlobalOrdinal> >(evaluator);
         TEST_ASSERT(gatherSolutionEval!=Teuchos::null);
   
         const std::vector<RCP<PHX::FieldTag> > & fields = gatherSolutionEval->evaluatedFields();
         TEST_EQUALITY(fields.size(),2);
   
         TEST_EQUALITY(fields[0]->name(),"ux");
         TEST_EQUALITY(fields[1]->name(),"p");
   
         TEST_EQUALITY(fields[0]->dataLayout().extent_int(0),Teuchos::as<int>(numCells));
         TEST_EQUALITY(fields[0]->dataLayout().extent_int(1),Teuchos::as<int>(4)); // for Q1
   
         TEST_EQUALITY(fields[1]->dataLayout().extent_int(0),Teuchos::as<int>(numCells));
         TEST_EQUALITY(fields[1]->dataLayout().extent_int(1),Teuchos::as<int>(4)); // for Q1
      }

      // scatter test
      {
         evaluator = la_factory->buildScatter<EvalType>(scatterParams);

         out << "SCATTER RES NAME: \"" << evaluator->getName() << "\"" << std::endl;
         TEST_ASSERT(evaluator!=Teuchos::null);
         RCP<ScatterResidual_Tpetra<EvalType,panzer::Traits,int,panzer::GlobalOrdinal> > scatterResidual 
               = rcp_dynamic_cast<ScatterResidual_Tpetra<EvalType,panzer::Traits,int,panzer::GlobalOrdinal> >(evaluator);
         TEST_ASSERT(scatterResidual!=Teuchos::null);

         const std::vector<RCP<PHX::FieldTag> > & evalFields = scatterResidual->evaluatedFields();
         TEST_EQUALITY(evalFields.size(),1); // this is a dummy holder for the sake of the field manager
   
         const std::vector<RCP<PHX::FieldTag> > & fields = scatterResidual->dependentFields();
         TEST_EQUALITY(fields.size(),2); // these store the residual values
   
         TEST_EQUALITY(fields[0]->name(),"Residual_ux");
         TEST_EQUALITY(fields[1]->name(),"Residual_p");
   
         TEST_EQUALITY(fields[0]->dataLayout().extent_int(0),Teuchos::as<int>(numCells));
         TEST_EQUALITY(fields[0]->dataLayout().extent_int(1),Teuchos::as<int>(4)); // for Q1

         TEST_EQUALITY(fields[1]->dataLayout().extent_int(0),Teuchos::as<int>(numCells));
         TEST_EQUALITY(fields[1]->dataLayout().extent_int(1),Teuchos::as<int>(4)); // for Q1
      }

      // scatter dirichlet test
      {
         evaluator = la_factory->buildScatterDirichlet<EvalType>(scatterDirichletParams);

         out << "SCATTER DIRICHLET RES NAME: \"" << evaluator->getName() << "\"" << std::endl;
         TEST_ASSERT(evaluator!=Teuchos::null);
         RCP<ScatterDirichletResidual_Tpetra<EvalType,panzer::Traits,int,panzer::GlobalOrdinal> > scatterResidual 
               = rcp_dynamic_cast<ScatterDirichletResidual_Tpetra<EvalType,panzer::Traits,int,panzer::GlobalOrdinal> >(evaluator);
         TEST_ASSERT(scatterResidual!=Teuchos::null);

         const std::vector<RCP<PHX::FieldTag> > & evalFields = scatterResidual->evaluatedFields();
         TEST_EQUALITY(evalFields.size(),1); // this is a dummy holder for the sake of the field manager
   
         const std::vector<RCP<PHX::FieldTag> > & fields = scatterResidual->dependentFields();
         TEST_EQUALITY(fields.size(),2); // these store the residual values
   
         TEST_EQUALITY(fields[0]->name(),"Residual_ux");
         TEST_EQUALITY(fields[1]->name(),"Residual_p");
   
         TEST_EQUALITY(fields[0]->dataLayout().extent_int(0),Teuchos::as<int>(numCells));
         TEST_EQUALITY(fields[0]->dataLayout().extent_int(1),Teuchos::as<int>(4)); // for Q1
   
         TEST_EQUALITY(fields[1]->dataLayout().extent_int(0),Teuchos::as<int>(numCells));
         TEST_EQUALITY(fields[1]->dataLayout().extent_int(1),Teuchos::as<int>(4)); // for Q1
      }
   }

   {
      typedef panzer::Traits::Jacobian EvalType;

      // gather test
      {
         evaluator = la_factory->buildGather<EvalType>(gatherParams);

         TEST_ASSERT(evaluator!=Teuchos::null);
         RCP<GatherSolution_Tpetra<EvalType,panzer::Traits,int,panzer::GlobalOrdinal> > gatherSolutionEval 
               = rcp_dynamic_cast<GatherSolution_Tpetra<EvalType,panzer::Traits,int,panzer::GlobalOrdinal> >(evaluator);
         TEST_ASSERT(gatherSolutionEval!=Teuchos::null);
   
         const std::vector<RCP<PHX::FieldTag> > & fields = gatherSolutionEval->evaluatedFields();
         TEST_EQUALITY(fields.size(),2);
   
         TEST_EQUALITY(fields[0]->name(),"ux");
         TEST_EQUALITY(fields[1]->name(),"p");
   
         TEST_EQUALITY(fields[0]->dataLayout().extent_int(0),Teuchos::as<int>(numCells));
         TEST_EQUALITY(fields[0]->dataLayout().extent_int(1),Teuchos::as<int>(4)); // for Q1
   
         TEST_EQUALITY(fields[1]->dataLayout().extent_int(0),Teuchos::as<int>(numCells));
         TEST_EQUALITY(fields[1]->dataLayout().extent_int(1),Teuchos::as<int>(4)); // for Q1
      }

      // scatter test
      {
         evaluator = la_factory->buildScatter<EvalType>(scatterParams);

         TEST_ASSERT(evaluator!=Teuchos::null);
         RCP<ScatterResidual_Tpetra<EvalType,panzer::Traits,int,panzer::GlobalOrdinal> > scatterResidual 
               = rcp_dynamic_cast<ScatterResidual_Tpetra<EvalType,panzer::Traits,int,panzer::GlobalOrdinal> >(evaluator);
         TEST_ASSERT(scatterResidual!=Teuchos::null);

         const std::vector<RCP<PHX::FieldTag> > & evalFields = scatterResidual->evaluatedFields();
         TEST_EQUALITY(evalFields.size(),1); // this is a dummy holder for the sake of the field manager
   
         const std::vector<RCP<PHX::FieldTag> > & fields = scatterResidual->dependentFields();
         TEST_EQUALITY(fields.size(),2); // these store the residual values
   
         TEST_EQUALITY(fields[0]->name(),"Residual_ux");
         TEST_EQUALITY(fields[1]->name(),"Residual_p");
   
         TEST_EQUALITY(fields[0]->dataLayout().extent_int(0),Teuchos::as<int>(numCells));
         TEST_EQUALITY(fields[0]->dataLayout().extent_int(1),Teuchos::as<int>(4)); // for Q1

         TEST_EQUALITY(fields[1]->dataLayout().extent_int(0),Teuchos::as<int>(numCells));
         TEST_EQUALITY(fields[1]->dataLayout().extent_int(1),Teuchos::as<int>(4)); // for Q1
      }

      // scatter dirichlet test
      {
         evaluator = la_factory->buildScatterDirichlet<EvalType>(scatterDirichletParams);

         out << "SCATTER DIRICHLET RES NAME: \"" << evaluator->getName() << "\"" << std::endl;
         TEST_ASSERT(evaluator!=Teuchos::null);
         RCP<ScatterDirichletResidual_Tpetra<EvalType,panzer::Traits,int,panzer::GlobalOrdinal> > scatterResidual 
               = rcp_dynamic_cast<ScatterDirichletResidual_Tpetra<EvalType,panzer::Traits,int,panzer::GlobalOrdinal> >(evaluator);
         TEST_ASSERT(scatterResidual!=Teuchos::null);

         const std::vector<RCP<PHX::FieldTag> > & evalFields = scatterResidual->evaluatedFields();
         TEST_EQUALITY(evalFields.size(),1); // this is a dummy holder for the sake of the field manager
   
         const std::vector<RCP<PHX::FieldTag> > & fields = scatterResidual->dependentFields();
         TEST_EQUALITY(fields.size(),2); // these store the residual values
   
         TEST_EQUALITY(fields[0]->name(),"Residual_ux");
         TEST_EQUALITY(fields[1]->name(),"Residual_p");
   
         TEST_EQUALITY(fields[0]->dataLayout().extent_int(0),Teuchos::as<int>(numCells));
         TEST_EQUALITY(fields[0]->dataLayout().extent_int(1),Teuchos::as<int>(4)); // for Q1
   
         TEST_EQUALITY(fields[1]->dataLayout().extent_int(0),Teuchos::as<int>(numCells));
         TEST_EQUALITY(fields[1]->dataLayout().extent_int(1),Teuchos::as<int>(4)); // for Q1
      }
   }
}

TEUCHOS_UNIT_TEST(tTpetraLinearObjFactory, initializeContainer)
{

   // build global (or serial communicator)
   #ifdef HAVE_MPI
      Teuchos::RCP<Teuchos::Comm<int> > tComm = Teuchos::rcp(new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(MPI_COMM_WORLD)));
   #else
      Teuchos::RCP<Teuchos::Comm<int> > failure_comm = THIS_,_SERIAL_BUILDS_,_SHOULD_FAIL;
   #endif

   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::rcp_dynamic_cast;

   int myRank = tComm->getRank();
   int numProc = tComm->getSize();
 
   typedef TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal> LOC;

   RCP<panzer::GlobalIndexer> indexer 
         = rcp(new unit_test::GlobalIndexer(myRank,numProc));

   std::vector<panzer::GlobalOrdinal> ownedIndices, ownedAndGhostedIndices;
   indexer->getOwnedIndices(ownedIndices);
   indexer->getOwnedAndGhostedIndices(ownedAndGhostedIndices);
 
   // setup factory
   Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits> > la_factory
         = Teuchos::rcp(new panzer::TpetraLinearObjFactory<panzer::Traits,double,int,panzer::GlobalOrdinal>(tComm.getConst(),indexer));

   RCP<LinearObjContainer> container = la_factory->buildLinearObjContainer();
   RCP<LinearObjContainer> ghostedContainer = la_factory->buildGhostedLinearObjContainer();

   RCP<TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal> > tContainer = rcp_dynamic_cast<TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal> >(container);
   RCP<TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal> > tGhostedContainer = rcp_dynamic_cast<TpetraLinearObjContainer<double,int,panzer::GlobalOrdinal> >(ghostedContainer);

   // tests global initialize
   {
      // Generic code
      /////////////////////////////////////////////////////////////
   
      // test individial initializers
      la_factory->initializeContainer(LOC::X,*container);
      TEST_ASSERT(tContainer->get_x()!=Teuchos::null);
      TEST_EQUALITY(tContainer->get_dxdt(), Teuchos::null)
      TEST_EQUALITY(tContainer->get_f(),    Teuchos::null)
      TEST_EQUALITY(tContainer->get_A(),    Teuchos::null)
	TEST_EQUALITY(tContainer->get_x()->getLocalLength(),(std::size_t) ownedIndices.size());
   
      la_factory->initializeContainer(LOC::DxDt,*container);
      TEST_EQUALITY(tContainer->get_x(),    Teuchos::null)
      TEST_ASSERT(tContainer->get_dxdt()!=Teuchos::null);
      TEST_EQUALITY(tContainer->get_f(),    Teuchos::null)
      TEST_EQUALITY(tContainer->get_A(),    Teuchos::null)
	TEST_EQUALITY(tContainer->get_dxdt()->getLocalLength(),(std::size_t) ownedIndices.size());
   
      la_factory->initializeContainer(LOC::F,*container);
      TEST_EQUALITY(tContainer->get_x(),    Teuchos::null)
      TEST_EQUALITY(tContainer->get_dxdt(), Teuchos::null)
      TEST_ASSERT(tContainer->get_f()!=Teuchos::null);
      TEST_EQUALITY(tContainer->get_A(),    Teuchos::null)
	TEST_EQUALITY(tContainer->get_f()->getLocalLength(),(std::size_t) ownedIndices.size());
   
      la_factory->initializeContainer(LOC::Mat,*container);
      TEST_EQUALITY(tContainer->get_x(),    Teuchos::null)
      TEST_EQUALITY(tContainer->get_dxdt(), Teuchos::null)
      TEST_EQUALITY(tContainer->get_f(),    Teuchos::null)
      TEST_ASSERT(tContainer->get_A()!=Teuchos::null);
      TEST_EQUALITY(tContainer->get_A()->getLocalNumRows(),(std::size_t) ownedIndices.size());
   
      // jacobian and residual vector output
      la_factory->initializeContainer(LOC::F | LOC::Mat,*container);
      TEST_EQUALITY(tContainer->get_x(),    Teuchos::null)
      TEST_EQUALITY(tContainer->get_dxdt(), Teuchos::null)
      TEST_ASSERT(tContainer->get_f()!=Teuchos::null);
      TEST_ASSERT(tContainer->get_A()!=Teuchos::null);
   
      // x and time dertivative input
      la_factory->initializeContainer(LOC::X | LOC::DxDt,*container);
      TEST_ASSERT(tContainer->get_x()!=Teuchos::null);
      TEST_ASSERT(tContainer->get_dxdt()!=Teuchos::null);
      TEST_EQUALITY(tContainer->get_f(),    Teuchos::null)
      TEST_EQUALITY(tContainer->get_A(),    Teuchos::null)
   
      // everything
      la_factory->initializeContainer(LOC::X | LOC::DxDt | LOC::F | LOC::Mat,*container);
      TEST_ASSERT(tContainer->get_x()!=Teuchos::null);
      TEST_ASSERT(tContainer->get_dxdt()!=Teuchos::null);
      TEST_ASSERT(tContainer->get_f()!=Teuchos::null);
      TEST_ASSERT(tContainer->get_A()!=Teuchos::null);
   
      // Tpetra specific code
      /////////////////////////////////////////////////////////////
   
      // test individial initializers
      la_factory->initializeContainer(LOC::X,*tContainer);
      TEST_ASSERT(tContainer->get_x()!=Teuchos::null);
      TEST_EQUALITY(tContainer->get_dxdt(), Teuchos::null)
      TEST_EQUALITY(tContainer->get_f(),    Teuchos::null)
      TEST_EQUALITY(tContainer->get_A(),    Teuchos::null)
   
      la_factory->initializeContainer(LOC::DxDt,*tContainer);
      TEST_EQUALITY(tContainer->get_x(),    Teuchos::null)
      TEST_ASSERT(tContainer->get_dxdt()!=Teuchos::null);
      TEST_EQUALITY(tContainer->get_f(),    Teuchos::null)
      TEST_EQUALITY(tContainer->get_A(),    Teuchos::null)
   
      la_factory->initializeContainer(LOC::F,*tContainer);
      TEST_EQUALITY(tContainer->get_x(),    Teuchos::null)
      TEST_EQUALITY(tContainer->get_dxdt(), Teuchos::null)
      TEST_ASSERT(tContainer->get_f()!=Teuchos::null);
      TEST_EQUALITY(tContainer->get_A(),    Teuchos::null)
   
      la_factory->initializeContainer(LOC::Mat,*tContainer);
      TEST_EQUALITY(tContainer->get_x(),    Teuchos::null)
      TEST_EQUALITY(tContainer->get_dxdt(), Teuchos::null)
      TEST_EQUALITY(tContainer->get_f(),    Teuchos::null)
      TEST_ASSERT(tContainer->get_A()!=Teuchos::null);
   
      // jacobian and residual vector output
      la_factory->initializeContainer(LOC::F | LOC::Mat,*tContainer);
      TEST_EQUALITY(tContainer->get_x(),    Teuchos::null)
      TEST_EQUALITY(tContainer->get_dxdt(), Teuchos::null)
      TEST_ASSERT(tContainer->get_f()!=Teuchos::null);
      TEST_ASSERT(tContainer->get_A()!=Teuchos::null);
   
      // x and time dertivative input
      la_factory->initializeContainer(LOC::X | LOC::DxDt,*tContainer);
      TEST_ASSERT(tContainer->get_x()!=Teuchos::null);
      TEST_ASSERT(tContainer->get_dxdt()!=Teuchos::null);
      TEST_EQUALITY(tContainer->get_f(),    Teuchos::null)
      TEST_EQUALITY(tContainer->get_A(),    Teuchos::null)
   
      // everything
      la_factory->initializeContainer(LOC::X | LOC::DxDt | LOC::F | LOC::Mat,*tContainer);
      TEST_ASSERT(tContainer->get_x()!=Teuchos::null);
      TEST_ASSERT(tContainer->get_dxdt()!=Teuchos::null);
      TEST_ASSERT(tContainer->get_f()!=Teuchos::null);
      TEST_ASSERT(tContainer->get_A()!=Teuchos::null);
   }

   // tests ghosted initialize
   {
      // Generic code
      /////////////////////////////////////////////////////////////
   
      // test individial initializers
      la_factory->initializeGhostedContainer(LOC::X,*ghostedContainer);
      TEST_ASSERT(tGhostedContainer->get_x()!=Teuchos::null);
      TEST_EQUALITY(tGhostedContainer->get_dxdt(), Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_f(),    Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_A(),    Teuchos::null)
	TEST_EQUALITY(tGhostedContainer->get_x()->getLocalLength(),(std::size_t) ownedAndGhostedIndices.size());
   
      la_factory->initializeGhostedContainer(LOC::DxDt,*ghostedContainer);
      TEST_EQUALITY(tGhostedContainer->get_x(),    Teuchos::null)
      TEST_ASSERT(tGhostedContainer->get_dxdt()!=Teuchos::null);
      TEST_EQUALITY(tGhostedContainer->get_f(),    Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_A(),    Teuchos::null)
	TEST_EQUALITY(tGhostedContainer->get_dxdt()->getLocalLength(),(std::size_t) ownedAndGhostedIndices.size());
   
      la_factory->initializeGhostedContainer(LOC::F,*ghostedContainer);
      TEST_EQUALITY(tGhostedContainer->get_x(),    Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_dxdt(), Teuchos::null)
      TEST_ASSERT(tGhostedContainer->get_f()!=Teuchos::null);
      TEST_EQUALITY(tGhostedContainer->get_A(),    Teuchos::null)
	TEST_EQUALITY(tGhostedContainer->get_f()->getLocalLength(),(std::size_t) ownedAndGhostedIndices.size());
   
      la_factory->initializeGhostedContainer(LOC::Mat,*ghostedContainer);
      TEST_EQUALITY(tGhostedContainer->get_x(),    Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_dxdt(), Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_f(),    Teuchos::null)
      TEST_ASSERT(tGhostedContainer->get_A()!=Teuchos::null);
      TEST_EQUALITY(tGhostedContainer->get_A()->getLocalNumRows(),(std::size_t) ownedAndGhostedIndices.size());
   
      // jacobian and residual vector output
      la_factory->initializeGhostedContainer(LOC::F | LOC::Mat,*ghostedContainer);
      TEST_EQUALITY(tGhostedContainer->get_x(),    Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_dxdt(), Teuchos::null)
      TEST_ASSERT(tGhostedContainer->get_f()!=Teuchos::null);
      TEST_ASSERT(tGhostedContainer->get_A()!=Teuchos::null);
   
      // x and time dertivative input
      la_factory->initializeGhostedContainer(LOC::X | LOC::DxDt,*ghostedContainer);
      TEST_ASSERT(tGhostedContainer->get_x()!=Teuchos::null);
      TEST_ASSERT(tGhostedContainer->get_dxdt()!=Teuchos::null);
      TEST_EQUALITY(tGhostedContainer->get_f(),    Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_A(),    Teuchos::null)
   
      // everything
      la_factory->initializeGhostedContainer(LOC::X | LOC::DxDt | LOC::F | LOC::Mat,*ghostedContainer);
      TEST_ASSERT(tGhostedContainer->get_x()!=Teuchos::null);
      TEST_ASSERT(tGhostedContainer->get_dxdt()!=Teuchos::null);
      TEST_ASSERT(tGhostedContainer->get_f()!=Teuchos::null);
      TEST_ASSERT(tGhostedContainer->get_A()!=Teuchos::null);
   
      // Tpetra specific code
      /////////////////////////////////////////////////////////////
   
      // test individial initializers
      la_factory->initializeGhostedContainer(LOC::X,*tGhostedContainer);
      TEST_ASSERT(tGhostedContainer->get_x()!=Teuchos::null);
      TEST_EQUALITY(tGhostedContainer->get_dxdt(), Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_f(),    Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_A(),    Teuchos::null)
   
      la_factory->initializeGhostedContainer(LOC::DxDt,*tGhostedContainer);
      TEST_EQUALITY(tGhostedContainer->get_x(),    Teuchos::null)
      TEST_ASSERT(tGhostedContainer->get_dxdt()!=Teuchos::null);
      TEST_EQUALITY(tGhostedContainer->get_f(),    Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_A(),    Teuchos::null)
   
      la_factory->initializeGhostedContainer(LOC::F,*tGhostedContainer);
      TEST_EQUALITY(tGhostedContainer->get_x(),    Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_dxdt(), Teuchos::null)
      TEST_ASSERT(tGhostedContainer->get_f()!=Teuchos::null);
      TEST_EQUALITY(tGhostedContainer->get_A(),    Teuchos::null)
   
      la_factory->initializeGhostedContainer(LOC::Mat,*tGhostedContainer);
      TEST_EQUALITY(tGhostedContainer->get_x(),    Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_dxdt(), Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_f(),    Teuchos::null)
      TEST_ASSERT(tGhostedContainer->get_A()!=Teuchos::null);
   
      // jacobian and residual vector output
      la_factory->initializeGhostedContainer(LOC::F | LOC::Mat,*tGhostedContainer);
      TEST_EQUALITY(tGhostedContainer->get_x(),    Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_dxdt(), Teuchos::null)
      TEST_ASSERT(tGhostedContainer->get_f()!=Teuchos::null);
      TEST_ASSERT(tGhostedContainer->get_A()!=Teuchos::null);
   
      // x and time dertivative input
      la_factory->initializeGhostedContainer(LOC::X | LOC::DxDt,*tGhostedContainer);
      TEST_ASSERT(tGhostedContainer->get_x()!=Teuchos::null);
      TEST_ASSERT(tGhostedContainer->get_dxdt()!=Teuchos::null);
      TEST_EQUALITY(tGhostedContainer->get_f(),    Teuchos::null)
      TEST_EQUALITY(tGhostedContainer->get_A(),    Teuchos::null)
   
      // everything
      la_factory->initializeGhostedContainer(LOC::X | LOC::DxDt | LOC::F | LOC::Mat,*tGhostedContainer);
      TEST_ASSERT(tGhostedContainer->get_x()!=Teuchos::null);
      TEST_ASSERT(tGhostedContainer->get_dxdt()!=Teuchos::null);
      TEST_ASSERT(tGhostedContainer->get_f()!=Teuchos::null);
      TEST_ASSERT(tGhostedContainer->get_A()!=Teuchos::null);
   }
}

}
