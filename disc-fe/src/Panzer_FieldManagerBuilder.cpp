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

#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "Panzer_FieldManagerBuilder.hpp"

#include "Phalanx_DataLayout_MDALayout.hpp"
#include "Phalanx_FieldManager.hpp"

#include "Teuchos_FancyOStream.hpp"

#include "Shards_CellTopology.hpp"

#include "Panzer_Traits.hpp"
#include "Panzer_Workset.hpp"
#include "Panzer_Workset_Builder.hpp"
#include "Panzer_PhysicsBlock.hpp"
#include "Panzer_BCStrategy_Factory.hpp"
#include "Panzer_BCStrategy_TemplateManager.hpp"
#include "Panzer_CellData.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_EquationSet_Factory.hpp"
#include "Panzer_GlobalIndexer.hpp"

#include "TianXin_Neumann.hpp"
#include "TianXin_Response_Integral.hpp"

//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::print(std::ostream& os) const
{
  os << "panzer::FieldManagerBuilder output:  Not implemented yet!";
}

//=======================================================================
panzer::FieldManagerBuilder::
FieldManagerBuilder(bool disablePhysicsBlockScatter,
                    bool disablePhysicsBlockGather)
  : disablePhysicsBlockScatter_(disablePhysicsBlockScatter)
  , disablePhysicsBlockGather_(disablePhysicsBlockGather)
  , active_evaluation_types_(Sacado::mpl::size<panzer::Traits::EvalTypes>::value, true)
{}

//=======================================================================
namespace {
  struct PostRegistrationFunctor {

    const std::vector<bool>& active_;
    PHX::FieldManager<panzer::Traits>& fm_;
    panzer::Traits::SD& setup_data_;

    PostRegistrationFunctor(const std::vector<bool>& active,
                            PHX::FieldManager<panzer::Traits>& fm,
                            panzer::Traits::SD& setup_data)
      : active_(active),fm_(fm),setup_data_(setup_data) {}

    template<typename T>
    void operator()(T) const {
      auto index = Sacado::mpl::find<panzer::Traits::EvalTypes,T>::value;
      if (active_[index])
        fm_.postRegistrationSetupForType<T>(setup_data_);
    }
  };
}

//=======================================================================
void panzer::FieldManagerBuilder::setupVolumeFieldManagers(
                                            const std::vector<Teuchos::RCP<panzer::PhysicsBlock> >& physicsBlocks,
                                            const std::vector<WorksetDescriptor> & wkstDesc,
					    const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>& cm_factory,
					    const Teuchos::ParameterList& closure_models,
                                            const panzer::LinearObjFactory<panzer::Traits> & lo_factory,
					    const Teuchos::ParameterList& user_data,
                                            const GenericEvaluatorFactory & gEvalFact,
                                            bool closureModelByEBlock)
{
  TEUCHOS_TEST_FOR_EXCEPTION(getWorksetContainer()==Teuchos::null,std::logic_error,
                            "panzer::FMB::setupVolumeFieldManagers: method function getWorksetContainer() returns null. "
                            "Please call setWorksetContainer() before calling this method");
  TEUCHOS_TEST_FOR_EXCEPTION(physicsBlocks.size()!=wkstDesc.size(),std::runtime_error,
                            "panzer::FMB::setupVolumeFieldManagers: physics block count must match workset descriptor count.");

  phx_volume_field_managers_.clear();

  Teuchos::RCP<const panzer::GlobalIndexer> globalIndexer = lo_factory.getRangeGlobalIndexer();

  for (std::size_t blkInd=0;blkInd<physicsBlocks.size();++blkInd) {
    Teuchos::RCP<panzer::PhysicsBlock> pb = physicsBlocks[blkInd];
    const WorksetDescriptor wd = wkstDesc[blkInd];

    Traits::SD setupData;
    setupData.worksets_ = getWorksetContainer()->getWorksets(wd);
    setupData.orientations_ = getWorksetContainer()->getOrientations();
    if(setupData.worksets_->size()==0)
      continue;

    // sanity check
    TEUCHOS_ASSERT(wd.getElementBlock()==pb->elementBlockID());

    // build a field manager object
    Teuchos::RCP<PHX::FieldManager<panzer::Traits> > fm
          = Teuchos::rcp(new PHX::FieldManager<panzer::Traits>);

    // use the physics block to register active evaluators
    pb->setActiveEvaluationTypes(active_evaluation_types_);
    pb->buildAndRegisterEquationSetEvaluators(*fm, user_data);
    if(!physicsBlockGatherDisabled())
      pb->buildAndRegisterGatherAndOrientationEvaluators(*fm,lo_factory,user_data);
    pb->buildAndRegisterDOFProjectionsToIPEvaluators(*fm,Teuchos::ptrFromRef(lo_factory),user_data);
    if(!physicsBlockScatterDisabled())
      pb->buildAndRegisterScatterEvaluators(*fm,lo_factory,user_data);

    if(closureModelByEBlock)
      pb->buildAndRegisterClosureModelEvaluators(*fm,cm_factory,pb->elementBlockID(),closure_models,user_data);
    else
      pb->buildAndRegisterClosureModelEvaluators(*fm,cm_factory,closure_models,user_data);
    pb->buildAndRegisterMaterialEvaluators(*fm,cm_factory);

    // Reset active evaluation types
    pb->activateAllEvaluationTypes();

    // register additional model evaluator from the generic evaluator factory
    gEvalFact.registerEvaluators(*fm,wd,*pb);

    // setup derivative information
    setKokkosExtendedDataTypeDimensions(wd.getElementBlock(),*globalIndexer,user_data,*fm);

    // call postRegistrationSetup() for each active type
    Sacado::mpl::for_each_no_kokkos<panzer::Traits::EvalTypes>(PostRegistrationFunctor(active_evaluation_types_,*fm,setupData));

    // make sure to add the field manager & workset to the list
    volume_workset_desc_.push_back(wd);
    phx_volume_field_managers_.push_back(fm);
  }
}

//=======================================================================
void panzer::FieldManagerBuilder::setupVolumeFieldManagers(
                                            const std::vector<Teuchos::RCP<panzer::PhysicsBlock> >& physicsBlocks,
					    const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>& cm_factory,
					    const Teuchos::ParameterList& closure_models,
                                            const panzer::LinearObjFactory<panzer::Traits> & lo_factory,
					    const Teuchos::ParameterList& user_data)
{
   std::vector<WorksetDescriptor> wkstDesc;
   for(std::size_t i=0;i<physicsBlocks.size();i++)
     wkstDesc.push_back(blockDescriptor(physicsBlocks[i]->elementBlockID()));

   EmptyEvaluatorFactory eef;
   setupVolumeFieldManagers(physicsBlocks,wkstDesc,cm_factory,closure_models,lo_factory,user_data,eef);
}

//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::
setupBCFieldManagers(const std::vector<panzer::BC> & bcs,
                     const std::vector<Teuchos::RCP<panzer::PhysicsBlock> >& physicsBlocks,
                     const Teuchos::Ptr<const panzer::EquationSetFactory>& /* eqset_factory */,
                     const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>& cm_factory,
                     const panzer::BCStrategyFactory& bc_factory,
                     const Teuchos::ParameterList& closure_models,
                     const panzer::LinearObjFactory<panzer::Traits> & lo_factory,
                     const Teuchos::ParameterList& user_data)
{
  TEUCHOS_TEST_FOR_EXCEPTION(getWorksetContainer()==Teuchos::null,std::logic_error,
                            "panzer::FMB::setupBCFieldManagers: method function getWorksetContainer() returns null. "
                            "Plase call setWorksetContainer() before calling this method");

  Teuchos::RCP<const panzer::GlobalIndexer> globalIndexer = lo_factory.getRangeGlobalIndexer();

  // for convenience build a map (element block id => physics block)
  std::map<std::string,Teuchos::RCP<panzer::PhysicsBlock> > physicsBlocks_map;
  {
     std::vector<Teuchos::RCP<panzer::PhysicsBlock> >::const_iterator blkItr;
     for(blkItr=physicsBlocks.begin();blkItr!=physicsBlocks.end();++blkItr) {
        Teuchos::RCP<panzer::PhysicsBlock> pb = *blkItr;
        std::string blockId = pb->elementBlockID();

        // add block id, physics block pair to the map
        physicsBlocks_map.insert(std::make_pair(blockId,pb));
     }
  }

  // ***************************
  // BCs
  // ***************************
  std::vector<panzer::BC>::const_iterator bc;
  for (bc=bcs.begin(); bc != bcs.end(); ++bc) {
    WorksetDescriptor wd = panzer::bcDescriptor(*bc);
    const Teuchos::RCP<std::map<unsigned,panzer::Workset> >
      currentWkst = getWorksetContainer()->getSideWorksets(wd);
    if (currentWkst.is_null()) continue;

    BCType bc_type = bc->bcType();

    if (bc_type == BCT_Interface) {
      // Loop over local face indices and setup each field manager
      for (std::map<unsigned,panzer::Workset>::const_iterator wkst = currentWkst->begin();
           wkst != currentWkst->end(); ++wkst) {
        // Build one FieldManager for each local side workset for each bc
        std::map<unsigned,PHX::FieldManager<panzer::Traits> >& field_managers =
          bc_field_managers_[*bc];

        PHX::FieldManager<panzer::Traits>& fm = field_managers[wkst->first];

        int gid_count = 0;
        for (int block_id_index = 0; block_id_index < 2; ++block_id_index) {
          const std::string element_block_id = block_id_index == 0 ? bc->elementBlockID() : bc->elementBlockID2();

          std::map<std::string,Teuchos::RCP<panzer::PhysicsBlock> >::const_iterator
            volume_pb_itr = physicsBlocks_map.find(element_block_id);

          TEUCHOS_TEST_FOR_EXCEPTION(volume_pb_itr == physicsBlocks_map.end(), std::logic_error,
            "panzer::FMB::setupBCFieldManagers: Cannot find physics block corresponding to element block \""
            << element_block_id << "\"");

          const Teuchos::RCP<const panzer::PhysicsBlock> volume_pb = physicsBlocks_map.find(element_block_id)->second;
          const Teuchos::RCP<const shards::CellTopology> volume_cell_topology = volume_pb->cellData().getCellTopology();

          // register evaluators from strategy
          const panzer::CellData side_cell_data(wkst->second.num_cells,
                                                wkst->second.details(block_id_index).subcell_index,
                                                volume_cell_topology);

          // Copy the physics block for side integrations
          Teuchos::RCP<panzer::PhysicsBlock> side_pb = volume_pb->copyWithCellData(side_cell_data);

          Teuchos::RCP<panzer::BCStrategy_TemplateManager<panzer::Traits> >
            bcstm = bc_factory.buildBCStrategy(*bc, side_pb->globalData());

          // Iterate over evaluation types
          int i=0;
          for (panzer::BCStrategy_TemplateManager<panzer::Traits>::iterator
                 bcs_type = bcstm->begin(); bcs_type != bcstm->end(); ++bcs_type,++i) {
            if (active_evaluation_types_[i]) {
              bcs_type->setDetailsIndex(block_id_index);
              side_pb->setDetailsIndex(block_id_index);
              bcs_type->setup(*side_pb, user_data);
              bcs_type->buildAndRegisterEvaluators(fm, *side_pb, cm_factory, closure_models, user_data);
              bcs_type->buildAndRegisterGatherAndOrientationEvaluators(fm, *side_pb, lo_factory, user_data);
              if ( ! physicsBlockScatterDisabled())
                bcs_type->buildAndRegisterScatterEvaluators(fm, *side_pb, lo_factory, user_data);
            }
          }

          gid_count += globalIndexer->getElementBlockGIDCount(element_block_id);
        }

        { // Use gid_count to set up the derivative information.
          std::vector<PHX::index_size_type> derivative_dimensions;
          derivative_dimensions.push_back(gid_count);
          fm.setKokkosExtendedDataTypeDimensions<panzer::Traits::Jacobian>(derivative_dimensions);

          #ifdef Panzer_BUILD_HESSIAN_SUPPORT
            fm.setKokkosExtendedDataTypeDimensions<panzer::Traits::Hessian>(derivative_dimensions);
          #endif

          derivative_dimensions[0] = 1;
          if (user_data.isType<int>("Tangent Dimension"))
            derivative_dimensions[0] = user_data.get<int>("Tangent Dimension");
          fm.setKokkosExtendedDataTypeDimensions<panzer::Traits::Tangent>(derivative_dimensions);
        }

        // Set up the field manager
        Traits::SD setupData;
        Teuchos::RCP<std::vector<panzer::Workset> > worksets = Teuchos::rcp(new std::vector<panzer::Workset>);
        worksets->push_back(wkst->second);
        setupData.worksets_ = worksets;
        setupData.orientations_ = getWorksetContainer()->getOrientations();

        Sacado::mpl::for_each_no_kokkos<panzer::Traits::EvalTypes>(PostRegistrationFunctor(active_evaluation_types_,fm,setupData));

      }
    } else {
      const std::string element_block_id = bc->elementBlockID();

      std::map<std::string,Teuchos::RCP<panzer::PhysicsBlock> >::const_iterator volume_pb_itr
	= physicsBlocks_map.find(element_block_id);

      TEUCHOS_TEST_FOR_EXCEPTION(volume_pb_itr==physicsBlocks_map.end(),std::logic_error,
				 "panzer::FMB::setupBCFieldManagers: Cannot find physics block corresponding to element block \"" << element_block_id << "\"");

      Teuchos::RCP<const panzer::PhysicsBlock> volume_pb = physicsBlocks_map.find(element_block_id)->second;
      Teuchos::RCP<const shards::CellTopology> volume_cell_topology = volume_pb->cellData().getCellTopology();

      // Build one FieldManager for each local side workset for each dirichlet bc
      std::map<unsigned,PHX::FieldManager<panzer::Traits> >& field_managers =
        bc_field_managers_[*bc];

      // Loop over local face indices and setup each field manager
      for (std::map<unsigned,panzer::Workset>::const_iterator wkst =
	     currentWkst->begin(); wkst != currentWkst->end();
	   ++wkst) {

        PHX::FieldManager<panzer::Traits>& fm = field_managers[wkst->first];

        // register evaluators from strategy
        const panzer::CellData side_cell_data(wkst->second.num_cells,
	                                      wkst->first,volume_cell_topology);

	// Copy the physics block for side integrations
	Teuchos::RCP<panzer::PhysicsBlock> side_pb = volume_pb->copyWithCellData(side_cell_data);

	Teuchos::RCP<panzer::BCStrategy_TemplateManager<panzer::Traits> > bcstm =
	  bc_factory.buildBCStrategy(*bc,side_pb->globalData());

	// Iterate over evaluation types
        int i=0;
	for (panzer::BCStrategy_TemplateManager<panzer::Traits>::iterator
	       bcs_type = bcstm->begin(); bcs_type != bcstm->end(); ++bcs_type,++i) {
          if (active_evaluation_types_[i]) {
            bcs_type->setup(*side_pb,user_data);
            bcs_type->buildAndRegisterEvaluators(fm,*side_pb,cm_factory,closure_models,user_data);
            bcs_type->buildAndRegisterGatherAndOrientationEvaluators(fm,*side_pb,lo_factory,user_data);
            if(!physicsBlockScatterDisabled())
              bcs_type->buildAndRegisterScatterEvaluators(fm,*side_pb,lo_factory,user_data);
          }
	}

	// Setup the fieldmanager
	Traits::SD setupData;
	Teuchos::RCP<std::vector<panzer::Workset> > worksets =
	  Teuchos::rcp(new(std::vector<panzer::Workset>));
	worksets->push_back(wkst->second);
	setupData.worksets_ = worksets;
        setupData.orientations_ = getWorksetContainer()->getOrientations();

	// setup derivative information
	setKokkosExtendedDataTypeDimensions(element_block_id,*globalIndexer,user_data,fm);

        Sacado::mpl::for_each_no_kokkos<panzer::Traits::EvalTypes>(PostRegistrationFunctor(active_evaluation_types_,fm,setupData));
      }
    }
  }
}


//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::
setupDiricheltFieldManagers(const Teuchos::ParameterList& pl, const Teuchos::RCP<const TianXin::AbstractDiscretation>& mesh,
      const Teuchos::RCP<const panzer::GlobalIndexer> & indexer )
{
	if( !phx_dirichlet_field_manager_ )
		 phx_dirichlet_field_manager_ = std::shared_ptr<PHX::FieldManager<panzer::Traits>>( new PHX::FieldManager<panzer::Traits>());

    for (Teuchos::ParameterList::ConstIterator bc_pl=pl.begin(); bc_pl != pl.end(); ++bc_pl) {
		TEUCHOS_TEST_FOR_EXCEPTION( !(bc_pl->second.isList()), std::logic_error,
				"Error - All objects in the Dirichlet Conditions sublist must be sublists!" );
		Teuchos::ParameterList& sublist = Teuchos::getValue<Teuchos::ParameterList>(bc_pl->second);

		Teuchos::RCP< TianXin::DirichletEvalautor<panzer::Traits::Residual, panzer::Traits> > re =
			Teuchos::rcp( new TianXin::DirichletEvalautor<panzer::Traits::Residual, panzer::Traits>(sublist, mesh, indexer) );
		phx_dirichlet_field_manager_->template registerEvaluator<panzer::Traits::Residual>(re);
		phx_dirichlet_field_manager_->requireField<panzer::Traits::Residual>(*re->evaluatedFields()[0]);

		Teuchos::RCP< TianXin::DirichletEvalautor<panzer::Traits::Jacobian, panzer::Traits> > je =
			Teuchos::rcp( new TianXin::DirichletEvalautor<panzer::Traits::Jacobian, panzer::Traits>(sublist, mesh, indexer) );
		phx_dirichlet_field_manager_->registerEvaluator<panzer::Traits::Jacobian>(je);
		phx_dirichlet_field_manager_->requireField<panzer::Traits::Jacobian>(*je->evaluatedFields()[0]);
	}

	panzer::Traits::SD setupData;

	std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(1);
    phx_dirichlet_field_manager_->setKokkosExtendedDataTypeDimensions<panzer::Traits::Jacobian>(derivative_dimensions);
    //phx_dirichlet_field_manager_->setKokkosExtendedDataTypeDimensions<panzer::Traits::Tangent>(derivative_dimensions);
    phx_dirichlet_field_manager_->postRegistrationSetup(setupData);
}

//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::
setupNeumannFieldManagers(const Teuchos::ParameterList& pl, const Teuchos::RCP<const TianXin::AbstractDiscretation>& mesh,
      const std::vector<Teuchos::RCP<panzer::PhysicsBlock> >& physicsBlocks,
      const panzer::LinearObjFactory<panzer::Traits> & lo_factory,
      const Teuchos::ParameterList& user_data)
{
	TEUCHOS_TEST_FOR_EXCEPTION(getWorksetContainer()==Teuchos::null,std::logic_error,
                            "panzer::FMB::setupBCFieldManagers: method function getWorksetContainer() returns null. "
                            "Plase call setWorksetContainer() before calling this method");

    Teuchos::RCP<const panzer::GlobalIndexer> globalIndexer = lo_factory.getRangeGlobalIndexer();

    // for convenience build a map (element block id => physics block)
    std::map<std::string,Teuchos::RCP<panzer::PhysicsBlock> > physicsBlocks_map;
    {
       std::vector<Teuchos::RCP<panzer::PhysicsBlock> >::const_iterator blkItr;
       for(blkItr=physicsBlocks.begin();blkItr!=physicsBlocks.end();++blkItr) {
          Teuchos::RCP<panzer::PhysicsBlock> pb = *blkItr;
          std::string blockId = pb->elementBlockID();

          // add block id, physics block pair to the map
          physicsBlocks_map.insert(std::make_pair(blockId,pb));
       }
    }

    // ***************************
    // Neumans
    // ***************************
	for (Teuchos::ParameterList::ConstIterator bc_pl=pl.begin(); bc_pl != pl.end(); ++bc_pl) {
		TEUCHOS_TEST_FOR_EXCEPTION( !(bc_pl->second.isList()), std::logic_error,
				"Error - All objects in the Neumann Conditions sublist must be sublists!" );
		Teuchos::ParameterList& sublist = Teuchos::getValue<Teuchos::ParameterList>(bc_pl->second);
		
		std::shared_ptr<PHX::FieldManager<panzer::Traits> > fm
          = std::shared_ptr<PHX::FieldManager<panzer::Traits>>( new PHX::FieldManager<panzer::Traits>());
		
		WorksetDescriptor wd(sublist);
		const Teuchos::RCP<panzer::Workset> currentWkst = getWorksetContainer()->getSideWorkset(wd);
		if (currentWkst.is_null()) continue;
		
		const std::string element_block_id = wd.getElementBlock();
		const auto& volume_pb_itr = physicsBlocks_map.find(element_block_id);
        TEUCHOS_TEST_FOR_EXCEPTION(volume_pb_itr==physicsBlocks_map.end(),std::logic_error,
				 "panzer::FMB::setupBCFieldManagers: Cannot find physics block corresponding to element block \"" << element_block_id << "\"");
		
		Teuchos::RCP<const panzer::PhysicsBlock> volume_pb = physicsBlocks_map.find(element_block_id)->second;
        Teuchos::RCP<const shards::CellTopology> volume_cell_topology = volume_pb->cellData().getCellTopology();
		const panzer::CellData side_cell_data(currentWkst->num_cells,currentWkst->subcell_index,volume_cell_topology);
		//auto nd = side_cell_data.getCellTopology()->getNodeCount();

	    // Copy the physics block for side integrations
	    Teuchos::RCP<panzer::PhysicsBlock> side_pb = volume_pb->copyWithCellData(side_cell_data);
		Teuchos::ParameterList plist(sublist);
		const std::string dof_name= sublist.get<std::string>("DOF Name");
		
		// ---- Define bais and ir -------
		//RCP<const panzer::FieldLibrary> fieldLib = physicsBlock.getFieldLibrary();
  //RCP<const panzer::PureBasis> basis = fieldLib->lookupBasis("TEMPERATURE");
		Teuchos::RCP<const panzer::PureBasis> basis = side_pb->getBasisForDOF(dof_name);
		plist.set<Teuchos::RCP<const panzer::PureBasis>>("Basis", basis);
		const std::map<int,Teuchos::RCP< panzer::IntegrationRule > >& map_ir = side_pb->getIntegrationRules();
        TEUCHOS_ASSERT(map_ir.size() == 1); 
		Teuchos::RCP<panzer::IntegrationRule> ir = map_ir.begin()->second;
		plist.set<Teuchos::RCP<const panzer::IntegrationRule>>("IR", ir.getConst());
        //const int integration_order = ir.begin()->second->order();
		//const int integration_order = side_pb->getIntegrationOrder();
		//Teuchos::RCP<panzer::IntegrationRule> ir = Teuchos::rcp(new panzer::IntegrationRule(integration_order,side_cell_data));
		
		// ====== Residual evaluator ========
		const std::string Identifier= sublist.get<std::string>("Type");
		std::unique_ptr<TianXin::NeumannBase<panzer::Traits::Residual, panzer::Traits>> evalr = 
			TianXin::NeumannResidualFactory::Instance().Create(Identifier, plist);
		auto sr = evalr->buildScatterEvaluator(plist,lo_factory);
		const std::string& scatterNamer = evalr->getScatterFieldName();
		Teuchos::RCP<PHX::Evaluator<panzer::Traits> > re = Teuchos::rcp(evalr.release());
		fm->template registerEvaluator<panzer::Traits::Residual>(re);
		fm->requireField<panzer::Traits::Residual>(*re->evaluatedFields()[0]);
		fm->template registerEvaluator<panzer::Traits::Residual>(sr);
		{
			PHX::Tag<typename panzer::Traits::Residual::ScalarT> tagr(scatterNamer,
					      Teuchos::rcp(new PHX::MDALayout<panzer::Dummy>(0)));
			fm->template requireField<panzer::Traits::Residual>(tagr);
		}

		// ====== Jacobian evaluator =======
		std::unique_ptr<TianXin::NeumannBase<panzer::Traits::Jacobian, panzer::Traits>> evalj = 
			TianXin::NeumannJacobianFactory::Instance().Create(Identifier, plist);
		auto sj = evalj->buildScatterEvaluator(plist,lo_factory);
		const std::string& scatterNamej = evalj->getScatterFieldName();
		Teuchos::RCP<PHX::Evaluator<panzer::Traits> > rj = Teuchos::rcp(evalj.release());
		fm->template registerEvaluator<panzer::Traits::Jacobian>(rj);
		fm->requireField<panzer::Traits::Jacobian>(*rj->evaluatedFields()[0]);
		fm->template registerEvaluator<panzer::Traits::Jacobian>(sj);
		{
			PHX::Tag<typename panzer::Traits::Jacobian::ScalarT> tagj(scatterNamej,
					      Teuchos::rcp(new PHX::MDALayout<panzer::Dummy>(0)));
			fm->template requireField<panzer::Traits::Jacobian>(tagj);
		}

		// gather
		side_pb->buildAndRegisterGatherAndOrientationEvaluators(*fm,lo_factory,user_data);
		
		Traits::SD setupData;
	    Teuchos::RCP<std::vector<panzer::Workset> > worksets = Teuchos::rcp(new(std::vector<panzer::Workset>));
	    worksets->push_back(*currentWkst);
	    setupData.worksets_ = worksets;
        setupData.orientations_ = getWorksetContainer()->getOrientations();

	   // For Kokkos extended types (Sacado FAD) set derivtive array size
	    //std::vector<PHX::index_size_type> derivative_dimensions;
        //derivative_dimensions.push_back(basis->cardinality());   
	    //fm->setKokkosExtendedDataTypeDimensions<panzer::Traits::Jacobian>(derivative_dimensions);
		setKokkosExtendedDataTypeDimensions(element_block_id,*globalIndexer,user_data,*fm);
		fm->postRegistrationSetup(setupData);
		
		neumann_workset_desc_.push_back(wd);
		phx_neumann_field_manager_.push_back(fm);
	}
}

//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::
setupResponseFieldManagers(const Teuchos::ParameterList& pl, 
      const Teuchos::RCP<const TianXin::AbstractDiscretation>& mesh,
      const std::vector<Teuchos::RCP<panzer::PhysicsBlock> >& physicsBlocks,
      const panzer::LinearObjFactory<panzer::Traits> & lo_factory,
	  const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>& cm_factory,
      const Teuchos::ParameterList& closure_models,
      const Teuchos::ParameterList& user_data,
	  std::unordered_map<std::string, std::vector<TianXin::TemplatedResponse>>& respContainer)
{
	TEUCHOS_TEST_FOR_EXCEPTION(getWorksetContainer2()==Teuchos::null,std::logic_error,
                            "panzer::FMB::setupResponseFieldManagers: method function getWorksetContainer2() returns null. "
                            "Plase call setWorksetContainer() before calling this method");

    Teuchos::RCP<const panzer::GlobalIndexer> globalIndexer = lo_factory.getRangeGlobalIndexer();

    // for convenience build a map (element block id => physics block)
    std::map<std::string,Teuchos::RCP<panzer::PhysicsBlock> > physicsBlocks_map;
    {
       std::vector<Teuchos::RCP<panzer::PhysicsBlock> >::const_iterator blkItr;
       for(blkItr=physicsBlocks.begin();blkItr!=physicsBlocks.end();++blkItr) {
          Teuchos::RCP<panzer::PhysicsBlock> pb = *blkItr;
          std::string blockId = pb->elementBlockID();

          // add block id, physics block pair to the map
          physicsBlocks_map.insert(std::make_pair(blockId,pb));
       }
    }

    // ***************************
    // Neumans
    // ***************************
	for (Teuchos::ParameterList::ConstIterator bc_pl=pl.begin(); bc_pl != pl.end(); ++bc_pl) {
		TEUCHOS_TEST_FOR_EXCEPTION( !(bc_pl->second.isList()), std::logic_error,
				"Error - All objects in the Sideset Response Conditions sublist must be sublists!" );
		Teuchos::ParameterList& sublist = Teuchos::getValue<Teuchos::ParameterList>(bc_pl->second);
		
		Teuchos::Array<std::string> eblocks = sublist.get<Teuchos::Array<std::string>>("Element Block Name");
		Teuchos::Array<std::string> esides = sublist.get<Teuchos::Array<std::string>>("SideSet Name",Teuchos::tuple<std::string>(""));
		bool notSideset = esides[0].empty();
		if( !notSideset ) {
			if( eblocks.size() != esides.size() ) 
				TEUCHOS_TEST_FOR_EXCEPTION( (eblocks.size() != esides.size()), std::logic_error,
				"Error - Cannot define eblock-side pair!" );
		}
	
		std::vector<TianXin::TemplatedResponse> resps;
		std::string respname;
		for( unsigned int i=0; i<eblocks.size(); ++i ) {
			const auto& volume_pb_itr = physicsBlocks_map.find(eblocks[i]);
			TEUCHOS_TEST_FOR_EXCEPTION(volume_pb_itr==physicsBlocks_map.end(),std::logic_error,
				 "panzer::FMB::setupBCFieldManagers: Cannot find physics block corresponding to element block \"" << eblocks[i] << "\"");
			Teuchos::RCP<panzer::PhysicsBlock> volume_pb = physicsBlocks_map.find(eblocks[i])->second;
			
			std::shared_ptr<PHX::FieldManager<panzer::Traits> > fm
				= std::shared_ptr<PHX::FieldManager<panzer::Traits>>( new PHX::FieldManager<panzer::Traits>());

			Teuchos::RCP<panzer::Workset> currentWkst;
			Teuchos::RCP<panzer::PhysicsBlock> side_pb;
			if( notSideset ) {
				WorksetDescriptor wd(eblocks[i],WorksetSizeType::ALL_ELEMENTS);
				Teuchos::RCP<std::vector<Workset> > wksts = getWorksetContainer2()->getWorksets(wd);
				if (wksts.is_null()) continue;
				response_workset_desc_.push_back(wd);
				currentWkst = Teuchos::rcpFromRef((*wksts)[0]);
				side_pb = volume_pb;
			} else {
				WorksetDescriptor wd(eblocks[i],esides[i]);
				currentWkst = getWorksetContainer2()->getSideWorkset(wd);
				if (currentWkst.is_null()) continue;
				response_workset_desc_.push_back(wd);
				Teuchos::RCP<const shards::CellTopology> volume_cell_topology = volume_pb->cellData().getCellTopology();
				const panzer::CellData side_cell_data(currentWkst->num_cells,currentWkst->subcell_index,volume_cell_topology);
				// Copy the physics block for side integrations
				side_pb = volume_pb->copyWithCellData(side_cell_data);
			}

			side_pb->buildAndRegisterEquationSetEvaluators(*fm, user_data);
			side_pb->buildAndRegisterClosureModelEvaluatorsForType<panzer::Traits::Residual>(*fm,cm_factory,closure_models,user_data);
			//side_pb->buildAndRegisterClosureModelEvaluatorsForType<panzer::Traits::Tangent>(*fm,cm_factory,closure_models,user_data);

			// ---- Define bais and ir -------
			Teuchos::ParameterList plist(sublist);
			const std::string dof_name= sublist.get<std::string>("DOF Name");
		//RCP<const panzer::FieldLibrary> fieldLib = physicsBlock.getFieldLibrary();
  //RCP<const panzer::PureBasis> basis = fieldLib->lookupBasis("TEMPERATURE");
			Teuchos::RCP<const panzer::PureBasis> basis = side_pb->getBasisForDOF(dof_name);
			plist.set<Teuchos::RCP<const panzer::PureBasis>>("Basis", basis);
			const std::map<int,Teuchos::RCP< panzer::IntegrationRule > >& map_ir = side_pb->getIntegrationRules();
			TEUCHOS_ASSERT(map_ir.size() == 1); 
			Teuchos::RCP<panzer::IntegrationRule> ir = map_ir.begin()->second;
			plist.set<Teuchos::RCP<const panzer::IntegrationRule>>("IR", ir.getConst());
        //const int integration_order = ir.begin()->second->order();
		//const int integration_order = side_pb->getIntegrationOrder();
		//Teuchos::RCP<panzer::IntegrationRule> ir = Teuchos::rcp(new panzer::IntegrationRule(integration_order,side_cell_data));
	
			// ====== Residual evaluator ========
			const std::string Identifier= sublist.get<std::string>("Type");
			std::unique_ptr<TianXin::ResponseBase<panzer::Traits::Residual, panzer::Traits>> evalr = 
				TianXin::ResponseResidualFactory::Instance().Create(Identifier, plist);
			TEUCHOS_TEST_FOR_EXCEPTION(!evalr,std::logic_error,
                            "panzer::FMB::setupResponseFieldManagers: Create ResponseBase returns null. ");
            if( evalr->isResiudal() ) {
				auto sr = evalr->buildScatterEvaluator(plist,lo_factory);
				const std::string& scatterNamer = evalr->getScatterFieldName();
				fm->template registerEvaluator<panzer::Traits::Residual>(sr);
				PHX::Tag<typename panzer::Traits::Residual::ScalarT> tagr(scatterNamer,
					      Teuchos::rcp(new PHX::MDALayout<panzer::Dummy>(0)));
				fm->template requireField<panzer::Traits::Residual>(tagr);
			}
			respname = evalr->getResponseName();
			Teuchos::RCP<TianXin::ResponseBase<panzer::Traits::Residual, panzer::Traits> > re = Teuchos::rcp(evalr.release());
			fm->template registerEvaluator<panzer::Traits::Residual>(re);
			fm->requireField<panzer::Traits::Residual>(*re->evaluatedFields()[0]);

		// ====== Tangent evaluator =======
		/*std::unique_ptr<TianXin::ResponseBase<panzer::Traits::Tangent, panzer::Traits>> evalj = 
			TianXin::ResponseTangentFactory::Instance().Create(Identifier, plist);
		auto sj = evalj->buildScatterEvaluator(plist,lo_factory);
		const std::string& scatterNamej = evalj->getScatterFieldName();
		Teuchos::RCP<PHX::Evaluator<panzer::Traits> > rj = Teuchos::rcp(evalj.release());
		fm->template registerEvaluator<panzer::Traits::Tangent>(rj);
		fm->requireField<panzer::Traits::Tangent>(*rj->evaluatedFields()[0]);
		fm->template registerEvaluator<panzer::Traits::Tangent>(sj);
		{
			PHX::Tag<typename panzer::Traits::Tangent::ScalarT> tagj(scatterNamej,
					      Teuchos::rcp(new PHX::MDALayout<panzer::Dummy>(0)));
			fm->template requireField<panzer::Traits::Tangent>(tagj);
		}*/

			// ==== Save in container =====
			TianXin::TemplatedResponse aresp;
			aresp.set<panzer::Traits::Residual>( re );
			resps.emplace_back(aresp);

			// gather
			side_pb->buildAndRegisterGatherAndOrientationEvaluators(*fm,lo_factory,user_data);

			Traits::SD setupData;
			Teuchos::RCP<std::vector<panzer::Workset> > worksets = Teuchos::rcp(new(std::vector<panzer::Workset>));
			worksets->push_back(*currentWkst);
			setupData.worksets_ = worksets;
			//setupData.orientations_ = getWorksetContainer2()->getOrientations();
	   // For Kokkos extended types (Sacado FAD) set derivtive array size
	    //std::vector<PHX::index_size_type> derivative_dimensions;
        //derivative_dimensions.push_back(basis->cardinality());   
	    //fm->setKokkosExtendedDataTypeDimensions<panzer::Traits::Jacobian>(derivative_dimensions);
			setKokkosExtendedDataTypeDimensions(eblocks[i],*globalIndexer,user_data,*fm);
//std::cout << lo_factory.getComm().getRank() << ", " << eblocks[i] << ",  " << esides[i] << "," << currentWkst->num_cells << std::endl;
			fm->postRegistrationSetup(setupData);
		
			phx_response_field_manager_.push_back(fm);
		}
		respContainer.emplace( respname, resps );
	};
}

//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::
writeVolumeGraphvizDependencyFiles(std::string filename_prefix,
				   const std::vector<Teuchos::RCP<panzer::PhysicsBlock> >& physicsBlocks) const
{
  if(phx_volume_field_managers_.empty())
    return; // nothing to see here folks

  TEUCHOS_ASSERT(phx_volume_field_managers_.size()==physicsBlocks.size());

  std::vector<Teuchos::RCP<panzer::PhysicsBlock> >::const_iterator blkItr;
  int index = 0;
  for (blkItr=physicsBlocks.begin();blkItr!=physicsBlocks.end();++blkItr,++index) {
    std::string blockId = (*blkItr)->elementBlockID();
    phx_volume_field_managers_[index]->writeGraphvizFile(filename_prefix+"_VOLUME_"+blockId);
  }

}

//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::
writeBCGraphvizDependencyFiles(std::string filename_prefix) const
{
  typedef std::map<panzer::BC,std::map<unsigned,PHX::FieldManager<panzer::Traits> >,panzer::LessBC> FMMap;

  FMMap::const_iterator blkItr;
  int bc_index = 0;
  for (blkItr=bc_field_managers_.begin();blkItr!=bc_field_managers_.end();++blkItr,++bc_index) {
    panzer::BC bc = blkItr->first;
    const PHX::FieldManager<panzer::Traits> & fm = blkItr->second.begin()->second; // get the first field manager

    BCType bc_type = bc.bcType();
    std::string type;
    if (bc_type == BCT_Dirichlet)
	type = "_Dirichlet_";
    else if (bc_type == BCT_Neumann)
        type = "_Neumann_";
    else if (bc_type == BCT_Interface)
        type = "_Interface_";
    else
        TEUCHOS_ASSERT(false);

    std::string blockId = bc.elementBlockID();
    std::string sideId = bc.sidesetID();
    fm.writeGraphvizFile(filename_prefix+"_BC_"+std::to_string(bc_index)+type+sideId+"_"+blockId);
  }

}

//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::
writeNeumannGraphvizDependencyFiles(std::string filename_prefix) const
{
  if(phx_neumann_field_manager_.empty()) return;

  int bc_index = 0;
  for( const auto& fm : phx_neumann_field_manager_ ) {
    fm->writeGraphvizFile(filename_prefix+"_Neumann_"+std::to_string(bc_index));
  }

}

//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::
writeVolumeTextDependencyFiles(std::string filename_prefix,
			       const std::vector<Teuchos::RCP<panzer::PhysicsBlock> >& physicsBlocks) const
{
  if(phx_volume_field_managers_.size()<1)
    return; // nothing to see here folks

  TEUCHOS_ASSERT(phx_volume_field_managers_.size()==physicsBlocks.size());

  std::vector<Teuchos::RCP<panzer::PhysicsBlock> >::const_iterator blkItr;
  int index = 0;
  for (blkItr=physicsBlocks.begin();blkItr!=physicsBlocks.end();++blkItr,++index) {

    std::string blockId = (*blkItr)->elementBlockID();

    std::string filename = filename_prefix+"_VOLUME_"+blockId+".txt";
    std::ofstream ofs;
    ofs.open(filename.c_str());

    ofs << *(phx_volume_field_managers_[index]) << std::endl;

    ofs.close();
  }

}

//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::
writeBCTextDependencyFiles(std::string filename_prefix) const
{
  typedef std::map<panzer::BC,std::map<unsigned,PHX::FieldManager<panzer::Traits> >,panzer::LessBC> FMMap;

  FMMap::const_iterator blkItr;
  int bc_index = 0;
  for (blkItr=bc_field_managers_.begin();blkItr!=bc_field_managers_.end();++blkItr,++bc_index) {
    panzer::BC bc = blkItr->first;
    const PHX::FieldManager<panzer::Traits> & fm = blkItr->second.begin()->second; // get the first field manager

    BCType bc_type = bc.bcType();
    std::string type;
    if (bc_type == BCT_Dirichlet)
	type = "_Dirichlet_";
    else if (bc_type == BCT_Neumann)
        type = "_Neumann_";
    else if (bc_type == BCT_Interface)
        type = "_Interface_";
    else
        TEUCHOS_ASSERT(false);

    std::string blockId = bc.elementBlockID();
    std::string sideId = bc.sidesetID();

    std::string filename = filename_prefix+"_BC_"+std::to_string(bc_index)+type+sideId+"_"+blockId+".txt";
    std::ofstream ofs;
    ofs.open(filename.c_str());

    ofs << fm << std::endl;

    ofs.close();
  }

}

//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::
writeNeumannTextDependencyFiles(std::string filename_prefix) const
{
  if(phx_neumann_field_manager_.empty()) return;

  int bc_index = 0;
  for( const auto& fm : phx_neumann_field_manager_ ) {
    std::string filename = filename_prefix+"_Neumann_"+std::to_string(++bc_index)+".txt";
    std::ofstream ofs;
    ofs.open(filename.c_str());
    ofs << *fm << std::endl;
    ofs.close();
  }
}

//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::
writeResponseTextDependencyFiles(std::string filename_prefix) const
{
  if(phx_response_field_manager_.empty()) return;

  int bc_index = 0;
  for( const auto& fm : phx_response_field_manager_ ) {
    std::string filename = filename_prefix+"_Response_"+std::to_string(++bc_index)+".txt";
    std::ofstream ofs;
    ofs.open(filename.c_str());
    ofs << *fm << std::endl;
    ofs.close();
  }
}

//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::
setKokkosExtendedDataTypeDimensions(const std::string & eblock,
                                    const panzer::GlobalIndexer & globalIndexer,
                                    const Teuchos::ParameterList& user_data,
                                    PHX::FieldManager<panzer::Traits> & fm) const
{
  // setup Jacobian derivative terms
  {
    std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(globalIndexer.getElementBlockGIDCount(eblock));

    fm.setKokkosExtendedDataTypeDimensions<panzer::Traits::Jacobian>(derivative_dimensions);

  }

  #ifdef Panzer_BUILD_HESSIAN_SUPPORT
  {
    std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(globalIndexer.getElementBlockGIDCount(eblock));

    fm.setKokkosExtendedDataTypeDimensions<panzer::Traits::Hessian>(derivative_dimensions);
  }
  #endif

  {
    std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(1);
    if (user_data.isType<int>("Tangent Dimension"))
      derivative_dimensions[0] = user_data.get<int>("Tangent Dimension");
    fm.setKokkosExtendedDataTypeDimensions<panzer::Traits::Tangent>(derivative_dimensions);
  }
}

void panzer::FieldManagerBuilder::setActiveEvaluationTypes(const std::vector<bool>& aet)
{active_evaluation_types_ = aet;}

//=======================================================================
//=======================================================================
void panzer::FieldManagerBuilder::clearVolumeFieldManagers(bool clearVolumeWorksets)
{
  phx_volume_field_managers_.clear();
  volume_workset_desc_.clear();
  if (clearVolumeWorksets)
    worksetContainer_->clearVolumeWorksets();
}

//=======================================================================
//=======================================================================
std::ostream& panzer::operator<<(std::ostream& os, const panzer::FieldManagerBuilder& rfd)
{
  rfd.print(os);
  return os;
}
