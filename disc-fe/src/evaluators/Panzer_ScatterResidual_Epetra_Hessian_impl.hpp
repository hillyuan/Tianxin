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

#ifndef __Panzer_ScatterResidual_Epetra_Hessian_impl_hpp__
#define __Panzer_ScatterResidual_Epetra_Hessian_impl_hpp__

// only do this if required by the user
#ifdef Panzer_BUILD_HESSIAN_SUPPORT

#include "Teuchos_RCP.hpp"
#include "Teuchos_Assert.hpp"

#include "Phalanx_DataLayout.hpp"

#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"

#include "Panzer_GlobalIndexer.hpp"
#include "Panzer_PureBasis.hpp"
#include "Panzer_EpetraLinearObjContainer.hpp"
#include "Panzer_LOCPair_GlobalEvaluationData.hpp"
#include "Panzer_ParameterList_GlobalEvaluationData.hpp"

#include "Phalanx_DataLayout_MDALayout.hpp"

#include "Teuchos_FancyOStream.hpp"

// **********************************************************************
// Specialization: Hessian
// **********************************************************************

template<typename TRAITS,typename LO,typename GO>
panzer::ScatterResidual_Epetra<panzer::Traits::Hessian, TRAITS,LO,GO>::
ScatterResidual_Epetra(const Teuchos::RCP<const GlobalIndexer> & indexer,
                       const Teuchos::RCP<const panzer::GlobalIndexer> & cIndexer,
                       const Teuchos::ParameterList& p,
                       bool useDiscreteAdjoint)
   : globalIndexer_(indexer)
   , colGlobalIndexer_(cIndexer) 
   , globalDataKey_("Residual Scatter Container")
   , useDiscreteAdjoint_(useDiscreteAdjoint)
{ 
  std::string scatterName = p.get<std::string>("Scatter Name");
  scatterHolder_ = 
    Teuchos::rcp(new PHX::Tag<ScalarT>(scatterName,Teuchos::rcp(new PHX::MDALayout<Dummy>(0))));

  // get names to be evaluated
  const std::vector<std::string>& names = 
    *(p.get< Teuchos::RCP< std::vector<std::string> > >("Dependent Names"));

  // grab map from evaluated names to field names
  fieldMap_ = p.get< Teuchos::RCP< std::map<std::string,std::string> > >("Dependent Map");

  Teuchos::RCP<PHX::DataLayout> dl = 
    p.get< Teuchos::RCP<const panzer::PureBasis> >("Basis")->functional;
  
  // build the vector of fields that this is dependent on
  scatterFields_.resize(names.size());
  for (std::size_t eq = 0; eq < names.size(); ++eq) {
    scatterFields_[eq] = PHX::MDField<const ScalarT,Cell,NODE>(names[eq],dl);

    // tell the field manager that we depend on this field
    this->addDependentField(scatterFields_[eq]);
  }

  // this is what this evaluator provides
  this->addEvaluatedField(*scatterHolder_);

  if (p.isType<std::string>("Global Data Key"))
     globalDataKey_ = p.get<std::string>("Global Data Key");
  if (p.isType<bool>("Use Discrete Adjoint"))
     useDiscreteAdjoint = p.get<bool>("Use Discrete Adjoint");

  // discrete adjoint does not work with non-square matrices
  if(useDiscreteAdjoint)
  { TEUCHOS_ASSERT(colGlobalIndexer_==globalIndexer_); }

  this->setName(scatterName+" Scatter Residual Epetra (Jacobian)");
}

// **********************************************************************
template<typename TRAITS,typename LO,typename GO> 
void panzer::ScatterResidual_Epetra<panzer::Traits::Hessian, TRAITS,LO,GO>::
postRegistrationSetup(typename TRAITS::SetupData /* d */,
                      PHX::FieldManager<TRAITS>& /* fm */)
{
  fieldIds_.resize(scatterFields_.size());
  // load required field numbers for fast use
  for(std::size_t fd=0;fd<scatterFields_.size();++fd) {
    // get field ID from DOF manager
    std::string fieldName = fieldMap_->find(scatterFields_[fd].fieldTag().name())->second;
    fieldIds_[fd] = globalIndexer_->getFieldNum(fieldName);
  }
}

// **********************************************************************
template<typename TRAITS,typename LO,typename GO>
void panzer::ScatterResidual_Epetra<panzer::Traits::Hessian, TRAITS,LO,GO>::
preEvaluate(typename TRAITS::PreEvalData d)
{
  // extract linear object container
  epetraContainer_ = Teuchos::rcp_dynamic_cast<EpetraLinearObjContainer>(d.gedc->getDataObject(globalDataKey_));
 
  if(epetraContainer_==Teuchos::null) {
    // extract linear object container
    Teuchos::RCP<LinearObjContainer> loc = Teuchos::rcp_dynamic_cast<LOCPair_GlobalEvaluationData>(d.gedc->getDataObject(globalDataKey_),true)->getGhostedLOC();
    epetraContainer_ = Teuchos::rcp_dynamic_cast<EpetraLinearObjContainer>(loc);
  }
}

// **********************************************************************
template<typename TRAITS,typename LO,typename GO>
void panzer::ScatterResidual_Epetra<panzer::Traits::Hessian, TRAITS,LO,GO>::
evaluateFields(typename TRAITS::EvalData workset)
{
  using PHX::View;
  using PHX::Device;
  using std::vector;

   std::vector<double> jacRow;

   bool useColumnIndexer = colGlobalIndexer_!=Teuchos::null;

   // for convenience pull out some objects from workset
   std::string blockId = this->wda(workset).block_id;
   const std::vector<std::size_t> & localCellIds = this->wda(workset).cell_local_ids;

   Teuchos::RCP<Epetra_Vector> r = epetraContainer_->get_f(); 
   Teuchos::RCP<Epetra_CrsMatrix> Jac = epetraContainer_->get_A();

   const Teuchos::RCP<const panzer::GlobalIndexer>&
     colGlobalIndexer = useColumnIndexer ? colGlobalIndexer_ : globalIndexer_;
   
   // NOTE: A reordering of these loops will likely improve performance
   //       The "getGIDFieldOffsets" may be expensive.  However the
   //       "getElementGIDs" can be cheaper. However the lookup for LIDs
   //       may be more expensive!

   // scatter operation for each cell in workset
   for(std::size_t worksetCellIndex=0;worksetCellIndex<localCellIds.size();++worksetCellIndex) {
      std::size_t cellLocalId = localCellIds[worksetCellIndex];

      auto rLIDs = globalIndexer_->getElementLIDs(cellLocalId); 
      auto initial_cLIDs = colGlobalIndexer->getElementLIDs(cellLocalId);
      vector<int> cLIDs;
      for (int i(0); i < static_cast<int>(initial_cLIDs.extent(0)); ++i)
        cLIDs.push_back(initial_cLIDs(i));
      if (Teuchos::nonnull(workset.other)) {
        const std::size_t other_cellLocalId = workset.other->cell_local_ids[worksetCellIndex];
	auto other_cLIDs = colGlobalIndexer->getElementLIDs(other_cellLocalId);
        for (int i(0); i < static_cast<int>(other_cLIDs.extent(0)); ++i)
          cLIDs.push_back(other_cLIDs(i));
      }

      // loop over each field to be scattered
      for(std::size_t fieldIndex = 0; fieldIndex < scatterFields_.size(); fieldIndex++) {
         int fieldNum = fieldIds_[fieldIndex];
         const std::vector<int> & elmtOffset = globalIndexer_->getGIDFieldOffsets(blockId,fieldNum);

         // loop over the basis functions (currently they are nodes)
         for(std::size_t rowBasisNum = 0; rowBasisNum < elmtOffset.size(); rowBasisNum++) {
            const ScalarT scatterField = (scatterFields_[fieldIndex])(worksetCellIndex,rowBasisNum);
            int rowOffset = elmtOffset[rowBasisNum];
            int row = rLIDs[rowOffset];
    
            // loop over the sensitivity indices: all DOFs on a cell
            jacRow.resize(scatterField.size());
            
            for(int sensIndex=0;sensIndex<scatterField.size();++sensIndex)
              jacRow[sensIndex] = scatterField.fastAccessDx(sensIndex).fastAccessDx(0);

            {
               int err = Jac->SumIntoMyValues(
                 row,
                 std::min(cLIDs.size(), static_cast<size_t>(scatterField.size())),
                 jacRow.data(),
                 cLIDs.data() );
               TEUCHOS_ASSERT_EQUALITY(err,0);
            }
         } // end rowBasisNum
      } // end fieldIndex
   }
}

// **********************************************************************

#endif

#endif
