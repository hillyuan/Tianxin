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

#ifndef _TIANXIN_NEUMANN_IMPL_HPP
#define _TIANXIN_NEUMANN_IMPL_HPP

#include <cstddef>
#include <string>
#include <vector>
#include "Panzer_BasisIRLayout.hpp"
#include "Panzer_Workset_Utilities.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Teuchos_RCP.hpp"

namespace TianXin {

//**********************************************************************
template<typename EvalT, typename Traits>
NeumannBase<EvalT, Traits>::
NeumannBase( const Teuchos::ParameterList& p)
{
  std::string residual_name = p.get<std::string>("Residual Name");
  const Teuchos::RCP<const panzer::PureBasis> basis =
    p.get< Teuchos::RCP<const panzer::PureBasis> >("Basis");
  const Teuchos::RCP<const panzer::IntegrationRule> ir = 
    p.get< Teuchos::RCP<const panzer::IntegrationRule> >("IR");

  residual = PHX::MDField<ScalarT>(residual_name, basis->functional);
  this->addEvaluatedField(residual);
 
  basis_name = panzer::basisIRLayout(basis,*ir)->name();
}

//**********************************************************************
template<typename EvalT, typename Traits>
void
NeumannBase<EvalT, Traits>::
postRegistrationSetup( typename Traits::SetupData sd,
  PHX::FieldManager<Traits>& /* fm */)
{
  basis_index = panzer::getBasisIndex(basis_name, (*sd.worksets_)[0], this->wda);
}

// **************************************************************
// Flux
// **************************************************************
template<typename EvalT, typename Traits>
Flux<EvalT, Traits>::Flux( const Teuchos::ParameterList& p)
: NeumannBase<EvalT, Traits>(p)
{
  std::string n = "Neumann Flux Evaluator";
  this->setName(n);
  
  auto& value_type = p.get<std::string>("Value Type","Constant");
  this->pFunc = GeneralFunctorFactory::Instance().Create(value_type, p);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void
Flux<EvalT, Traits>::evaluateFields(typename Traits::EvalData d)
{
  //basis_index = panzer::getBasisIndex(basis_name, (*sd.worksets_)[0], this->wda);
  // Grab the basis information.
  //  basis_ = this->wda(workset).bases[basisIndex_]->weighted_basis_scalar;
}


}

#endif
