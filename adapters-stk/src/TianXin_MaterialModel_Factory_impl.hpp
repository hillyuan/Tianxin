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

#ifndef _TIANXIN_MATERIAL_MODEL_FACTORY_IMPL_HPP
#define _TIANXIN_MATERIAL_MODEL_FACTORY_IMPL_HPP

#include <iostream>
#include <sstream>
#include <typeinfo>
#include "Panzer_IntegrationRule.hpp"
#include "Panzer_BasisIRLayout.hpp"
#include "Panzer_Integrator_Scalar.hpp"

#include "Phalanx_FieldTag_Tag.hpp"

#include "Teuchos_ParameterEntry.hpp"
#include "Teuchos_TypeNameTraits.hpp"

#include "Panzer_Parameter.hpp"
#include "Panzer_GlobalStatistics.hpp"
#include "Panzer_CoordinatesEvaluator.hpp"
#include "Panzer_Constant.hpp"
#include "Panzer_LinearObjFactory.hpp"
#include "Panzer_DOF.hpp"
#include "Panzer_GlobalData.hpp"
#include "TianXin_Functor.hpp"

// ********************************************************************
// ********************************************************************
template<typename EvalT>
Teuchos::RCP< std::vector< Teuchos::RCP<PHX::Evaluator<panzer::Traits> > > > 
TianXin::MaterialModelFactory<EvalT>::
buildMaterialModels(const std::string& model_id,
                                const Teuchos::ParameterList& models,
                                const Teuchos::RCP<panzer::IntegrationRule>& ir,
                                const Teuchos::RCP<panzer::GlobalData>& global_data,
                                PHX::FieldManager<panzer::Traits>& fm) const
{
  std::vector< Teuchos::RCP<PHX::Evaluator<panzer::Traits> > >  evaluators;
  
  for( auto a: c0) {
        if( global_data->functors.find(a)== global_data->functors.end() ) {
            std::cout << "Material Property: " << a << "  NOT FOUND!\n";
            throw std::runtime_error("Material Property not defined");
        }
     //   Teuchos::RCP< PHX::Evaluator<panzer::Traits> > e =
    //        Teuchos::rcp( new TianXin::FunctorEvaluator<EvalT, panzer::Traits>(a, m_gd->functos[a],*ir) );
     //   evaluators.push_back(e);
  }
  return evaluators;
}

#endif
