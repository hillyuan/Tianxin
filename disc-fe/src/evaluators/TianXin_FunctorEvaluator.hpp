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

#ifndef _TIANXIN_EVALUATOR_FUNCTOR_HPP
#define _TIANXIN_EVALUATOR_FUNCTOR_HPP

#include "PanzerDiscFE_config.hpp"

#include "Phalanx_Evaluator_Macros.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_RCP.hpp"

#include "Panzer_Evaluator_WithBaseImpl.hpp"
#include "TianXin_Functor.hpp"

namespace TianXin {
	
// **************************************************************
// This class defines functor evaluators act upon integration points of cell
// **************************************************************

template<typename EvalT, typename TRAITS>
class FunctorEvaluator : public panzer::EvaluatorWithBaseImpl<TRAITS>,
    public PHX::EvaluatorDerived<EvalT, TRAITS> 
{
	typedef typename EvalT::ScalarT ScalarT;
  
  public:
    FunctorEvaluator(const std::string parameter_name,
	      std::shared_ptr< TianXin::GeneralFunctor<panzer::Traits::RealType> > pf,
	      const Teuchos::RCP<PHX::DataLayout>& data_layout);
    void postRegistrationSetup(typename panzer::Traits::SetupData d, PHX::FieldManager<panzer::Traits>& fm);
    void evaluateFields(typename TRAITS::EvalData ud);
    
  private:    
    PHX::MDField<ScalarT, panzer::Cell, panzer::Point> target_field0;
	PHX::MDField<ScalarT, panzer::Cell, panzer::Point, panzer::Dim> target_field;
	PHX::MDField<ScalarT, panzer::Cell, panzer::Point, panzer::Dim> state_variables;
    unsigned int nitems;
	std::shared_ptr< TianXin::GeneralFunctor<panzer::Traits::RealType> > pFunc;
};


}

#include "TianXin_FunctorEvaluator_impl.hpp"

#endif
