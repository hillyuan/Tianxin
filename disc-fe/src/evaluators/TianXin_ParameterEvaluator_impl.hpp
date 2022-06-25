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

#ifndef _TIANXIN_EVALUATOR_PARAMETER_IMPL_HPP
#define _TIANXIN_EVALUATOR_PARAMETER_IMPL_HPP

#include "PanzerDiscFE_config.hpp"
#include "Panzer_ScalarParameterEntry.hpp"
#include "Panzer_ParameterLibraryUtilities.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace TianXin {

//**********************************************************************
template<typename EvalT, typename TRAITS>
ParameterEvaluator<EvalT, TRAITS>::
ParameterEvaluator(const std::string parameter_name,
	      std::shared_ptr< TianXin::GeneralFunctor<ScalarT> > pf,
	      const Teuchos::RCP<PHX::DataLayout>& data_layout)
: pFunc(pf)
{
	nitems = pFunc->nitems();
	const auto ncell = data_layout->extent(0);
	const auto npoint = data_layout->extent(1);
    target_field = PHX::MDField<ScalarT, panzer::Cell, panzer::Point, panzer::Dim>(parameter_name, ncell, npoint, nitems);

    this->addEvaluatedField(target_field);

    std::string n = "Parameter Evaluator";
    this->setName(n);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void 
ParameterEvaluator<EvalT, Traits>::
postRegistrationSetup( typename panzer::Traits::SetupData  /* worksets */,
  PHX::FieldManager<panzer::Traits>&  fm)
{
  using namespace PHX;
  this->utils.setFieldData(_fieldValue,fm);

  TEUCHOS_ASSERT(static_cast<std::size_t>(_fieldValue.extent(2)) == _size);

  if( pFunc->isConstant() ) {
    auto param_val = (*pf)();
	auto target_field_v = target_field.get_static_view();
    auto target_field_h = Kokkos::create_mirror_view(target_field_v);
	
    for (int cell=0; cell < workset.num_cells; ++cell) {
		for (std::size_t pt=0; pt<target_field_v.extent(1); ++pt)
			for( unsigned int k=0; k<nitems; ++k )
				target_field_h(cell,pt,k) = param_val[k];
	}
	Kokkos::deep_copy(target_field_v, target_field_h);
  }
}

//**********************************************************************
template<typename EvalT, typename TRAITS>
void ParameterEvaluator<EvalT, TRAITS>::
evaluateFields(typename TRAITS::EvalData workset)
{
	if( pFunc->isConstant() ) return;

  auto param_val = (*pf)();
  auto target_field_v = target_field.get_static_view();
  auto target_field_h = Kokkos::create_mirror_view(target_field_v);
  
  for (int cell=0; cell < workset.num_cells; ++cell) {
    for (std::size_t pt=0; pt<target_field_v.extent(1); ++pt)
		for( unsigned int k=0; k<nitems; ++k )
			target_field_h(cell,pt,k) = param_val[k];
  }
  Kokkos::deep_copy(target_field_v, target_field_h);

}

//**********************************************************************

}

#endif
