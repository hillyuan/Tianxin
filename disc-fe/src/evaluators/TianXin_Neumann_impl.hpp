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
#include "Intrepid2_CellTools.hpp"
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
  basis_index = panzer::getBasisIndex(basis_name, (*sd.worksets_)[0]);
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
Flux<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  auto weighted_basis_scalar = workset.bases[this->basis_index]->weighted_basis_scalar.get_static_view();
  auto residual_v = residual.get_static_view();
  auto val = this->pFunc();
  Kokkos::parallel_for("Neuman_nResidual", workset.num_cells, KOKKOS_LAMBDA (const index_t cell) {
    for (std::size_t basis = 0; basis < residual_v.extent(1); ++basis) {
      for (std::size_t qp = 0; qp < l_num_ip; ++qp) {
        residual_v(cell,basis) += val*weighted_basis_scalar(cell,basis,qp);
      }
    }
  });
  
  // if(workset.num_cells>0)
  //   Intrepid2::FunctionSpaceTools<PHX::exec_space>::
  //     integrate<ScalarT>(residual.get_view(),
  //                        val, 
  //                        (this->wda(workset).bases[basis_index])->weighted_basis_scalar.get_view());
}

// **************************************************************
// Pressure
// **************************************************************
template<typename EvalT, typename Traits>
Pressure<EvalT, Traits>::Pressure( const Teuchos::ParameterList& p)
: NeumannBase<EvalT, Traits>(p)
{
  std::string n = "Neumann Pressure Evaluator";
  this->setName(n);
  
  Teuchos::RCP<PHX::DataLayout> vector_dl = ir->dl_vector;
  quad_order = ir->cubature_degree;
  normals = PHX::MDField<ScalarT,Cell,Point,Dim>(name, vector_dl);
  
  auto& value_type = p.get<std::string>("Value Type","Constant");
  this->pFunc = GeneralFunctorFactory::Instance().Create(value_type, p);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void
Normals<EvalT, Traits>::
postRegistrationSetup( typename Traits::SetupData sd, PHX::FieldManager<Traits>& /* fm */)
{
  num_qp  = normals.extent(1);
  num_dim = normals.extent(2);
  
  quad_index =  panzer::getIntegrationRuleIndex(quad_order,(*sd.worksets_)[0], this->wda);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void
Pressure<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
	auto normal_view = normals.get_view();
	Intrepid2::CellTools<PHX::exec_space>::getPhysicalSideNormals(normal_view,
                            workset.int_rules[quad_index]->jac.get_view(),
              side_id, workset.int_rules[quad_index]->int_rule->topology);

  auto weighted_basis_scalar = workset.bases[this->basis_index]->weighted_basis_scalar.get_static_view();
  auto residual_v = residual.get_static_view();
  auto val = this->pFunc();
  Kokkos::parallel_for("Neuman_nResidual", workset.num_cells, KOKKOS_LAMBDA (const index_t cell) {
    for (std::size_t basis = 0; basis < residual_v.extent(1); ++basis) {
      for (std::size_t qp = 0; qp < l_num_ip; ++qp) {
		  for(std::size_t dim =0; dim< num_dim; ++dim ) {
			residual_v(cell,basis,dim) += val*normal_view(cell,qp,dim)*weighted_basis_scalar(cell,basis,qp);
		  }
      }
    }
  });
  
  // if(workset.num_cells>0)
  //   Intrepid2::FunctionSpaceTools<PHX::exec_space>::
  //     integrate<ScalarT>(residual.get_view(),
  //                        normals.get_view(), 
  //                        workset.bases[basis_index])->weighted_basis_scalar.get_view());
}


}

#endif
