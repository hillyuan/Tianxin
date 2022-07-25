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

#ifndef __TianXin_Response_Integral_Impl_hpp__
#define __TianXin_Response_Integral_Impl_hpp__

#include <string>

namespace TianXin {
	
// This elass evaluate integral alone elements or sides, e.g., heat flux along a boundary
	
// **************************************************************
// Residual
// **************************************************************
template<typename EvalT, typename Traits>
Response_Integral<EvalT,Traits>::
Response_Integral(const Teuchos::ParameterList& plist)
: ResponseBase<EvalT,Traits>(plist)
{
	Teuchos::ParameterList p(plist);
    std::string integral_name = p.get<std::string>("Integral Name");
	std::string integrand_name= p.get< std::string >("Integrand Name");
    this->response_name = "RESPONSE_" + integrand_name;
    const Teuchos::RCP<const panzer::PureBasis> basis =
      p.get< Teuchos::RCP<const panzer::PureBasis> >("Basis");
    const Teuchos::RCP<const panzer::IntegrationRule> ir = 
      p.get< Teuchos::RCP<const panzer::IntegrationRule> >("IR");
	quad_order = ir->cubature_degree;

	PHX::Layout dl_scalar("dl1",1);
	value_ = PHX::MDField<const ScalarT,panzer::Dim>(integral_name,dl_scalar);
	this->addEvaluatedField(value_);
	
	// Input : values upon cell IPs
	cellvalue_ = PHX::MDField<const ScalarT,panzer::Cell,panzer::IP>( integrand_name, ir->dl_scalar);
	this->addDependentField(cellvalue_);

	std::string n = "Integral Response " + this->response_name;
	this->setName(n);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void Response_Integral<EvalT, Traits>::
postRegistrationSetup( typename Traits::SetupData sd,
  PHX::FieldManager<Traits>& /* fm */)
{
  //basis_index = panzer::getBasisIndex(basis_name, (*sd.worksets_)[0]);
  //ir_index = panzer::getIntegrationRuleIndex(quad_order,(*sd.worksets_)[0]);
  num_cell  = cellvalue_.extent(0);
  num_qp  = cellvalue_.extent(1);
  quad_index =  panzer::getIntegrationRuleIndex(quad_order,(*sd.worksets_)[0]);
}

template<typename EvalT, typename Traits>
void Response_Integral<EvalT,Traits>::
evaluateFields(typename Traits::EvalData workset)
{
	const auto wm = this->wda(workset).int_rules[quad_index]->weighted_measure;

	value_(0) = 0.0;
	Kokkos::parallel_for("IntegratorScalar", workset.num_cells, KOKKOS_LAMBDA (int cell) {
		ScalarT cell_integral = 0.0;
		for (std::size_t qp = 0; qp < num_qp; ++qp) {
			cell_integral += cellvalue_(cell, qp)*wm(cell, qp);
		}
		value_(0) += cell_integral;
	} );
	Kokkos::fence();

	ScalarT glbValue = 0.0;
	ScalarT value = value_(0);
    Teuchos::reduceAll(*this->getComm(), Teuchos::REDUCE_SUM, static_cast<Thyra::Ordinal>(1), &value,&glbValue);
	this->getThyraVector()[0] = glbValue;
}

}

#endif
