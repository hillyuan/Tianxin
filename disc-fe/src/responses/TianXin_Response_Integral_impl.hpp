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
	
	std::string dummyName = this->response_name + " dummy target";

	// build dummy target tag
	Teuchos::RCP<PHX::DataLayout> dl_dummy = Teuchos::rcp(new PHX::MDALayout<panzer::Dummy>(0));
	scatterHolder_ = Teuchos::rcp(new PHX::Tag<ScalarT>(dummyName,dl_dummy));
	this->addEvaluatedField(*scatterHolder_);

	// build dendent field
	Teuchos::RCP<PHX::DataLayout> dl_cell = Teuchos::rcp(new PHX::MDALayout<panzer::Cell>(ir->dl_scalar->extent(0)));
	cellIntegral_ = PHX::MDField<const ScalarT,panzer::Cell>(integrand_name,dl_cell);
	this->addDependentField(cellIntegral_);

	std::string n = "Integral Response " + this->response_name;
	this->setName(n);

}

template<typename EvalT, typename Traits>
void Response_Integral<EvalT,Traits>::
evaluateFields(typename Traits::EvalData d)
{
	value = 0.0;
	for(std::size_t i=0;i<d.num_cells;i++) {
		value += cellIntegral_(i);
	}

	ScalarT glbValue = 0.0;
    Teuchos::reduceAll(*this->getComm(), Teuchos::REDUCE_SUM, static_cast<Thyra::Ordinal>(1), &value,&glbValue);
	this->getThyraVector()[0] = glbValue;
}

}

#endif
