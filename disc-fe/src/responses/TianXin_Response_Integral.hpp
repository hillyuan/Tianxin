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

#ifndef __TianXin_Response_Integral_hpp__
#define __TianXin_Response_Integral_hpp__

#include <string>

#include "Panzer_GlobalIndexer.hpp"
#include "TianXin_ResponseBase.hpp"

namespace TianXin {
	
// This elass evaluate integral alone elements or sides, e.g., heat flux along a boundary
//template<typename EvalT, typename Traits> class Response_Integral;
	
// **************************************************************
// Residual
// **************************************************************
template<typename EvalT, typename Traits>
class Response_Integral : public ResponseBase<EvalT,Traits> {
public:
	typedef typename EvalT::ScalarT ScalarT;
	using map_type = Tpetra::Map<int, panzer::GlobalOrdinal>;
	
    Response_Integral(const Teuchos::ParameterList& plist);
	 
	void postRegistrationSetup(typename Traits::SetupData d,PHX::FieldManager<Traits>& fm);
	void evaluateFields(typename Traits::EvalData d);
	
	//! provide direct access of result integral
    PHX::MDField<ScalarT> value_;
	
	virtual std::size_t localSizeRequired() const final { return 1; }
	virtual bool isDistributed() const final {return false;}

private:
	//std::string response_name;
	PHX::MDField<const ScalarT,panzer::Cell,panzer::IP> cellvalue_;
	
    // common data used by neumann calculation
    std::string basis_name;
	std::size_t num_cell, num_qp;
	int quad_order, quad_index;
	
public:
  const PHX::FieldTag & getFieldTag() const
  { return value_.fieldTag(); }

};

// **************************************************************
// Specialize: Jacobian
// **************************************************************
template<typename Traits>
class Response_Integral<panzer::Traits::Jacobian,Traits> : public ResponseBase<panzer::Traits::Jacobian,Traits> {
public:
	typedef typename panzer::Traits::Jacobian::ScalarT ScalarT;
	using map_type = Tpetra::Map<int, panzer::GlobalOrdinal>;
	
    Response_Integral(const Teuchos::ParameterList& plist);
	 
	void postRegistrationSetup(typename Traits::SetupData d,PHX::FieldManager<Traits>& fm);
	void evaluateFields(typename Traits::EvalData d);
	
	//! provide direct access of result integral
    PHX::MDField<double> value_;
	
	virtual std::size_t localSizeRequired() const final { return 1; }
	virtual bool isDistributed() const final {return false;}

private:
	//std::string response_name;
	PHX::MDField<const ScalarT,panzer::Cell,panzer::IP> cellvalue_;
	
    // common data used by neumann calculation
    std::string basis_name;
	std::size_t num_cell, num_qp;
	int quad_order, quad_index;
	
	std::vector<Teuchos::RCP<const panzer::GlobalIndexer> > ugis_;
	
public:
  const PHX::FieldTag & getFieldTag() const
  { return value_.fieldTag(); }

};

namespace ResponseRegister {
  static bool const INTEGRAL_ROK = ResponseResidualFactory::Instance().template Register< Response_Integral<panzer::Traits::Residual,panzer::Traits> >("Integral");
  static bool const INTEGRAL_JOK = ResponseJacobianFactory::Instance().template Register< Response_Integral<panzer::Traits::Jacobian,panzer::Traits> >( "Integral");
  //static bool const INTEGRAL_TOK = ResponseTangentFactory::Instance().template Register< Response_Integral<panzer::Traits::Tangent,panzer::Traits> >( "Integral");
}

}

#include "TianXin_Response_Integral_impl.hpp"

#endif
