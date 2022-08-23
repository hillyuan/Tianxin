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

#include "Teuchos_DefaultComm.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Panzer_Workset_Utilities.hpp"
#include "Panzer_GlobalIndexer_Utilities.hpp"

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
	this->integrand_name= p.get< std::string >("Integrand Name");
	std::string integral_name = p.get<std::string>("Integral Name","Integral_" +this->integrand_name);
    this->response_name = "RESPONSE_" + this->integrand_name;
    const Teuchos::RCP<const panzer::PureBasis> basis =
      p.get< Teuchos::RCP<const panzer::PureBasis> >("Basis");
    const Teuchos::RCP<const panzer::IntegrationRule> ir = 
      p.get< Teuchos::RCP<const panzer::IntegrationRule> >("IR");
	quad_order = ir->cubature_degree;

	Teuchos::RCP<PHX::DataLayout> dl_dummy = Teuchos::rcp(new PHX::MDALayout<panzer::Dim>(1));
	//Teuchos::RCP<PHX::DataLayout> dl_scalar = Teuchos::rcp(new PHX::DataLayout("dl1",1));
	this->value_ = PHX::MDField<ScalarT>(this->response_name, dl_dummy);
	this->addEvaluatedField(this->value_);
//value_.print(std::cout);
	// Input : values upon cell IPs
	num_cell = ir->dl_scalar->extent(0);
	if( num_cell>0 ) {
		cellvalue_ = PHX::MDField<const ScalarT,panzer::Cell,panzer::BASIS>( this->integrand_name, basis->functional);
		this->addDependentField(cellvalue_);
	}

	std::string n = "Integral Response " + this->response_name;
	this->setName(n);
	
	// ResponseBase related
	const std::size_t num_non_zero = 1;
	this->tMap_ = Tpetra::createLocalMap<int, panzer::GlobalOrdinal>(num_non_zero, this->tComm_);
    //this->tVector_ = Tpetra::createVector<double, int, panzer::GlobalOrdinal>(this->tMap_);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Response_Integral<EvalT, Traits>::
postRegistrationSetup( typename Traits::SetupData sd,
  PHX::FieldManager<Traits>& /* fm */)
{std::cout << num_cell << "  OK0\n";
  //basis_index = panzer::getBasisIndex(basis_name, (*sd.worksets_)[0]);
  //ir_index = panzer::getIntegrationRuleIndex(quad_order,(*sd.worksets_)[0]);
  if( num_cell>0 ) {
	num_qp  = cellvalue_.extent(1);
	quad_index =  panzer::getIntegrationRuleIndex(quad_order,(*sd.worksets_)[0]);
  }
  
}

template<typename EvalT, typename Traits>
void Response_Integral<EvalT,Traits>::
evaluateFields(typename Traits::EvalData workset)
{//std::cout << this->tComm_->getRank() << "," << workset.num_cells << "   enter\n";
	double result = 0.0;
    if( num_cell>0 ) {
		const auto wm = workset.int_rules[quad_index]->weighted_measure;
		Kokkos::parallel_reduce("IntegratorScalar", workset.num_cells, KOKKOS_LAMBDA (int cell, double& v) {
			double cell_integral = 0.0;
			for (std::size_t qp = 0; qp < num_qp; ++qp) {
				cell_integral += cellvalue_(cell, qp)*wm(cell, qp);
			}
			v += cell_integral;
		}, result );
		Kokkos::fence();
	}

	double glbValue = 0.0;
    Teuchos::reduceAll(*(this->tComm_), Teuchos::REDUCE_SUM, 1, &result, &glbValue);
	this->value_ .deep_copy(glbValue);
	if( this->tVector_==Teuchos::null ) 
		TEUCHOS_TEST_FOR_EXCEPTION(this->tVector_==Teuchos::null,std::logic_error,
                            "TianXin::Response_Integral: reponse vector not defined. "
                            "Please call setVector() before calling this method");
	Teuchos::rcp_dynamic_cast<Tpetra::Vector<double, int, panzer::GlobalOrdinal>>(this->tVector_)->sumIntoLocalValue(0, glbValue);
}

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
Response_Integral<panzer::Traits::Jacobian,Traits>::
Response_Integral(const Teuchos::ParameterList& plist)
: ResponseBase<panzer::Traits::Jacobian,Traits>(plist)
{
	Teuchos::ParameterList p(plist);
	std::string integrand_name= p.get< std::string >("Integrand Name");
	std::string integral_name = p.get<std::string>("Integral Name","Integral_" +integrand_name);
    this->response_name = "RESPONSE_" + integrand_name;
    const Teuchos::RCP<const panzer::PureBasis> basis =
      p.get< Teuchos::RCP<const panzer::PureBasis> >("Basis");
    const Teuchos::RCP<const panzer::IntegrationRule> ir = 
      p.get< Teuchos::RCP<const panzer::IntegrationRule> >("IR");
	quad_order = ir->cubature_degree;

	Teuchos::RCP<PHX::DataLayout> dl_dummy = Teuchos::rcp(new PHX::MDALayout<panzer::Dim>(1));
	//Teuchos::RCP<PHX::DataLayout> dl_scalar = Teuchos::rcp(new PHX::DataLayout("dl1",1));
	this->value_ = PHX::MDField<double>(this->response_name, dl_dummy);
	this->addEvaluatedField(this->value_);

	// Input : values upon cell IPs
	num_cell = ir->dl_scalar->extent(0);
	if( num_cell>0 ) {
		cellvalue_ = PHX::MDField<const ScalarT,panzer::Cell,panzer::IP>( integrand_name, ir->dl_scalar);
		this->addDependentField(cellvalue_);
	}

	std::string n = "Integral Response " + this->response_name;
	this->setName(n);
	
	// ResponseBase related
	const std::size_t num_non_zero = 1;
	this->tMap_ = Tpetra::createLocalMap<int, panzer::GlobalOrdinal>(num_non_zero, this->tComm_);
    //this->tVector_ = Tpetra::createVector<double, int, panzer::GlobalOrdinal>(this->tMap_);
}

//**********************************************************************
template<typename Traits>
void Response_Integral<panzer::Traits::Jacobian,Traits>::
postRegistrationSetup( typename Traits::SetupData sd,
  PHX::FieldManager<Traits>& /* fm */)
{
  //basis_index = panzer::getBasisIndex(basis_name, (*sd.worksets_)[0]);
  //ir_index = panzer::getIntegrationRuleIndex(quad_order,(*sd.worksets_)[0]);
  if( num_cell>0 ) {
	num_qp  = cellvalue_.extent(1);
	quad_index =  panzer::getIntegrationRuleIndex(quad_order,(*sd.worksets_)[0]);
  }
}

template<typename Traits>
void Response_Integral<panzer::Traits::Jacobian,Traits>::
evaluateFields(typename Traits::EvalData workset)
{
	const auto wm = workset.int_rules[quad_index]->weighted_measure;
	if( this->tVector_==Teuchos::null ) 
		TEUCHOS_TEST_FOR_EXCEPTION(this->tVector_==Teuchos::null,std::logic_error,
                            "TianXin::Response_Integral: reponse vector not defined. "
                            "Please call setVector() before calling this method");
	//TEUCHOS_ASSERT(this->tVector_.size()==ugis_.size());
	
	PHX::View<const panzer::LocalOrdinal*> LIDs;
	// for convenience pull out some objects from workset
    std::string blockId = workset.block_id;
    std::vector<int> blockOffsets;
    computeBlockOffsets(blockId,ugis_,blockOffsets);
	
	const std::vector<std::size_t> & localCellIds = workset.cell_local_ids;
	std::size_t worksetCellIndex=-1;
	for( const std::size_t cellLocalId : localCellIds ) {
		++worksetCellIndex;
		for(std::size_t b=0;b<ugis_.size();b++) {
			int start = blockOffsets[b];
			LIDs = ugis_[b]->getElementLIDs(cellLocalId); 
			// loop over basis functions
			for(std::size_t i=0;i<LIDs.size();i++) {
				this->tVector_->sumIntoLocalValue(LIDs[i], 0, 100.0);
			//	this->tVector_->sumIntoGlobalValue(0,LIDs[i],cellvalue_(++worksetCellIndex).dx(start+i));
			//	dgdx_b[LIDs[i]] += cellIntegral(worksetCellIndex).dx(start+i); // its possible functional is independent of solution value!
			}
		}
	}

	double result = 0.0;
	Kokkos::parallel_reduce("IntegratorScalar", workset.num_cells, KOKKOS_LAMBDA (int cell, double& v) {
		ScalarT cell_integral = 0.0;
		for (std::size_t qp = 0; qp < num_qp; ++qp) {
			cell_integral += cellvalue_(cell, qp)*wm(cell, qp);
		}
		v += cell_integral.val();
	}, result );
	Kokkos::fence();

	double glbValue = 0.0;
    Teuchos::reduceAll<int,double>(*(this->tComm_), Teuchos::REDUCE_SUM, static_cast<Thyra::Ordinal>(1), &result,&glbValue);
	this->value_ .deep_copy(glbValue);
	
	//Teuchos::rcp_dynamic_cast<Tpetra::Vector<double, int, panzer::GlobalOrdinal>>(this->tVector_)->sumIntoLocalValue(0, glbValue);
}

}

#endif
