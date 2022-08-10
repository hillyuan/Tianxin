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

#ifndef __TianXin_ResponseBase_hpp__
#define __TianXin_ResponseBase_hpp__

#include <string>

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_Field.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_DefaultComm.hpp"

#include "Tpetra_Map.hpp"
#include "Tpetra_Vector.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"

#include "Panzer_Traits.hpp"

#include "TianXin_WorksetFunctor.hpp"
#include "TianXin_Factory.hpp"
#include "TianXin_TemplateTypeContainer.hpp"

namespace TianXin {
	
/* Non-templated Response class 
   Usge: 0. Prepare: set tMap_ in this class constructor
         1. when setting OutArgs: construct vector from map given by getMap()
         2. set above vector to current class by setVector
*/
struct Response {
	using vector_type = Tpetra::MultiVector<double, int, panzer::GlobalOrdinal>;
	virtual Teuchos::RCP<const Thyra::VectorSpaceBase<double> > getVectorSpace() const = 0;
	virtual Teuchos::RCP< const Tpetra::Map<int> > getMap() const =0;
	virtual void setVector(const Teuchos::RCP<vector_type > & destVec) = 0;
	virtual Teuchos::RCP<vector_type> getVector() const=0;
};


typedef TemplateTypeContainer<panzer::Traits::EvalTypes,Teuchos::RCP<Response> > TemplatedResponse;

// **************************************************************
// Residual
// **************************************************************
template<typename EvalT, typename Traits>
class ResponseBase : public Response, public PHX::EvaluatorWithBaseImpl<Traits> {
	
	using map_type = Tpetra::Map<int, panzer::GlobalOrdinal>;
	using vector_type = Tpetra::MultiVector<double, int, panzer::GlobalOrdinal>;
public:
   ResponseBase(const Teuchos::ParameterList& p)
   : tComm_(Teuchos::DefaultComm<int>::getComm())
   {}
	 
   virtual void evaluateFields(typename Traits::EvalData d)=0;
	 
	 
   /** Get the unmodified name for this response.
     */
   std::string getResponseName() const
	{
		return response_name;
	}
	
	/* number of response items*/
	virtual std::size_t localSizeRequired() const =0;
	
	virtual bool isDistributed() const =0;
	
   //! Get the vector space for this response, vector space is constructed lazily.
   Teuchos::RCP< const Tpetra::Map<int> > getMap() const final {
     return tMap_;
   }
   
   Teuchos::RCP<const Thyra::VectorSpaceBase<double> > getVectorSpace() const final {
	   return Thyra::createVectorSpace<double, int, panzer::GlobalOrdinal>( tMap_ );
   }
   
   //! Access the thyra MultiVector
    Teuchos::RCP<vector_type> getVector() const final
    { return tVector_; }
	
/*	void setVector(const Teuchos::RCP<Thyra::VectorBase<double> > destVec) final {
		tVector_ = Thyra::createVector();
	}*/
	
	void setVector(const Teuchos::RCP<vector_type> & destVec) final {
		tVector_ = destVec;
		TEUCHOS_TEST_FOR_EXCEPTION(this->tVector_==Teuchos::null,std::logic_error,
                            "TianXin::setVector: reponse vector not defined. "
                            "Please call setVector() before calling this method");
	}

protected:
   std::string response_name;
   Teuchos::RCP<const Teuchos::Comm<int> > tComm_;
   Teuchos::RCP<const map_type >  tMap_;
   Teuchos::RCP<vector_type > tVector_;
   
public:
   virtual const PHX::FieldTag & getFieldTag() const = 0;
};

typedef Factory<ResponseBase<panzer::Traits::Residual,panzer::Traits>,std::string,Teuchos::ParameterList> ResponseResidualFactory;
typedef Factory<ResponseBase<panzer::Traits::Jacobian,panzer::Traits>,std::string,Teuchos::ParameterList> ResponseJacobianFactory;
typedef Factory<ResponseBase<panzer::Traits::Tangent,panzer::Traits>,std::string,Teuchos::ParameterList> ResponseTangentFactory;

}

#endif
